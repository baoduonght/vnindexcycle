import pandas as pd
import numpy as np
import requests
try:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()
except Exception:
    tv = None
    Interval = None
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import warnings
import logging
import sys
import os
import datetime

# Nạp biến môi trường từ file .env nếu có (dành cho chạy cục bộ)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vn30_universe import ALL_HISTORICAL_VN30_TICKERS, get_vn30_constituents

warnings.filterwarnings('ignore')
logging.getLogger("tvDatafeed").setLevel(logging.CRITICAL)

# ==========================================
# CÁC HÀM TÍNH TOÁN CHỈ BÁO KỸ THUẬT (ĐỒNG BỘ 100% VỚI QUANT_BOT.PY)
# ==========================================
def calc_adx(high, low, close, n=14):
    up = high - high.shift()
    down = low.shift() - low
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    ndm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr = pd.Series(tr, index=close.index).rolling(n).mean().replace(0, np.nan)
    pdi = 100.0 * pd.Series(pdm, index=close.index).rolling(n).mean() / atr
    ndi = 100.0 * pd.Series(ndm, index=close.index).rolling(n).mean() / atr
    di_sum = (pdi + ndi).replace(0, np.nan)
    dx = 100.0 * abs(pdi - ndi) / di_sum
    return dx.rolling(n).mean()

def calc_atr(high, low, close, n=14):
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    return pd.Series(tr, index=close.index).rolling(n).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1.0 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where((ema_down == 0) & (ema_up > 0), 100.0, rsi)
    rsi = np.where((ema_down == 0) & (ema_up == 0), 50.0, rsi)
    return pd.Series(rsi, index=series.index)

def calc_mfi(high, low, close, volume, n=14):
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    diff = tp.diff()
    pos_mf = rmf.where(diff > 0, 0.0).rolling(n).sum()
    neg_mf = rmf.where(diff < 0, 0.0).rolling(n).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    mfi = np.where((neg_mf == 0) & (pos_mf > 0), 100.0, mfi)
    mfi = np.where((neg_mf == 0) & (pos_mf == 0), 50.0, mfi)
    return pd.Series(mfi, index=close.index)

def calc_macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal

# ==========================================
# GIẢI PHÁP 1: TIME-DECAYED SAMPLE WEIGHTING
# ==========================================
def compute_time_decay_weights(dates_series, ref_date):
    """
    Tính trọng số thời gian suy giảm (Time-Decayed Sample Weights):
    - 3 năm gần nhất so với ref_date: weight = 1.0
    - 4 - 7 năm: weight = 0.6
    - > 7 năm: weight = 0.3
    """
    ref_dt = pd.to_datetime(ref_date)
    dt_series = pd.to_datetime(dates_series)
    diff_years = (ref_dt - dt_series).dt.total_seconds() / (365.25 * 86400.0)

    weights = np.where(
        diff_years <= 3.0, 1.0,
        np.where(diff_years <= 7.0, 0.6, 0.3)
    )
    return weights

# ==========================================
# HÀM TẢI DỮ LIỆU ĐA NGUỒN (ĐỒNG BỘ CACHE LOCAL DATA/ + 4 API)
# ==========================================
def _clean_ohlcv_df(df, quality, source):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = df['Close'] if col != 'Volume' else 0.0
    df['Volume'] = df['Volume'].fillna(0.0)
    df = df.dropna(subset=['Date', 'Close'])
    df = df[df['Close'] > 0]
    for col in ['Open', 'High', 'Low']:
        df[col] = df[col].where(df[col] > 0, df['Close'])
    df['Data_Quality'] = quality
    df['Data_Source'] = source
    return (
        df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Data_Quality', 'Data_Source']]
        .drop_duplicates(subset=['Date'], keep='last')
        .sort_values('Date')
        .reset_index(drop=True)
    )

def fetch_ticker_data(symbol, n_bars=4000):
    now_ts = int(datetime.datetime.now().timestamp())
    start_ts = int(datetime.datetime(2012, 1, 1).timestamp())

    # 0. Nguồn 0: File CSV cục bộ từ thư mục data/ (đồng bộ 100% với quant_bot.py)
    local_csv = os.path.join("data", f"{symbol}.csv")
    if os.path.exists(local_csv):
        try:
            loc_df = pd.read_csv(local_csv)
            if 'Date' in loc_df.columns and 'Close' in loc_df.columns and len(loc_df) > 50:
                cleaned = _clean_ohlcv_df(loc_df, 'REAL_OHLC', 'LOCAL_DATA_DIR')
                if len(cleaned) > 50:
                    return cleaned
        except Exception:
            pass

    is_index = symbol.upper() in ['VNINDEX', 'VN30', 'VN100', 'VNXALL', 'HNX', 'HNX30', 'UPCOM']
    entrade_endpoint = 'index' if is_index else 'stock'

    # 1. Nguồn 1: DNSE Entrade API
    try:
        url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{entrade_endpoint}?from={start_ts}&to={now_ts}&symbol={symbol}&resolution=1D"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=6)
        if res.status_code == 200:
            data = res.json()
            if 't' in data and len(data['t']) > 50:
                df = pd.DataFrame({
                    'Date': pd.to_datetime(data['t'], unit='s'),
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data.get('v', [0] * len(data['t']))
                })
                cleaned = _clean_ohlcv_df(df, 'REAL_OHLC', 'DNSE_ENTRADE')
                if len(cleaned) > 50:
                    return cleaned
    except Exception:
        pass

    # 2. Nguồn 2: VNDirect DChart API
    try:
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={start_ts}&to={now_ts}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=6)
        if res.status_code == 200:
            data = res.json()
            if data.get('s') == 'ok' and len(data.get('t', [])) > 50:
                df = pd.DataFrame({
                    'Date': pd.to_datetime(data['t'], unit='s'),
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data.get('v', [0] * len(data['t']))
                })
                cleaned = _clean_ohlcv_df(df, 'REAL_OHLC', 'VNDIRECT')
                if len(cleaned) > 50:
                    return cleaned
    except Exception:
        pass

    # 3. Nguồn 3: TCBS Public API
    try:
        api_type = 'index' if is_index else 'stock'
        url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={symbol}&type={api_type}&resolution=D&from={start_ts}&to={now_ts}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=6)
        if res.status_code == 200:
            data = res.json().get('data', [])
            if len(data) > 50:
                df = pd.DataFrame(data)
                df.rename(columns={'tradingDate': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                cleaned = _clean_ohlcv_df(df, 'REAL_OHLC', 'TCBS')
                if len(cleaned) > 50:
                    return cleaned
    except Exception:
        pass

    # 4. Nguồn 4: TradingView
    if tv is not None and Interval is not None:
        try:
            df = tv.get_hist(symbol=symbol, exchange='HOSE', interval=Interval.in_daily, n_bars=n_bars)
            if df is not None and len(df) > 50:
                df = df.reset_index()
                df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                cleaned = _clean_ohlcv_df(df, 'REAL_OHLC', 'TRADINGVIEW')
                if len(cleaned) > 50:
                    return cleaned
        except Exception:
            pass

    # Fallback cuối cùng nếu mất mạng: CSV
    if symbol.upper() == 'VNINDEX' and os.path.exists('markets_database10years.csv'):
        try:
            print("   ⚠️ CẢNH BÁO: Tải API thất bại, đọc VNINDEX từ markets_database10years.csv!")
            csv_df = pd.read_csv('markets_database10years.csv')
            if 'VNINDEX' in csv_df.columns:
                c_df = csv_df[['Date', 'VNINDEX', 'VN_VOLUME']].dropna(subset=['VNINDEX']).copy()
                c_df['Date'] = pd.to_datetime(c_df['Date'], errors='coerce').dt.tz_localize(None).dt.normalize()
                c_df.rename(columns={'VNINDEX': 'Close', 'VN_VOLUME': 'Volume'}, inplace=True)
                c_df['Close'] = pd.to_numeric(c_df['Close'], errors='coerce')
                c_df['Volume'] = pd.to_numeric(c_df['Volume'], errors='coerce').fillna(0.0)
                c_df = c_df.dropna(subset=['Date', 'Close']).drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)

                daily_vol = c_df['Close'].pct_change().abs().rolling(20, min_periods=5).mean().fillna(0.012)
                c_df['High'] = c_df['Close'] * (1.0 + 0.6 * daily_vol)
                c_df['Low'] = c_df['Close'] * (1.0 - 0.6 * daily_vol)
                c_df['Open'] = c_df['Close'].shift(1).fillna(c_df['Close'])
                print("   ℹ️ Đã gắn nhãn 'SYNTHETIC_CLOSE_ONLY' và ước lượng True Range cho VN-Index (tránh phẳng nến).")
                return _clean_ohlcv_df(c_df, 'SYNTHETIC_CLOSE_ONLY', 'MARKETS_DB_CSV')
        except Exception:
            pass

    return None


def main():
    # ==========================================
    # 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG (OHLCV THẬT)
    # ==========================================
    print("1. Đang tải Market Proxy (VNINDEX)...")
    vnindex_raw = fetch_ticker_data('VNINDEX', n_bars=4500)
    if vnindex_raw is None or vnindex_raw.empty:
        sys.exit("❌ LỖI: Không thể tải dữ liệu VNINDEX.")

    vnindex_df = (
        vnindex_raw.rename(columns={'Close': 'VNINDEX'})
        .drop_duplicates(subset=['Date'], keep='last')
        .sort_values('Date')
        .reset_index(drop=True)
    )
    vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50, min_periods=20).mean()
    vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100, min_periods=30).mean()
    vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200, min_periods=40).mean()
    vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
    vnindex_df['ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14) / vnindex_df['VNINDEX']
    vnindex_df['VN_ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 20) / vnindex_df['VNINDEX']
    vnindex_df['VN_MFI'] = calc_mfi(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], vnindex_df['Volume'], 14)
    df_clean = vnindex_df.dropna().drop_duplicates(subset=['Date'], keep='last').copy().reset_index(drop=True)

    # ==========================================
    # 2. TẢI TOÀN BỘ RỔ CỔ PHIẾU VN30 LỊCH SỬ (POINT-IN-TIME)
    # ==========================================
    print(f"2. Đang tải dữ liệu {len(ALL_HISTORICAL_VN30_TICKERS)} mã VN30 lịch sử để AI học tập toàn diện...")

    timeline_dates = pd.DatetimeIndex(df_clean['Date'].drop_duplicates().sort_values())
    close_cols, vol_cols = {}, {}
    loaded_tickers = []

    for ticker in ALL_HISTORICAL_VN30_TICKERS:
        df_t = fetch_ticker_data(ticker, n_bars=4000)
        if df_t is not None and not df_t.empty:
            df_t = df_t.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
            df_t.set_index('Date', inplace=True)
            df_t = df_t[~df_t.index.duplicated(keep='last')]
            close_cols[ticker] = df_t['Close']
            vol_cols[f'{ticker}_V'] = df_t['Volume']
            loaded_tickers.append(ticker)

    stocks_close = pd.DataFrame(close_cols, index=timeline_dates)
    stocks_volume = pd.DataFrame(vol_cols, index=timeline_dates)

    print(f"   ✅ Đã nạp thành công {len(loaded_tickers)} mã vào tập huấn luyện.")

    # KHÔNG DÙNG LỆNH FILL: Tính chỉ báo trực tiếp trên dữ liệu giao dịch thực tế
    ma50_cols, vol_ma20_cols, rsi_cols, macd_cols = {}, {}, {}, {}

    for c in loaded_tickers:
        s_c = stocks_close[c].dropna()
        if len(s_c) >= 20:
            ma50_cols[c] = s_c.rolling(50, min_periods=20).mean()
            rsi_cols[c] = calc_rsi(s_c)
            macd_cols[c] = calc_macd_hist(s_c)
        if f'{c}_V' in stocks_volume.columns:
            s_v = stocks_volume[f'{c}_V'].dropna()
            if len(s_v) >= 20:
                vol_ma20_cols[f'{c}_V'] = s_v.rolling(20).mean().shift(1)

    stocks_ma50 = pd.DataFrame(ma50_cols, index=stocks_close.index)
    stocks_vol_ma20 = pd.DataFrame(vol_ma20_cols, index=stocks_close.index)
    stocks_rsi = pd.DataFrame(rsi_cols, index=stocks_close.index)
    stocks_macd = pd.DataFrame(macd_cols, index=stocks_close.index)

    # =====================================================================
    # TÍNH ĐỘ RỘNG THỊ TRƯỜNG POINT-IN-TIME (TỶ LỆ MÃ VN30 THỰC TẾ TRÊN MA50)
    # =====================================================================
    breadth_map = {}
    for dt in stocks_close.index:
        d_str = dt.strftime('%Y-%m-%d')
        active_vn30 = get_vn30_constituents(d_str, use_live=False)
        above_cnt = 0
        valid_cnt = 0
        for ticker in active_vn30:
            if ticker in stocks_close.columns and ticker in stocks_ma50.columns:
                c_val = stocks_close.at[dt, ticker]
                m_val = stocks_ma50.at[dt, ticker]
                if pd.notna(c_val) and pd.notna(m_val) and c_val > 0 and m_val > 0:
                    valid_cnt += 1
                    if c_val > m_val:
                        above_cnt += 1
        breadth_map[d_str] = (above_cnt / valid_cnt) if valid_cnt > 0 else 0.5

    # ==========================================
    # 3. CHUẨN BỊ TẬP DỮ LIỆU ĐỂ AI HỌC (TRAIN DATA)
    # ==========================================
    print("3. Khởi tạo màng lưới tính toán & Đồng bộ Vĩ mô + Độ rộng...")
    try:
        macro_df = pd.read_csv('markets_database10years.csv')
        macro_df = macro_df[['Date', 'DXY', 'US10Y', 'USD_VND']].copy()
        macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
        macro_df = macro_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
    except Exception as e:
        sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

    macro_df['USD_VND'] = pd.to_numeric(macro_df['USD_VND'], errors='coerce')
    macro_df['USD_VND'] = macro_df['USD_VND'].apply(lambda x: x * 1000.0 if (pd.notna(x) and 10.0 <= x <= 40.0) else x)
    macro_df['USD_VND'] = macro_df['USD_VND'].apply(lambda x: np.nan if (pd.notna(x) and (x < 15000.0 or x > 35000.0)) else x)
    macro_df['USD_VND'] = macro_df['USD_VND'].ffill()

    # Publication Lag shift(1) chống Look-ahead Bias
    macro_df['DXY'] = pd.to_numeric(macro_df['DXY'], errors='coerce').shift(1)
    macro_df['US10Y'] = pd.to_numeric(macro_df['US10Y'], errors='coerce').shift(1)
    macro_df['USD_VND'] = macro_df['USD_VND'].shift(1)

    market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200', 'VN_ATR_PCT']].copy()
    market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']
    market_context['Date'] = pd.to_datetime(market_context['Date']).dt.strftime('%Y-%m-%d')
    market_context['VN_Breadth_MA50'] = market_context['Date'].map(breadth_map).fillna(0.5)
    market_context = market_context.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)

    macro_df = macro_df.merge(
        market_context[['Date', 'VN_MFI', 'VN_Dist_MA200', 'VN_ATR_PCT', 'VN_Breadth_MA50']],
        on='Date',
        how='inner'
    )
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)

    macro_cols = ['DXY', 'US10Y', 'USD_VND']
    for col in macro_cols:
        macro_df[col] = pd.to_numeric(macro_df[col], errors='coerce')
        roll_m = macro_df[col].rolling(120, min_periods=5).mean()
        roll_s = macro_df[col].rolling(120, min_periods=5).std().replace(0, np.nan)
        macro_df[f'Z_{col}'] = (macro_df[col] - roll_m) / roll_s
    z_cols = [f'Z_{c}' for c in macro_cols]

    vn_close_aligned = vnindex_df.set_index('Date')['VNINDEX'].reindex(stocks_close.index)

    df_ml_list = []
    skipped_gap_samples = 0

    for c in loaded_tickers:
        s_close = stocks_close[c]
        s_vol = stocks_volume[f'{c}_V']
        s_vol_ma = stocks_vol_ma20[f'{c}_V']
        s_ma50 = stocks_ma50[c]

        valid_mask = s_close.notna() & (s_close > 0)
        if valid_mask.sum() < 30:
            continue

        # Dữ liệu nguyên bản 100% (RAW DATA, TUYỆT ĐỐI KHÔNG DÙNG LỆNH FILL)
        s_valid_close = s_close[valid_mask]
        s_future_ret = s_valid_close.shift(-30) / s_valid_close - 1.0

        vn_aligned_c = vn_close_aligned.reindex(s_valid_close.index)
        vn_future_ret = vn_aligned_c.shift(-30) / vn_aligned_c - 1.0

        # Ngày kết thúc của tương lai 30 phiên để Purged Walk-Forward & kiểm tra đứt quãng đình chỉ
        curr_dates = pd.Series(s_valid_close.index, index=s_valid_close.index)
        future_dates = curr_dates.shift(-30)
        calendar_span_days = (future_dates - curr_dates).dt.days

        # Chặn nối nhãn qua đợt đình chỉ giao dịch kéo dài (> 65 ngày lịch cho 30 phiên)
        valid_span = calendar_span_days.notna() & (calendar_span_days <= 65)
        skipped_gap_samples += int((future_dates.notna() & ~valid_span).sum())

        valid_future = s_future_ret.notna() & vn_future_ret.notna() & valid_span
        target_series = np.where(valid_future, (s_future_ret > vn_future_ret).astype(int), np.nan)

        rsi_valid = stocks_rsi[c].reindex(s_valid_close.index)

        df_c = pd.DataFrame({
            'Date': pd.to_datetime(s_valid_close.index).strftime('%Y-%m-%d'),
            'Future_Date': pd.to_datetime(future_dates).dt.strftime('%Y-%m-%d'),
            'Ticker': c,
            'Vol_Ratio': (s_vol[valid_mask] / s_vol_ma[valid_mask].replace(0, np.nan)).values,
            'Dist_MA50': (s_valid_close / s_ma50[valid_mask]).values,
            'RSI': rsi_valid.values,
            'RSI_Momentum': rsi_valid.diff(3).values,
            'MACD_Hist': stocks_macd[c].reindex(s_valid_close.index).values,
            'Target': target_series
        })
        df_ml_list.append(df_c)

    if skipped_gap_samples > 0:
        print(f"   🧹 Đã loại bỏ {skipped_gap_samples:,} mẫu nhãn bị đứt quãng đình chỉ giao dịch (> 65 ngày lịch).")

    ml_master = pd.concat(df_ml_list, ignore_index=True).merge(macro_df, on='Date', how='left')
    ml_features = [
        'Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum',
        'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200', 'VN_Breadth_MA50', 'VN_ATR_PCT'
    ] + z_cols
    for f in ml_features:
        ml_master[f] = pd.to_numeric(ml_master[f], errors='coerce')

    # ==========================================
    # 4. HUẤN LUYỆN WALK-FORWARD RETRAINING (TIME-DECAYED WEIGHTS)
    # ==========================================
    print("4. 🔄 KHỞI ĐỘNG LÒ LUYỆN ĐAN: HUẤN LUYỆN CUỐN CHIẾU WALK-FORWARD (TIME-DECAYED ROLLING)...")

    train_data = ml_master.dropna(subset=['Target']).copy()
    train_data['Target'] = train_data['Target'].astype(int)

    models_saved = {}
    print(f"   📊 Tổng số mẫu dữ liệu trích xuất hợp lệ: {len(train_data):,}")

    def build_xgb(scale_pos_w):
        return xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_w,
            random_state=42,
            eval_metric='logloss',
            missing=np.nan
        )

    # 1. HUẤN LUYỆN NÃO BỘ CƠ SỞ: Học toàn diện 10 năm chu kỳ vàng (2012 -> 2021)
    print("\n" + "=" * 65)
    print("🧠 HUẤN LUYỆN NÃO BỘ CƠ SỞ CHU KỲ VÀNG: 2012 -> 2021 (VỚI TIME DECAY)")
    print("   (Purged Walk-Forward: Khóa rò rỉ nhãn tương lai trước 01/01/2022)")
    print("=" * 65)
    base_train_mask = (train_data['Date'] >= "2012-01-01") & (train_data['Future_Date'] < "2022-01-01")
    X_base = train_data.loc[base_train_mask, ml_features]
    y_base = train_data.loc[base_train_mask, 'Target']
    scale_base = len(y_base[y_base == 0]) / len(y_base[y_base == 1]) if len(y_base[y_base == 1]) > 0 else 1.0
    base_weights = compute_time_decay_weights(train_data.loc[base_train_mask, 'Date'], "2022-01-01")

    base_model = build_xgb(scale_base)
    print(f"   ⚙️ Đang huấn luyện trên toàn bộ {base_train_mask.sum():,} mẫu (2012 -> 2021) với Time-Decayed Weights...")
    base_model.fit(X_base, y_base, sample_weight=base_weights)

    # Giữ nguyên xgboost_ultimate.json là mô hình cơ sở (2012 -> 2021) để an toàn khi backtest từ quá khứ
    base_model.save_model("xgboost_model_2022.json")
    base_model.save_model("xgboost_ultimate.json")
    models_saved[2022] = base_model
    print("   ✅ Đã hoàn tất và lưu: xgboost_model_2022.json & xgboost_ultimate.json (Base 2012-2021)")

    # 2. HUẤN LUYỆN TIẾP CÁC NĂM OUT-OF-SAMPLE: 2023, 2024, 2025, 2026 (EXPANDING TỪ 2012)
    for year in [2023, 2024, 2025, 2026]:
        start_train_date = "2012-01-01"
        end_train_date = f"{year}-01-01"

        train_mask = (train_data['Date'] >= start_train_date) & (train_data['Future_Date'] < end_train_date)
        test_mask = (train_data['Date'] >= end_train_date) & (train_data['Date'] < f"{year + 1}-01-01")

        X_train = train_data.loc[train_mask, ml_features]
        y_train = train_data.loc[train_mask, 'Target']
        scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1.0
        sample_weights = compute_time_decay_weights(train_data.loc[train_mask, 'Date'], end_train_date)

        model = build_xgb(scale_weight)
        print(f"\n   ⚙️ Huấn luyện Expanding Model cho năm {year} (Học từ {start_train_date} -> {end_train_date}, {train_mask.sum():,} mẫu)...")
        model.fit(X_train, y_train, sample_weight=sample_weights)

        if test_mask.sum() > 0:
            X_test = train_data.loc[test_mask, ml_features]
            y_test = train_data.loc[test_mask, 'Target']
            if len(y_test.unique()) > 1:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred_50 = (y_pred_proba >= 0.50).astype(int)
                p80_val = float(np.percentile(y_pred_proba, 80))
                y_pred_top20 = (y_pred_proba >= p80_val).astype(int)

                auc = roc_auc_score(y_test, y_pred_proba)
                prec_50 = precision_score(y_test, y_pred_50, zero_division=0)
                rec_50 = recall_score(y_test, y_pred_50, zero_division=0)
                prec_top20 = precision_score(y_test, y_pred_top20, zero_division=0)
                print(
                    f"      🎯 [OOS {year}] AUC: {auc:.4f} | Prec@0.5: {prec_50:.4f} | Rec@0.5: {rec_50:.4f} "
                    f"| Prec@Top20% (P80={p80_val:.3f}): {prec_top20:.4f} (Mẫu: {test_mask.sum():,})"
                )

        model_filename = f"xgboost_model_{year}.json"
        model.save_model(model_filename)
        models_saved[year] = model

    # 3. HUẤN LUYỆN MÔ HÌNH LIVE MỚI NHẤT (HỌC ĐẾN DỮ LIỆU ĐÃ ĐỦ 30 PHIÊN GẦN NHẤT)
    latest_valid_date = train_data['Date'].max()
    live_train_mask = train_data['Date'] >= "2012-01-01"
    X_live = train_data.loc[live_train_mask, ml_features]
    y_live = train_data.loc[live_train_mask, 'Target']
    scale_live = len(y_live[y_live == 0]) / len(y_live[y_live == 1]) if len(y_live[y_live == 1]) > 0 else 1.0
    live_weights = compute_time_decay_weights(train_data.loc[live_train_mask, 'Date'], latest_valid_date)

    live_model = build_xgb(scale_live)
    live_model.fit(X_live, y_live, sample_weight=live_weights)
    LIVE_MODEL_FILE = "xgboost_live_latest.json"
    live_model.save_model(LIVE_MODEL_FILE)

    print(f"\n✅ Đã hoàn tất huấn luyện Rolling Walk-Forward cho các năm {list(models_saved.keys())}!")
    print(f"💾 Mô hình cơ sở (2012-2021): xgboost_ultimate.json | Mô hình Live (đến {latest_valid_date}): {LIVE_MODEL_FILE}")

    importances = pd.Series(live_model.feature_importances_, index=ml_features).sort_values(ascending=False)
    print("\n🏆 BẢNG XẾP HẠNG TẦM QUAN TRỌNG CỦA CÁC BIẾN (FEATURE IMPORTANCES - LATEST LIVE):")
    for feat, imp in importances.items():
        print(f"   • {feat:<18}: {imp*100:.2f}%")

if __name__ == "__main__":
    main()
