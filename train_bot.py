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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
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
# CÁC HÀM TÍNH TOÁN CHỈ BÁO KỸ THUẬT
# ==========================================
def calc_adx(high, low, close, n=14):
    up = high - high.shift()
    down = low.shift() - low
    pdm = np.where((up > down) & (up > 0), up, 0)
    ndm = np.where((down > up) & (down > 0), down, 0)
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr = pd.Series(tr, index=close.index).rolling(n).mean()
    pdi = 100 * pd.Series(pdm, index=close.index).rolling(n).mean() / atr
    ndi = 100 * pd.Series(ndm, index=close.index).rolling(n).mean() / atr
    dx = 100 * abs(pdi - ndi) / (pdi + ndi)
    return dx.rolling(n).mean()

def calc_atr(high, low, close, n=14):
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    return pd.Series(tr, index=close.index).rolling(n).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calc_mfi(high, low, close, volume, n=14):
    tp = (high + low + close) / 3
    rmf = tp * volume
    diff = tp.diff()
    pos_mf = rmf.where(diff > 0, 0.0).rolling(n).sum()
    neg_mf = rmf.where(diff < 0, 0.0).rolling(n).sum()
    return 100 - (100 / (1 + (pos_mf / neg_mf)))

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
    - 3 năm gần nhất so với ref_date: weight = 1.0 (trọng số cao nhất để nắm bắt cấu trúc thanh khoản, hành vi giao dịch hiện đại)
    - 4 - 7 năm: weight = 0.6
    - > 7 năm: weight = 0.3 (giữ lại ký ức khủng hoảng 2012, 2018 nhưng không để nó lấn át hành vi thị trường mới)
    """
    ref_dt = pd.to_datetime(ref_date)
    dt_series = pd.to_datetime(dates_series)
    diff_years = (ref_dt - dt_series).dt.total_seconds() / (365.25 * 86400.0)
    
    weights = np.where(diff_years <= 3.0, 1.0,
              np.where(diff_years <= 7.0, 0.6, 0.3))
    return weights

# ==========================================
# HÀM TẢI DỮ LIỆU ĐA NGUỒN (DNSE + VNDIRECT + TCBS + TRADINGVIEW)
# ==========================================
def fetch_ticker_data(symbol, n_bars=4000):
    """
    Tải dữ liệu chứng khoán OHLCV THẬT từ API:
    - VNINDEX & các chỉ số: DNSE Entrade /ohlcs/index (Open, High, Low, Close, Volume thật)
    - Cổ phiếu VN30: DNSE Entrade /ohlcs/stock
    *** KHÔNG làm phẳng nến (Open=High=Low=Close) — ATR, ADX, MFI tính trên OHLCV thật ***
    """
    now_ts = int(datetime.datetime.now().timestamp())
    # Lấy dữ liệu từ đầu năm 2012 để có đầy đủ nến trước 2014, giúp MA200/MA100/ATR sẵn sàng
    start_ts = int(datetime.datetime(2012, 1, 1).timestamp())

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
                    'Date': pd.to_datetime(data['t'], unit='s').tz_localize(None).normalize(),
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data.get('v', [0]*len(data['t'])),
                    'Data_Quality': 'REAL_OHLC',
                    'Data_Source': 'DNSE_ENTRADE'
                })
                return df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
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
                    'Date': pd.to_datetime(data['t'], unit='s').tz_localize(None).normalize(),
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data.get('v', [0]*len(data['t'])),
                    'Data_Quality': 'REAL_OHLC',
                    'Data_Source': 'VNDIRECT'
                })
                return df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
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
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
                df['Data_Quality'] = 'REAL_OHLC'
                df['Data_Source'] = 'TCBS'
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Data_Quality', 'Data_Source']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    # 4. Nguồn 4: TradingView
    if tv is not None and Interval is not None:
        try:
            df = tv.get_hist(symbol=symbol, exchange='HOSE', interval=Interval.in_daily, n_bars=n_bars)
            if df is not None and len(df) > 50:
                df = df.reset_index()
                df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
                df['Data_Quality'] = 'REAL_OHLC'
                df['Data_Source'] = 'TRADINGVIEW'
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Data_Quality', 'Data_Source']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
        except Exception:
            pass

    # Fallback cuối cùng nếu mất mạng: CSV
    if symbol.upper() == 'VNINDEX' and os.path.exists('markets_database10years.csv'):
        try:
            print("   ⚠️ CẢNH BÁO: Tải API thất bại, đọc VNINDEX từ markets_database10years.csv!")
            csv_df = pd.read_csv('markets_database10years.csv')
            if 'VNINDEX' in csv_df.columns:
                c_df = csv_df[['Date', 'VNINDEX', 'VN_VOLUME']].dropna(subset=['VNINDEX']).copy()
                c_df['Date'] = pd.to_datetime(c_df['Date']).dt.tz_localize(None).dt.normalize()
                c_df.rename(columns={'VNINDEX': 'Close', 'VN_VOLUME': 'Volume'}, inplace=True)
                # BẮT BUỘC: Sắp xếp Date tăng dần trước mọi phép tính thời gian (tránh nhìn trước tương lai khi file gốc xếp giảm dần)
                c_df = c_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
                
                # Tái tạo biên độ dao động nến dựa trên độ biến động lịch sử để không làm phẳng nến
                daily_vol = c_df['Close'].pct_change().abs().rolling(20, min_periods=5).mean().fillna(0.012)
                c_df['High'] = c_df['Close'] * (1.0 + 0.6 * daily_vol)
                c_df['Low'] = c_df['Close'] * (1.0 - 0.6 * daily_vol)
                c_df['Open'] = c_df['Close'].shift(1).fillna(c_df['Close'])
                c_df['Data_Quality'] = 'SYNTHETIC_CLOSE_ONLY'
                c_df['Data_Source'] = 'MARKETS_DB_CSV'
                print("   ℹ️ Đã gắn nhãn 'SYNTHETIC_CLOSE_ONLY' và ước lượng True Range cho VN-Index (tránh phẳng nến).")
                return c_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Data_Quality', 'Data_Source']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
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

    vnindex_df = vnindex_raw.rename(columns={'Close': 'VNINDEX'}).drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
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

    stocks_close = pd.DataFrame(index=timeline_dates)
    stocks_volume = pd.DataFrame(index=timeline_dates)

    loaded_tickers = []
    for ticker in ALL_HISTORICAL_VN30_TICKERS:
        df_t = fetch_ticker_data(ticker, n_bars=4000)
        if df_t is not None and not df_t.empty:
            df_t = df_t.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
            df_t.set_index('Date', inplace=True)
            df_t = df_t[~df_t.index.duplicated(keep='last')]
            stocks_close[ticker] = df_t['Close']
            stocks_volume[f'{ticker}_V'] = df_t['Volume']
            loaded_tickers.append(ticker)

    print(f"   ✅ Đã nạp thành công {len(loaded_tickers)} mã vào tập huấn luyện.")

    # KHÔNG DÙNG LỆNH FILL: Tính chỉ báo trực tiếp trên dữ liệu giao dịch thực tế
    stocks_ma50 = pd.DataFrame(index=stocks_close.index)
    stocks_vol_ma20 = pd.DataFrame(index=stocks_close.index)
    stocks_rsi = pd.DataFrame(index=stocks_close.index)
    stocks_macd = pd.DataFrame(index=stocks_close.index)

    for c in loaded_tickers:
        s_c = stocks_close[c].dropna()
        if len(s_c) >= 20:
            stocks_ma50[c] = s_c.rolling(50, min_periods=20).mean()
            stocks_rsi[c] = calc_rsi(s_c)
            stocks_macd[c] = calc_macd_hist(s_c)
        if f'{c}_V' in stocks_volume.columns:
            s_v = stocks_volume[f'{c}_V'].dropna()
            if len(s_v) >= 20:
                stocks_vol_ma20[f'{c}_V'] = s_v.rolling(20).mean().shift(1)

    # =====================================================================
    # TÍNH ĐỘ RỘNG THỊ TRƯỜNG POINT-IN-TIME (TỶ LỆ MÃ VN30 THỰC TẾ TRÊN MA50)
    # Tuyệt đối chỉ tính trên 30 mã đang thuộc rổ VN30 tại mỗi ngày giao dịch
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

    # LÀM SẠCH TỶ GIÁ USD/VND (Xử lý đơn vị nghìn 21.055, 21.0 và dị biệt 3210, 3211)
    macro_df['USD_VND'] = pd.to_numeric(macro_df['USD_VND'], errors='coerce')
    macro_df['USD_VND'] = macro_df['USD_VND'].apply(lambda x: x * 1000.0 if (pd.notna(x) and 10.0 <= x <= 40.0) else x)
    macro_df['USD_VND'] = macro_df['USD_VND'].apply(lambda x: np.nan if (pd.notna(x) and (x < 15000.0 or x > 35000.0)) else x)
    macro_df['USD_VND'] = macro_df['USD_VND'].interpolate(method='linear').ffill().bfill()

    # ĐỒNG BỘ THỜI ĐIỂM CÔNG BỐ (PUBLICATION LAG):
    # Thị trường VN đóng cửa lúc 15:00 (phát tín hiệu 15:30). Dữ liệu đóng cửa Mỹ (DXY, US10Y)
    # kết thúc sau đó (rạng sáng hôm sau giờ VN). Bắt buộc shift(1) để chống nhìn trước tương lai!
    macro_df['DXY'] = pd.to_numeric(macro_df['DXY'], errors='coerce').shift(1)
    macro_df['US10Y'] = pd.to_numeric(macro_df['US10Y'], errors='coerce').shift(1)

    market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200', 'VN_ATR_PCT']].copy()
    market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']
    market_context['Date'] = pd.to_datetime(market_context['Date']).dt.strftime('%Y-%m-%d')
    market_context['VN_Breadth_MA50'] = market_context['Date'].map(breadth_map).fillna(0.5)
    market_context = market_context.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)

    macro_df = macro_df.merge(market_context[['Date', 'VN_MFI', 'VN_Dist_MA200', 'VN_ATR_PCT', 'VN_Breadth_MA50']], on='Date', how='inner')
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
        s_future_ret = s_valid_close.shift(-30) / s_valid_close - 1

        vn_aligned_c = vn_close_aligned.reindex(s_valid_close.index)
        vn_future_ret = vn_aligned_c.shift(-30) / vn_aligned_c - 1

        # Lưu ngày kết thúc của tương lai 30 phiên để Purged Walk-Forward (chống rò rỉ nhãn)
        future_dates = pd.Series(s_valid_close.index, index=s_valid_close.index).shift(-30)

        # Chỉ đánh nhãn Target khi cả hai future return đều KHÔNG NaN
        valid_future = s_future_ret.notna() & vn_future_ret.notna()
        target_series = np.where(valid_future, (s_future_ret > vn_future_ret).astype(int), np.nan)

        df_c = pd.DataFrame({
            'Date': pd.to_datetime(s_valid_close.index).strftime('%Y-%m-%d'),
            'Future_Date': pd.to_datetime(future_dates).dt.strftime('%Y-%m-%d'),
            'Ticker': c,
            'Vol_Ratio': (s_vol[valid_mask] / s_vol_ma[valid_mask].replace(0, np.nan)).values,
            'Dist_MA50': (s_valid_close / s_ma50[valid_mask]).values,
            'RSI': stocks_rsi[c][valid_mask].values,
            'RSI_Momentum': pd.Series(stocks_rsi[c][valid_mask]).diff(3).values,
            'MACD_Hist': stocks_macd[c][valid_mask].values,
            'Target': target_series
        })
        df_ml_list.append(df_c)

    ml_master = pd.concat(df_ml_list, ignore_index=True).merge(macro_df, on='Date', how='left')
    ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200', 'VN_Breadth_MA50', 'VN_ATR_PCT'] + z_cols
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

    # 1. HUẤN LUYỆN NÃO BỘ CƠ SỞ: Học toàn diện 10 năm chu kỳ vàng (2012 -> 2021)
    # Purged Walk-Forward: Ngày tương lai 30 phiên (Future_Date) phải kết thúc trước 2022-01-01 (100% Zero Leakage)
    print("\n" + "="*65)
    print("🧠 HUẤN LUYỆN NÃO BỘ CƠ SỞ CHU KỲ VÀNG: 2012 -> 2021 (VỚI TIME DECAY)")
    print("   (Purged Walk-Forward: Khóa rò rỉ nhãn tương lai trước 01/01/2022)")
    print("="*65)
    base_train_mask = (train_data['Date'] >= "2012-01-01") & (train_data['Future_Date'] < "2022-01-01")
    X_base = train_data.loc[base_train_mask, ml_features]
    y_base = train_data.loc[base_train_mask, 'Target']
    scale_base = len(y_base[y_base == 0]) / len(y_base[y_base == 1]) if len(y_base[y_base == 1]) > 0 else 1.0

    # Trọng số suy giảm theo thời gian: 3 năm gần nhất = 1.0, 4-7 năm = 0.6, >7 năm = 0.3
    base_weights = compute_time_decay_weights(train_data.loc[base_train_mask, 'Date'], "2022-01-01")

    base_model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_base,
        random_state=42,
        eval_metric='logloss',
        missing=np.nan
    )
    print(f"   ⚙️ Đang huấn luyện trên toàn bộ {base_train_mask.sum():,} mẫu (2012 -> 2021) với Time-Decayed Weights...")
    base_model.fit(X_base, y_base, sample_weight=base_weights)

    # Lưu làm model cho năm 2022 và mô hình cơ sở
    base_model.save_model("xgboost_model_2022.json")
    base_model.save_model("xgboost_ultimate.json")
    models_saved[2022] = base_model
    print("   ✅ Đã hoàn tất và lưu: xgboost_model_2022.json & xgboost_ultimate.json")

    # 2. HUẤN LUYỆN TIẾP CÁC NĂM OUT-OF-SAMPLE: 2023, 2024, 2025, 2026 (EXPANDING TỪ 2012)
    for year in [2023, 2024, 2025, 2026]:
        start_train_date = "2012-01-01"
        end_train_date = f"{year}-01-01"
        
        # Purged Walk-Forward: Nhãn Target 30 phiên phải kết thúc nghiêm ngặt trước end_train_date
        train_mask = (train_data['Date'] >= start_train_date) & (train_data['Future_Date'] < end_train_date)
        test_mask = (train_data['Date'] >= end_train_date) & (train_data['Date'] < f"{year + 1}-01-01")

        X_train = train_data.loc[train_mask, ml_features]
        y_train = train_data.loc[train_mask, 'Target']
        
        scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1.0

        sample_weights = compute_time_decay_weights(train_data.loc[train_mask, 'Date'], end_train_date)

        model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='logloss',
            missing=np.nan
        )

        print(f"\n   ⚙️ Huấn luyện Expanding Model cho năm {year} (Học từ {start_train_date} -> {end_train_date}, {train_mask.sum():,} mẫu, Time-Decayed Weights)...")
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Đánh giá Out-of-Sample trên năm tương ứng
        if test_mask.sum() > 0:
            X_test = train_data.loc[test_mask, ml_features]
            y_test = train_data.loc[test_mask, 'Target']
            if len(y_test.unique()) > 1:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)
                auc = roc_auc_score(y_test, y_pred_proba)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                print(f"      🎯 [OOS {year}] AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} (Mẫu kiểm thử: {test_mask.sum():,})")

        # Lưu model cho năm đó
        model_filename = f"xgboost_model_{year}.json"
        model.save_model(model_filename)
        models_saved[year] = model

    # Mô hình cập nhật mới nhất cho 2026
    LATEST_MODEL_FILE = "xgboost_ultimate.json"
    if 2026 in models_saved:
        models_saved[2026].save_model(LATEST_MODEL_FILE)
    elif models_saved:
        last_year = max(models_saved.keys())
        models_saved[last_year].save_model(LATEST_MODEL_FILE)

    print(f"\n✅ Đã hoàn tất huấn luyện Rolling Walk-Forward cho các năm {list(models_saved.keys())}!")
    print(f"💾 Mô hình mới nhất đã lưu tại: {LATEST_MODEL_FILE}")

    # Bảng xếp hạng Feature Importances của mô hình mới nhất
    latest_model = models_saved.get(2026, list(models_saved.values())[-1])
    importances = pd.Series(latest_model.feature_importances_, index=ml_features).sort_values(ascending=False)
    print("\n🏆 BẢNG XẾP HẠNG TẦM QUAN TRỌNG CỦA CÁC BIẾN (FEATURE IMPORTANCES - LATEST):")
    for feat, imp in importances.items():
        print(f"   • {feat:<18}: {imp*100:.2f}%")

if __name__ == "__main__":
    main()
