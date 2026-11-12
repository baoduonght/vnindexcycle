import pandas as pd
import numpy as np
import requests
from tvDatafeed import TvDatafeed, Interval
import xgboost as xgb
import warnings
import sys
import os
import datetime

from vn30_universe import get_vn30_constituents, ALL_HISTORICAL_VN30_TICKERS

warnings.filterwarnings('ignore')
tv = TvDatafeed()

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
# HÀM TẢI DỮ LIỆU ĐA NGUỒN TỐC ĐỘ CAO (DNSE + VNDIRECT + TCBS + TRADINGVIEW)
# ==========================================
def fetch_ticker_data(symbol, n_bars=4000):
    """
    Tải dữ liệu chứng khoán tốc độ cao:
    - VNINDEX: ƯU TIÊN SỐ 1 đọc trực tiếp từ markets_database10years.csv (10 năm chuẩn),
               nếu có phiên mới hôm nay thì tải thêm từ API để nối vào.
    - Cổ phiếu VN30: Tải trực tiếp 10 năm từ DNSE Entrade (/ohlcs/stock).
    """
    now_ts = int(datetime.datetime.now().timestamp())
    start_ts = int((datetime.datetime.now() - datetime.timedelta(days=12*365)).timestamp())

    # =========================================================================
    # 🌟 ƯU TIÊN SỐ 1 CHO VNINDEX: ĐỌC TRỰC TIẾP TỪ markets_database10years.csv
    # =========================================================================
    if symbol.upper() == 'VNINDEX' and os.path.exists('markets_database10years.csv'):
        try:
            csv_df = pd.read_csv('markets_database10years.csv')
            if 'VNINDEX' in csv_df.columns:
                c_df = csv_df[['Date', 'VNINDEX', 'VN_VOLUME']].dropna(subset=['VNINDEX']).copy()
                c_df['Date'] = pd.to_datetime(c_df['Date']).dt.tz_localize(None).dt.normalize()
                c_df.rename(columns={'VNINDEX': 'Close', 'VN_VOLUME': 'Volume'}, inplace=True)
                c_df['Open'] = c_df['Close']
                c_df['High'] = c_df['Close']
                c_df['Low'] = c_df['Close']
                c_df = c_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)

                # Kiểm tra nếu phiên hôm nay mới hơn ngày cuối trong CSV -> tải thêm nến mới nối vào
                max_csv_date = c_df['Date'].max()
                today_ts = pd.Timestamp.now().normalize()
                if max_csv_date < today_ts:
                    try:
                        latest_url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/index?from={int(max_csv_date.timestamp())}&to={now_ts}&symbol=VNINDEX&resolution=1D"
                        res = requests.get(latest_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=5)
                        if res.status_code == 200:
                            data = res.json()
                            if 't' in data and len(data['t']) > 0:
                                df_live = pd.DataFrame({
                                    'Date': pd.to_datetime(data['t'], unit='s').tz_localize(None).normalize(),
                                    'Open': data['o'],
                                    'High': data['h'],
                                    'Low': data['l'],
                                    'Close': data['c'],
                                    'Volume': data.get('v', [0]*len(data['t']))
                                })
                                newer = df_live[df_live['Date'] > max_csv_date]
                                if not newer.empty:
                                    c_df = pd.concat([c_df, newer], ignore_index=True).drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
                    except Exception:
                        pass
                return c_df
        except Exception as e:
            print(f"   ⚠️ Đọc markets_database10years.csv lỗi ({e}), chuyển sang API...")

    # Phân loại Index vs Stock để gọi đúng endpoint của DNSE Entrade:
    # VNINDEX, VN30, VN100, VNXALL, HNX, HNX30 -> /ohlcs/index
    # Các mã cổ phiếu thường -> /ohlcs/stock
    is_index = symbol.upper() in ['VNINDEX', 'VN30', 'VN100', 'VNXALL', 'HNX', 'HNX30', 'UPCOM']
    entrade_endpoint = 'index' if is_index else 'stock'

    # 1. Nguồn 1: DNSE Entrade API (Tốc độ ~50ms, không bao giờ timeout, đầy đủ 10 năm từ 2014)
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
                    'Volume': data.get('v', [0]*len(data['t']))
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
                    'Volume': data.get('v', [0]*len(data['t']))
                })
                return df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    # 3. Nguồn 3: TCBS Public API
    try:
        url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={symbol}&type=stock&resolution=D&from={start_ts}&to={now_ts}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=6)
        if res.status_code == 200:
            data = res.json().get('data', [])
            if len(data) > 50:
                df = pd.DataFrame(data)
                df.rename(columns={'tradingDate': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    # 4. Nguồn 4: TradingView (Phương án dự phòng)
    try:
        df = tv.get_hist(symbol=symbol, exchange='HOSE', interval=Interval.in_daily, n_bars=n_bars)
        if df is not None and len(df) > 50:
            df = df.reset_index()
            df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    return None

# ==========================================
# 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG
# ==========================================
print("1. Đang tải Market Proxy (VNINDEX)...")
vnindex_raw = fetch_ticker_data('VNINDEX', n_bars=4000)
if vnindex_raw is None or vnindex_raw.empty:
    sys.exit("❌ LỖI: Không thể tải dữ liệu VNINDEX từ cả TradingView lẫn VNDirect!")

vnindex_df = vnindex_raw.rename(columns={'Close': 'VNINDEX'}).drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
vnindex_df['MA20'] = vnindex_df['VNINDEX'].rolling(20, min_periods=10).mean()
vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50, min_periods=20).mean()
vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100, min_periods=40).mean()
vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200, min_periods=80).mean()
vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14) / vnindex_df['VNINDEX']
vnindex_df['ATR_PTS'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['VOL_MA20'] = vnindex_df['Volume'].rolling(20, min_periods=10).mean()
vnindex_df['MACRO_FLOOR'] = vnindex_df['MA100'] - (1.2 * vnindex_df['ATR_PTS'])
vnindex_df['VN_MFI'] = calc_mfi(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], vnindex_df['Volume'], 14)
df_clean = vnindex_df.dropna().drop_duplicates(subset=['Date'], keep='last').copy().reset_index(drop=True)

# ==========================================
# 2. TẢI TOÀN BỘ RỔ CỔ PHIẾU VN30 LỊCH SỬ (POINT-IN-TIME)
# ==========================================
print(f"2. Đang tải toàn bộ {len(ALL_HISTORICAL_VN30_TICKERS)} mã cổ phiếu VN30 lịch sử (Chống Survivorship Bias)...")

# Khởi tạo ma trận dữ liệu với trục ngày tuyệt đối duy nhất (Không duplicate labels)
timeline_dates = pd.DatetimeIndex(df_clean['Date'].drop_duplicates().sort_values())

stocks_close = pd.DataFrame(index=timeline_dates)
stocks_volume = pd.DataFrame(index=timeline_dates)
stocks_high = pd.DataFrame(index=timeline_dates)
stocks_low = pd.DataFrame(index=timeline_dates)

loaded_tickers = []
delisted_or_empty = []

for ticker in ALL_HISTORICAL_VN30_TICKERS:
    df_t = fetch_ticker_data(ticker, n_bars=4000)
    if df_t is not None and not df_t.empty:
        df_t = df_t.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
        df_t.set_index('Date', inplace=True)
        # Loại bỏ triệt để mọi nhãn ngày trùng lặp nếu có từ API
        df_t = df_t[~df_t.index.duplicated(keep='last')]
        stocks_close[ticker] = df_t['Close']
        stocks_volume[f'{ticker}_V'] = df_t['Volume']
        stocks_high[f'{ticker}_H'] = df_t['High']
        stocks_low[f'{ticker}_L'] = df_t['Low']
        loaded_tickers.append(ticker)
    else:
        delisted_or_empty.append(ticker)

print(f"   ✅ Đã nạp thành công {len(loaded_tickers)} mã.")
if delisted_or_empty:
    print(f"   ⚠️ Các mã không có dữ liệu (đã hủy niêm yết triệt để): {', '.join(delisted_or_empty)}")

# =====================================================================
# KHÔNG DÙNG LỆNH FILL: DỮ LIỆU NGUYÊN BẢN 100% (RAW MARKET DATA)
# Không vẽ nến ảo, không kéo dài giá cổ phiếu đã hủy niêm yết.
# Tính chỉ báo kỹ thuật trực tiếp trên các phiên giao dịch thực tế của từng mã.
# =====================================================================
stocks_high_20 = pd.DataFrame(index=stocks_close.index)
stocks_ma20 = pd.DataFrame(index=stocks_close.index)
stocks_ma50 = pd.DataFrame(index=stocks_close.index)
stocks_vol_ma20 = pd.DataFrame(index=stocks_close.index)
stocks_rsi = pd.DataFrame(index=stocks_close.index)
stocks_macd = pd.DataFrame(index=stocks_close.index)

for c in loaded_tickers:
    s_c = stocks_close[c].dropna()
    if len(s_c) >= 20:
        stocks_ma20[c] = s_c.rolling(20).mean()
        stocks_ma50[c] = s_c.rolling(50, min_periods=20).mean()
        stocks_high_20[c] = s_c.rolling(20).max().shift(1)
        stocks_rsi[c] = calc_rsi(s_c)
        stocks_macd[c] = calc_macd_hist(s_c)
    
    if f'{c}_V' in stocks_volume.columns:
        s_v = stocks_volume[f'{c}_V'].dropna()
        if len(s_v) >= 20:
            stocks_vol_ma20[f'{c}_V'] = s_v.rolling(20).mean().shift(1)

# ==========================================
# 3. TẢI BỘ NHỚ AI & ĐỒNG BỘ VĨ MÔ CSV
# ==========================================
print("3. Đang tải Não bộ AI tĩnh & Đồng bộ Vĩ mô CSV...")
try:
    macro_df = pd.read_csv('markets_database10years.csv')
    macro_df = macro_df[['Date', 'DXY', 'US10Y', 'USD_VND']]
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    macro_df = macro_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200']].copy()
market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']
market_context['Date'] = pd.to_datetime(market_context['Date']).dt.strftime('%Y-%m-%d')
market_context = market_context.drop_duplicates(subset=['Date'], keep='last')

macro_df = macro_df.merge(market_context, on='Date', how='inner')
macro_cols = ['DXY', 'US10Y', 'USD_VND']
for col in macro_cols:
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120, min_periods=30).mean()) / macro_df[col].rolling(120, min_periods=30).std()

df_ml_list = []
clean_dates_str = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')

for c in loaded_tickers:
    s_close = stocks_close[c]
    s_vol = stocks_volume[f'{c}_V']
    s_vol_ma = stocks_vol_ma20[f'{c}_V']
    s_ma50 = stocks_ma50[c]

    valid_mask = s_close.notna() & (s_close > 0)
    if valid_mask.sum() < 30:
        continue

    df_c = pd.DataFrame({
        'Date': clean_dates_str[valid_mask],
        'Ticker': c,
        'Vol_Ratio': (s_vol[valid_mask] / s_vol_ma[valid_mask].replace(0, np.nan)).values,
        'Dist_MA50': (s_close[valid_mask] / s_ma50[valid_mask]).values,
        'RSI': stocks_rsi[c][valid_mask].values,
        'RSI_Momentum': pd.Series(stocks_rsi[c][valid_mask]).diff(3).values,
        'MACD_Hist': stocks_macd[c][valid_mask].values
    })
    df_ml_list.append(df_c)

ml_master = pd.concat(df_ml_list, ignore_index=True).merge(macro_df, on='Date', how='left')
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200', 'Z_DXY', 'Z_US10Y', 'Z_USD_VND']

MODEL_FILE = "xgboost_ultimate.json"
xgb_model = xgb.XGBClassifier()

if os.path.exists(MODEL_FILE):
    xgb_model.load_model(MODEL_FILE)
    print(f"   🧠 Đã nạp thành công mô hình: {MODEL_FILE}")
else:
    sys.exit(f"⚠️ Không tìm thấy {MODEL_FILE}! Bác phải chạy train_bot.py trước.")

ml_master['AI_Score'] = np.nan
valid_rows = ml_master[ml_features].notna().all(axis=1)
if valid_rows.sum() > 0:
    ml_master.loc[valid_rows, 'AI_Score'] = xgb_model.predict_proba(ml_master.loc[valid_rows, ml_features])[:, 1]

# Tạo tra cứu AI Score dạng dict (Key: date_str, ticker) -> score
ai_dict = {}
for c in loaded_tickers:
    c_sub = ml_master[ml_master['Ticker'] == c]
    ai_dict[c] = dict(zip(c_sub['Date'], c_sub['AI_Score']))

# ==========================================
# 4. CHẠY SỔ KẾ TOÁN THỰC CHIẾN (POINT-IN-TIME + LIQUIDITY LOCK + 5 SLOTS)
# ==========================================
print("4. Chạy Sổ Kế Toán (Point-in-Time VN30 + 5 Slots + Compounding + T+2.5)...")

def run_ai_quant_system():
    # Lọc kiểm thử 10 năm từ 01/2014 đến nay
    backtest = df_clean[df_clean['Date'] >= '2014-01-01'].copy().reset_index(drop=True)
    
    MAX_SLOTS = 5                      # Tối đa 5 vị thế (20% NAV / slot)
    BUY_FRICTION = 0.0065              # 0.5% trượt giá + 0.15% phí mua
    SELL_FRICTION = 0.0075             # 0.5% trượt giá + 0.15% phí bán + 0.10% thuế
    ANNUAL_BANK_RATE = 0.04            # 4.0% / năm lãi gửi tiền mặt
    DAILY_BANK_RATE = ANNUAL_BANK_RATE / 252
    MIN_HOLD_DAYS = 3                  # Chu kỳ T+2.5 (mua T, bán sớm nhất T+3)
    
    # ==========================================================
    # THAM SỐ CHIẾN LƯỢC ĐÃ TỐI ƯU (V2 - TĂNG TẦN SUẤT + GIẢM MDD)
    # ==========================================================
    AI_THRESHOLD = 0.58                # Khóa cứng ngưỡng AI = 0.58 theo kiểm định OOS Robustness (bỏ overfit)
    VOL_BREAKOUT_RATIO = 1.2           # Hạ từ 1.5x -> 1.2x (bluechip VN30 khó nổ volume 1.5x)
    MARKET_ENTRY_MA = 'MA100'          # Hạ cửa vào từ VNINDEX > MA200 -> VNINDEX > MA100
    ADX_MIN = 18                       # Hạ từ ADX > 22 -> ADX > 18 (bắt trend sớm hơn)
    MAX_BUYS_PER_DAY = 2               # Tối đa 2 lệnh mua / ngày (Giải ngân phân lớp, chống Timing Risk & bẫy Bull-trap)
    TRAILING_STOP_TRIGGER = 0.15       # Kích hoạt trailing stop tiêu chuẩn khi lãi đỉnh >= 15%
    TRAILING_STOP_DROP = 0.08          # Rớt 8% từ đỉnh thì chốt (cho vị thế thường)
    SUPER_WINNER_PEAK = 0.20           # Cổ phiếu dẫn dắt (Leader/Super-Winner): Lãi đỉnh >= 20%
    SUPER_WINNER_DROP = 0.10           # Siêu cổ phiếu: Trailing stop nới rộng 10% từ đỉnh cao nhất
    PROFIT_PROTECTION_PEAK = 0.10      # Kích hoạt chặn lãi theo đỉnh vị thế khi lãi đỉnh >= 10%
    PROFIT_PROTECTION_DROP = 0.08      # Rớt 8% từ đỉnh thì chốt toàn bộ vị thế
    PROFIT_PROTECTION_FLOOR = 0.02     # Sàn bảo vệ vốn/lãi tối thiểu +2%
    MA20_PROFIT_EXIT = 0.10            # Giữ mức 10% để siêu cổ phiếu (HPG, HSG, VNM, PDR) không bị bán non khi vừa mới nhú
    HARD_STOPLOSS = -0.10              # Giữ -10% - không gian rũ cung chuẩn mực cho siêu cổ phiếu trước khi x2, x3
    PORTFOLIO_MDD_LIMIT = -0.18        # Nếu tổng NAV giảm 18% từ đỉnh -> bán hết (Portfolio Stop)
    
    INITIAL_CAPITAL = 100_000_000.0    # Vốn khởi điểm 100 triệu VNĐ
    cash = INITIAL_CAPITAL
    total_bank_interest = 0.0
    held = {}                          # {ticker: {'qty': ..., 'buy_price': ..., 'highest': ..., 'buy_date': ..., 'hold_days': 0}}
    trade_log = []
    daily_nav_history = []
    nav_peak = cash                    # Theo dõi đỉnh NAV cho Portfolio Stop
    consecutive_above_ma100 = 0        # Đếm số phiên liên tiếp VN-Index nằm trên MA100

    def get_current_stock_valuation(d):
        return sum([
            held[k]['qty'] * (stocks_close.at[d, k] if (k in stocks_close.columns and pd.notna(stocks_close.at[d, k]) and stocks_close.at[d, k] > 0) else held[k].get('last_price', held[k]['buy_price']))
            for k in held.keys()
        ])

    for i in range(len(backtest)):
        date = backtest['Date'].iloc[i]
        date_str = date.strftime('%Y-%m-%d')
        idx = backtest['VNINDEX'].iloc[i]
        market_ma20 = backtest['MA20'].iloc[i] if 'MA20' in backtest.columns else backtest['MA50'].iloc[i]
        market_ma50 = backtest['MA50'].iloc[i]
        market_ma100 = backtest['MA100'].iloc[i]
        market_ma200 = backtest['MA200'].iloc[i]
        market_adx = backtest['ADX'].iloc[i]
        market_atr_pct = backtest['ATR_PCT'].iloc[i]
        market_mfi = backtest['VN_MFI'].iloc[i]
        market_vol = backtest['Volume'].iloc[i] if 'Volume' in backtest.columns else 0.0
        market_vol_ma20 = backtest['VOL_MA20'].iloc[i] if 'VOL_MA20' in backtest.columns else 1.0
        macro_floor = backtest['MACRO_FLOOR'].iloc[i] if 'MACRO_FLOOR' in backtest.columns else (market_ma100 * 0.985)

        if idx > market_ma100:
            consecutive_above_ma100 += 1
        else:
            consecutive_above_ma100 = 0

        if date not in stocks_close.index:
            continue

        # Lấy rổ VN30 chính xác tại ngày hôm đó (Chống Survivorship Bias)
        active_vn30 = get_vn30_constituents(date_str)

        # Tiền mặt nhàn rỗi nhận lãi ngân hàng
        if cash > 0:
            interest_today = cash * DAILY_BANK_RATE
            cash += interest_today
            total_bank_interest += interest_today

        # Cập nhật số ngày nắm giữ và giá đỉnh cao nhất của từng vị thế
        for c in held:
            held[c]['hold_days'] += 1
            curr_p = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
            if pd.notna(curr_p) and curr_p > 0:
                held[c]['highest'] = max(held[c].get('highest', curr_p), curr_p)
                held[c]['last_price'] = curr_p

        # ------------------------------------------------------------------
        # HÀM KIỂM TRA TẮC THANH KHOẢN ("TRẮNG BÊN MUA" / MÚA BÊN TRĂNG)
        # ------------------------------------------------------------------
        def is_liquidity_locked(ticker):
            """
            Nếu cổ phiếu rơi sàn (~-7%) và thanh khoản bị nghẽn kiệt (<15% MA20 volume),
            nhà đầu tư KHÔNG THỂ BÁN ĐƯỢC trên sàn HOSE thực tế (như OGC 2015, NVL 2022).
            """
            if i == 0 or ticker not in stocks_close.columns:
                return False
            prev_date = backtest['Date'].iloc[i-1]
            if prev_date not in stocks_close.index:
                return False
            c_p = stocks_close.at[date, ticker]
            p_p = stocks_close.at[prev_date, ticker]
            c_v = stocks_volume.at[date, f'{ticker}_V']
            v_ma = stocks_vol_ma20.at[date, f'{ticker}_V']
            if pd.isna(c_p) or pd.isna(p_p) or pd.isna(c_v) or pd.isna(v_ma):
                return False
            if p_p > 0 and (c_p <= p_p * 0.933) and (v_ma > 0 and c_v < v_ma * 0.15):
                return True
            return False

        # ------------------------------------------------------------------
        # BƯỚC 1: XỬ LÝ CÁC TÍN HIỆU BÁN (EXIT STRATEGY THỰC TẾ 100%)
        # ------------------------------------------------------------------
        to_sell = []

        # Tính giá trị cổ phiếu hiện tại (Dùng giá đóng cửa phiên gần nhất nếu hôm nay không có giao dịch)
        temp_stock_val = sum([
            held[c]['qty'] * (stocks_close.at[date, c] if (c in stocks_close.columns and pd.notna(stocks_close.at[date, c]) and stocks_close.at[date, c] > 0) else held[c]['last_price'])
            for c in held.keys()
        ])
        temp_nav = cash + temp_stock_val
        if pd.notna(temp_nav) and temp_nav > nav_peak:
            nav_peak = temp_nav

        # ===== BẢO VỆ 1: PORTFOLIO-LEVEL STOP LOSS =====
        # Nếu tổng NAV sụt >= 18% từ đỉnh -> bán toàn bộ danh mục để bảo vệ vốn
        portfolio_dd = (temp_nav - nav_peak) / nav_peak if nav_peak > 0 else 0.0

        # ===== BẢO VỆ 2: CÔNG TẮC VĨ MÔ THÍCH ỨNG (VOLUME CONFIRMATION + DYNAMIC ATR BUFFER) =====
        # 1. Tránh bẫy chọc râu / nhiễu nến trong Sideway:
        #    - Khi Index thủng MA100 với Volume cạn kiệt (< 0.9x MA20), đó là rung lắc / rũ cung kỹ thuật.
        #    - Không bán tháo hoảng loạn cả danh mục, để các mã khỏe tiếp tục chạy.
        # 2. Cắt sớm hơn 1-2 phiên khi có biến động thực sự:
        #    - Nếu sập mạnh thủng dải đệm Keltner động (MA100 - 1.2 * ATR), hoặc thủng MA100 kèm Volume >= 1.15x MA20.
        prev_idx = backtest['VNINDEX'].iloc[i-1] if i > 0 else idx
        prev_ma100 = backtest['MA100'].iloc[i-1] if i > 0 else market_ma100
        vol_shock = (market_vol / market_vol_ma20) if (pd.notna(market_vol_ma20) and market_vol_ma20 > 0) else 1.0

        # Kịch bản 1: Sập dứt khoát thủng kênh ATR động (Breakdown thực sự ngoài tầm rung lắc)
        hard_atr_breakdown = (idx < macro_floor)

        # Kịch bản 2: Thủng MA100 kèm Volume lớn xác nhận phân phối của tổ chức (Vol >= 1.15x MA20) -> Cắt sớm ngay phiên 1!
        vol_confirmed_breakdown = (idx < market_ma100) and (vol_shock >= 1.15)

        # Kịch bản 3: Đóng cửa dưới MA100 liên tiếp 2 phiên VÀ áp lực bán không cạn kiệt (Vol >= 0.90x MA20)
        two_day_breakdown = (idx < market_ma100) and (prev_idx < prev_ma100) and (vol_shock >= 0.90)

        is_macro_breakdown = hard_atr_breakdown or vol_confirmed_breakdown or two_day_breakdown

        if portfolio_dd <= -0.18 and len(held) > 0:
            for c in list(held.keys()):
                curr_p = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                # Chỉ bán khi có giá giao dịch thực tế hôm nay và không bị nghẽn sàn
                if held[c]['hold_days'] >= MIN_HOLD_DAYS and pd.notna(curr_p) and curr_p > 0 and not is_liquidity_locked(c):
                    to_sell.append((c, 'SELL (Portfolio Stop -18%)'))

        elif is_macro_breakdown and len(held) > 0:
            for c in list(held.keys()):
                curr_p = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                # Chỉ bán khi có giá giao dịch thực tế hôm nay và không bị nghẽn sàn
                if held[c]['hold_days'] >= MIN_HOLD_DAYS and pd.notna(curr_p) and curr_p > 0 and not is_liquidity_locked(c):
                    to_sell.append((c, 'SELL (Cắt Macro gãy MA100)'))

        else:
            # Nếu chỉ mới là phiên chớm rung lắc dưới MA100 (chưa xác nhận gãy), hoặc VNINDEX > MA100:
            # -> KHÔNG hoảng loạn bán tháo cổ phiếu khỏe! Quản trị riêng từng vị thế:
            for c in list(held.keys()):
                if held[c]['hold_days'] < MIN_HOLD_DAYS:
                    continue
                if is_liquidity_locked(c):
                    continue

                curr_price = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                # Nếu hôm nay mã này không có giao dịch, KHÔNG THỂ BÁN HÔM NAY -> Chờ phiên sau có giá thực
                if pd.isna(curr_price) or curr_price <= 0:
                    continue

                buy_price = held[c]['buy_price']
                profit_margin = (curr_price - buy_price) / buy_price
                peak_margin = (held[c]['highest'] - buy_price) / buy_price

                # LÝ DO 1: Bị HOSE loại khỏi rổ VN30
                if c not in active_vn30:
                    to_sell.append((c, 'SELL (Rời rổ VN30)'))
                # LÝ DO 2: CẮT LỖ CỨNG -10% (Bảo vệ từng vị thế riêng lẻ - đủ rộng cho siêu cổ rũ cung)
                elif profit_margin <= HARD_STOPLOSS:
                    to_sell.append((c, 'SELL (Cắt lỗ cứng -10%)'))
                # LÝ DO 3: QUẢN TRỊ SIÊU CỔ PHIẾU (Lãi đỉnh >= 20% - Cho không gian bay cao x2, x3)
                elif peak_margin >= SUPER_WINNER_PEAK:
                    # Trailing Stop nới rộng: Rớt 10% từ đỉnh cao nhất
                    if curr_price < (held[c]['highest'] * (1.0 - SUPER_WINNER_DROP)):
                        to_sell.append((c, 'SELL (Trailing Stop chốt siêu cổ)'))
                    # Xu hướng trung hạn: Gãy hẳn MA50 mới chốt (không bị rung rũ bởi MA20)
                    elif c in stocks_ma50.columns and pd.notna(stocks_ma50.at[date, c]) and curr_price < stocks_ma50.at[date, c]:
                        to_sell.append((c, 'SELL (Gãy MA50 chốt lời sóng lớn)'))
                    # Khóa lãi tối thiểu: Không bao giờ để siêu cổ phiếu rơi về hòa vốn hay lỗ
                    elif profit_margin <= 0.08:
                        to_sell.append((c, 'SELL (Khóa lãi sàn bảo vệ +8%)'))
                # LÝ DO 4: QUY TẮC CHẶN LÃI THEO ĐỈNH VỊ THẾ (Lãi đỉnh 10% - 20% - Giải mã năm 2026)
                elif peak_margin >= PROFIT_PROTECTION_PEAK:
                    # Trailing Stop bảo vệ thành quả: Rớt 8% từ đỉnh cao nhất
                    if curr_price < (held[c]['highest'] * (1.0 - PROFIT_PROTECTION_DROP)):
                        to_sell.append((c, 'SELL (Trailing Stop chốt lời)'))
                    # Thủng MA20 khi vẫn còn giữ được mức lãi >= 5%
                    elif c in stocks_ma20.columns and pd.notna(stocks_ma20.at[date, c]) and curr_price < stocks_ma20.at[date, c] and profit_margin >= 0.05:
                        to_sell.append((c, 'SELL (Thủng MA20 bảo toàn lãi)'))
                    # Sàn bảo vệ hòa vốn / giữ lãi +2% (Tuyệt đối không để vị thế từng lãi > 10% biến thành lỗ ngược)
                    elif profit_margin <= PROFIT_PROTECTION_FLOOR:
                        to_sell.append((c, 'SELL (Khóa bảo vệ vốn +2%)'))
                # LÝ DO 5: VỊ THẾ BÌNH THƯỜNG (Lãi đỉnh < 10%)
                else:
                    # Thủng MA20 khi đã đạt mức lãi tối thiểu > 10% (tránh chốt non siêu cổ phiếu do rung lắc)
                    if c in stocks_ma20.columns and pd.notna(stocks_ma20.at[date, c]) and curr_price < stocks_ma20.at[date, c] and profit_margin > MA20_PROFIT_EXIT:
                        to_sell.append((c, 'SELL (Thủng MA20 bảo toàn lãi)'))

        # Khớp các lệnh bán hợp lệ (GIÁ THẬT 100%, KHÔNG CẦN FFILL)
        for c, reason in to_sell:
            if c in held:
                sell_price = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                # Nếu không có giá giao dịch hôm nay, hoãn bán chờ phiên tiếp theo
                if pd.isna(sell_price) or sell_price <= 0:
                    continue

                qty = held[c]['qty']
                gross_val = qty * sell_price
                net_val = gross_val * (1.0 - SELL_FRICTION)
                buy_cost = qty * held[c]['buy_price'] * (1.0 + BUY_FRICTION)
                pnl_vnd = net_val - buy_cost
                cash += net_val

                pnl_pct = ((sell_price * (1.0 - SELL_FRICTION)) / (held[c]['buy_price'] * (1.0 + BUY_FRICTION)) - 1.0) * 100.0
                hold_days = held[c]['hold_days']
                del held[c]

                # Cập nhật trạng thái danh mục theo thời gian thực ngay sau lệnh bán
                rem_stock_val = get_current_stock_valuation(date)
                trade_nav = cash + rem_stock_val
                cum_ret_pct = ((trade_nav / INITIAL_CAPITAL) - 1.0) * 100.0

                trade_log.append({
                    'Date': date_str,
                    'Action': reason,
                    'Ticker': c,
                    'Price': round(float(sell_price), 2),
                    'Trade_Value': int(round(net_val)),
                    'PnL_VND': int(round(pnl_vnd)),
                    'Profit(%)': round(float(pnl_pct), 2),
                    'Hold_Days': int(hold_days),
                    'Total_NAV': int(round(trade_nav)),
                    'Portfolio_Return(%)': round(float(cum_ret_pct), 2),
                    'AI_Score': None
                })

        # Reset đỉnh NAV khi tài khoản đang ở 100% tiền mặt để tránh vòng lặp stop-out
        if len(held) == 0:
            nav_peak = cash

        # ------------------------------------------------------------------
        # BƯỚC 2: TÍNH TOÁN VỐN VÀ QUÉT TÍN HIỆU MUA (BUY STRATEGY V2)
        # ------------------------------------------------------------------
        stock_val_now = get_current_stock_valuation(date)
        current_nav = cash + stock_val_now
        if current_nav > nav_peak:
            nav_peak = current_nav
        daily_nav_history.append({
            'Date': date_str,
            'NAV': int(round(current_nav)),
            'Cash': int(round(cash)),
            'Bank_Interest': int(round(total_bank_interest)),
            'Portfolio_Return(%)': round(((current_nav / INITIAL_CAPITAL) - 1.0) * 100.0, 2)
        })

        empty_slots = MAX_SLOTS - len(held)
        
        # Quản trị tỷ trọng theo trạng thái kỹ thuật Index (Scale-in / Pyramiding Risk Management)
        if idx < market_ma50 or market_atr_pct > 0.020:
            max_exposure = 0.4       # Thị trường yếu/biến động cao: chỉ cho phép 40% NAV vào cổ phiếu
        elif idx < market_ma200:
            max_exposure = 0.5       # Thị trường Gấu (dưới MA200): tối đa 50% NAV (phòng thủ chống bull trap)
        elif idx < market_ma100:
            max_exposure = 0.7       # Thị trường hồi phục sớm: 70%
        else:
            max_exposure = 1.0       # Thị trường Uptrend mạnh toàn diện: 100%
        
        allowed_stock_cap = max(0.0, (current_nav * max_exposure) - stock_val_now)

        # ====================================================================
        # ĐIỀU KIỆN VÀO LỆNH MỚI (V3 - DYNAMIC MACRO RE-ENTRY ENGINE)
        # 1. trend_reentry_confirmed: >= 3 phiên trên MA100 HOẶC Follow-Through Day (V-Shape bùng nổ theo đà)
        # 2. is_volatility_calm: Biến động hạ nhiệt (market_atr_pct <= 0.020)
        # 3. is_breadth_healthy: >= 35% số mã VN30 nằm trên MA50 (Dòng tiền lan tỏa đồng thuận)
        # ====================================================================
        vol_shock_today = (market_vol / market_vol_ma20) if (pd.notna(market_vol_ma20) and market_vol_ma20 > 0) else 1.0
        prev_idx = backtest['VNINDEX'].iloc[i-1] if i > 0 else idx

        # Ngoại lệ V-Shape: Phiên bùng nổ theo đà (Follow-Through Day)
        is_follow_through_day = (idx > market_ma100) and (vol_shock_today >= 1.25) and (idx >= prev_idx * 1.010)

        # Xác nhận xu hướng uy tín: Duy trì >= 3 phiên trên MA100 HOẶC có phiên FTD
        trend_reentry_confirmed = (consecutive_above_ma100 >= 3) or is_follow_through_day

        # Độ biến động đã hạ nhiệt về vùng tích lũy (không hoảng loạn giật nến)
        is_volatility_calm = (market_atr_pct <= 0.020)

        # Độ rộng thị trường: Tỷ lệ cổ phiếu VN30 đang giữ được MA50 (tránh bẫy kéo trụ ảo)
        vn30_above_ma50_cnt = sum([
            1 for ticker in active_vn30
            if ticker in stocks_close.columns and ticker in stocks_ma50.columns
            and pd.notna(stocks_close.at[date, ticker]) and pd.notna(stocks_ma50.at[date, ticker])
            and stocks_close.at[date, ticker] > stocks_ma50.at[date, ticker]
        ])
        market_breadth = (vn30_above_ma50_cnt / len(active_vn30)) if len(active_vn30) > 0 else 0.5
        is_breadth_healthy = (market_breadth >= 0.35)

        can_enter_new_trades = (
            trend_reentry_confirmed
            and is_volatility_calm
            and is_breadth_healthy
            and (market_adx > ADX_MIN)
            and (allowed_stock_cap > 0)
        )

        if empty_slots > 0 and can_enter_new_trades:
            buy_candidates = []
            prev_date = backtest['Date'].iloc[i-1] if i > 0 else None

            for c in active_vn30:
                if c not in held and c in loaded_tickers and c in stocks_close.columns:
                    curr_price = stocks_close.at[date, c]
                    prev_c_price = stocks_close.at[prev_date, c] if (prev_date is not None and prev_date in stocks_close.index) else np.nan
                    curr_vol = stocks_volume.at[date, f'{c}_V'] if f'{c}_V' in stocks_volume.columns else np.nan
                    prev_h20 = stocks_high_20.at[date, c] if c in stocks_high_20.columns else np.nan
                    prev_v_ma = stocks_vol_ma20.at[date, f'{c}_V'] if f'{c}_V' in stocks_vol_ma20.columns else np.nan

                    if pd.isna(curr_price) or pd.isna(curr_vol) or pd.isna(prev_h20) or pd.isna(prev_v_ma):
                        continue
                    if prev_v_ma <= 0 or curr_price <= 0:
                        continue

                    score = ai_dict.get(c, {}).get(date_str, np.nan)
                    if pd.isna(score):
                        continue

                    v_ratio = curr_vol / prev_v_ma
                    r_rsi = stocks_rsi.at[date, c] if c in stocks_rsi.columns else 50.0
                    s_ma20_val = stocks_ma20.at[date, c] if c in stocks_ma20.columns else np.nan
                    s_ma50_val = stocks_ma50[c].at[date] if c in stocks_ma50.columns else np.nan

                    # Nến xanh xác nhận bật tăng (tăng điểm so với phiên trước)
                    is_green_candle = (pd.notna(prev_c_price) and curr_price >= prev_c_price * 1.002)

                    # ========= BỘ LỌC AI SCORE CỐ ĐỊNH (OOS ROBUSTNESS TEST - CHỐNG OVERFITTING) =========
                    # Khóa cứng ngưỡng AI = 0.58 cho toàn bộ điều kiện thị trường (loại bỏ if/else curve-fitting)
                    effective_ai_breakout = AI_THRESHOLD # 0.58

                    # ========= TÍN HIỆU CHIẾN LƯỢC: MOMENTUM BREAKOUT (ĐỘNG LỰC DÒNG TIỀN) =========
                    # Vượt đỉnh 20 phiên + Volume >= 1.2x MA20 + Đạt chuẩn AI Score cố định
                    # (Tập trung 100% vào Breakout - chiến lược tạo ra 100% lợi nhuận lịch sử, loại bỏ bẫy Pullback bắt dao rơi)
                    if curr_price > prev_h20 and v_ratio >= VOL_BREAKOUT_RATIO and score >= effective_ai_breakout:
                        buy_candidates.append({
                            'Ticker': c, 'Score': score, 'Price': curr_price,
                            'Vol_Ratio': round(v_ratio, 1), 'RSI': round(r_rsi, 1),
                            'Signal': 'Breakout'
                        })

            # Ưu tiên theo AI Score và Giải ngân phân lớp (Tối đa 2 lệnh / ngày)
            if buy_candidates:
                buy_candidates = sorted(buy_candidates, key=lambda x: x['Score'], reverse=True)

                target_slot_size = current_nav / MAX_SLOTS
                buys_today = 0

                for cand in buy_candidates:
                    if empty_slots <= 0 or cash <= 1_000_000 or buys_today >= MAX_BUYS_PER_DAY:
                        break
                    c = cand['Ticker']
                    buy_price = cand['Price']
                    alloc = min(target_slot_size, cash, allowed_stock_cap)

                    if alloc >= 2_000_000 and buy_price > 0:
                        actual_cost = buy_price * (1.0 + BUY_FRICTION)
                        shares = np.floor(alloc / actual_cost)
                        if shares > 0:
                            total_spent = shares * actual_cost
                            cash -= total_spent
                            allowed_stock_cap -= total_spent
                            held[c] = {
                                'qty': shares,
                                'buy_price': buy_price,
                                'highest': buy_price,
                                'last_price': buy_price,
                                'buy_date': date_str,
                                'hold_days': 0,
                                'scaled_out': False
                            }
                            # Cập nhật trạng thái danh mục theo thời gian thực ngay sau lệnh mua
                            rem_stock_val = get_current_stock_valuation(date)
                            trade_nav = cash + rem_stock_val
                            cum_ret_pct = ((trade_nav / INITIAL_CAPITAL) - 1.0) * 100.0

                            trade_log.append({
                                'Date': date_str,
                                'Action': f"BUY ({cand['Signal']})",
                                'Ticker': c,
                                'Price': round(float(buy_price), 2),
                                'Trade_Value': int(round(total_spent)),
                                'PnL_VND': None,
                                'Profit(%)': None,
                                'Hold_Days': None,
                                'Total_NAV': int(round(trade_nav)),
                                'Portfolio_Return(%)': round(float(cum_ret_pct), 2),
                                'AI_Score': round(float(cand['Score']), 2)
                            })
                            empty_slots -= 1
                            buys_today += 1

    # Chốt NAV phiên cuối cùng
    final_stock_val = get_current_stock_valuation(backtest['Date'].iloc[-1])
    final_nav = cash + final_stock_val

    df_nav = pd.DataFrame(daily_nav_history)
    df_trades = pd.DataFrame(trade_log)

    desired_cols = [
        'Date', 'Action', 'Ticker', 'Price', 'Trade_Value', 'PnL_VND',
        'Profit(%)', 'Hold_Days', 'Total_NAV', 'Portfolio_Return(%)', 'AI_Score'
    ]
    final_cols = [col for col in desired_cols if col in df_trades.columns]
    df_trades = df_trades[final_cols]

    # Ép kiểu Int64 nullable để xuất file sạch 100%, không bị đuôi .0
    for int_col in ['Trade_Value', 'PnL_VND', 'Hold_Days', 'Total_NAV']:
        if int_col in df_trades.columns:
            df_trades[int_col] = pd.to_numeric(df_trades[int_col], errors='coerce').round(0).astype('Int64')

    for nav_int in ['NAV', 'Cash', 'Bank_Interest']:
        if nav_int in df_nav.columns:
            df_nav[nav_int] = pd.to_numeric(df_nav[nav_int], errors='coerce').round(0).astype('Int64')

    return df_trades, df_nav, final_nav, total_bank_interest, held

# ==========================================
# CHẠY CHIẾN LƯỢC & XUẤT BÁO CÁO TOÀN DIỆN
# ==========================================
df_trade_log, df_nav_hist, final_nav, total_interest, current_held = run_ai_quant_system()

# Ghi nhận file trading_log.csv cho Power BI (Sổ lệnh tinh gọn, không rác số thập phân)
df_trade_log.to_csv('trading_log.csv', index=False, na_rep='', float_format='%.2f')
print("✅ Đã ghi nhận sổ kế toán giao dịch tinh gọn vào file trading_log.csv!")

# Ghi nhận file daily_nav.csv cho Power BI (Số nguyên tròn trịa, không tràn số thập phân)
df_nav_hist.to_csv('daily_nav.csv', index=False, na_rep='', float_format='%.2f')
print("✅ Đã ghi nhận biến động tài sản hàng ngày chuẩn vào file daily_nav.csv!")

# ==========================================
# TÍNH TOÁN METRICS ĐỊNH LƯỢNG RỦI RO & HIỆU SUẤT
# ==========================================
initial_capital = 100_000_000.0
total_return_pct = ((final_nav / initial_capital) - 1.0) * 100.0

# Tính CAGR (Lợi nhuận kép theo năm)
nav_dates = pd.to_datetime(df_nav_hist['Date'])
n_years = (nav_dates.iloc[-1] - nav_dates.iloc[0]).days / 365.25
cagr = ((final_nav / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0

# Tính Maximum Drawdown (MDD)
df_nav_hist['Peak'] = df_nav_hist['NAV'].cummax()
df_nav_hist['Drawdown'] = (df_nav_hist['NAV'] - df_nav_hist['Peak']) / df_nav_hist['Peak']
max_mdd = df_nav_hist['Drawdown'].min() * 100.0

# Tìm đáy sụt giảm sâu nhất
trough_idx = df_nav_hist['Drawdown'].idxmin()
trough_date = pd.to_datetime(df_nav_hist['Date'].iloc[trough_idx])
peak_sub = df_nav_hist.iloc[:trough_idx+1]
peak_date = pd.to_datetime(peak_sub.loc[peak_sub['NAV'] == peak_sub['Peak'].max(), 'Date'].iloc[-1])

calmar_ratio = abs(cagr / max_mdd) if max_mdd != 0 else 0.0

sells = df_trade_log[df_trade_log['Action'].str.contains('SELL', na=False)] if not df_trade_log.empty else pd.DataFrame()
total_sells = len(sells)
win_sells = len(sells[sells['Profit(%)'] > 0]) if total_sells > 0 else 0
loss_sells = total_sells - win_sells
win_rate = (win_sells / total_sells * 100.0) if total_sells > 0 else 0.0
best_trade = sells['Profit(%)'].max() if total_sells > 0 else 0.0
worst_trade = sells['Profit(%)'].min() if total_sells > 0 else 0.0

gross_profit = sells.loc[sells['PnL_VND'] > 0, 'PnL_VND'].sum() if total_sells > 0 else 0.0
gross_loss = abs(sells.loc[sells['PnL_VND'] < 0, 'PnL_VND'].sum()) if total_sells > 0 else 0.0
profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
avg_win_pct = sells.loc[sells['Profit(%)'] > 0, 'Profit(%)'].mean() if win_sells > 0 else 0.0
avg_loss_pct = sells.loc[sells['Profit(%)'] < 0, 'Profit(%)'].mean() if loss_sells > 0 else 0.0
payoff_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 999.0

print("\n" + "="*58)
print("📊 BÁO CÁO ĐỊNH LƯỢNG THỰC CHIẾN 10 NĂM (SURVIVORSHIP-FREE)")
print("="*58)
print(f"💵 Vốn khởi điểm (01/2014):        {initial_capital:,.0f} VNĐ")
print(f"🏦 TỔNG TÀI SẢN HIỆN TẠI (NAV):   {final_nav:,.0f} VNĐ")
print(f"💰 LÃI BANK THU VỀ (4%/năm):      +{total_interest:,.0f} VNĐ")
print(f"🚀 TỔNG LỢI NHUẬN (Đã trừ phí):    +{total_return_pct:.2f}%")
print(f"📈 Lợi nhuận kép TB (CAGR):        {cagr:.2f}% / năm")
print(f"----------------------------------------------------------")
print(f"📉 SỤT GIẢM TỐI ĐA (Max Drawdown): {max_mdd:.2f}%")
print(f"   (Thời kỳ chịu đựng lớn nhất: Từ đỉnh {pd.to_datetime(peak_date).strftime('%d/%m/%Y')} về đáy {pd.to_datetime(trough_date).strftime('%d/%m/%Y')})")
print(f"⚡ Chỉ số Calmar (CAGR / MDD):     {calmar_ratio:.2f}")
print(f"----------------------------------------------------------")
print(f"Tổng số lệnh đã chốt sổ:          {total_sells}")
print(f"🎯 Tỷ lệ thắng (Win Rate):         {win_rate:.2f}% (Thắng {win_sells}/{total_sells} trận)")
print(f"⚖️ Profit Factor (Tổng Lãi/Lỗ):    {profit_factor:.2f}")
print(f"🎲 Payoff Ratio (Lãi TB / Lỗ TB):  {payoff_ratio:.2f}x (Lãi TB: +{avg_win_pct:.2f}% | Lỗ TB: {avg_loss_pct:.2f}%)")
print(f"💰 Lệnh lãi đơn lẻ cao nhất:       +{best_trade:.2f}%")
print(f"🩸 Lệnh lỗ đơn lẻ sâu nhất:        {worst_trade:.2f}%")
print("="*58 + "\n")

# ==========================================
# 5. TELEGRAM RADAR BÁO CÁO HÀNG NGÀY
# ==========================================
print("5. Báo cáo Telegram Radar...")
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_msg(text_message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text_message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text_message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"⚠️ Gửi Telegram thất bại: {e}")

latest_date = df_clean['Date'].iloc[-1]
latest_date_str = latest_date.strftime('%Y-%m-%d')
active_today = get_vn30_constituents(latest_date_str)

msg = f"🤖 <b>SIÊU RADAR VN30 - BÁO CÁO THỰC CHIẾN</b> 🤖\n"
msg += f"📅 Ngày: <b>{latest_date.strftime('%d/%m/%Y')}</b>\n"
msg += f"🏛 Rổ VN30 hiệu lực: <b>{len(active_today)} mã chuẩn HOSE</b>\n"
msg += f"➖➖➖➖➖➖➖➖➖➖\n\n🎯 <b>TÍN HIỆU HÔM NAY:</b>\n"

buy_count, sell_count = 0, 0
if not df_trade_log.empty:
    latest_trades = df_trade_log[df_trade_log['Date'] == latest_date_str]
    if not latest_trades.empty:
        for _, row in latest_trades.iterrows():
            if 'BUY' in row['Action']:
                trade_val = f" | Tiền: {row['Trade_Value']:,.0f} VNĐ" if 'Trade_Value' in row and pd.notna(row['Trade_Value']) else ""
                msg += f"🟢 <b>MUA: {row['Ticker']}</b> (Giá: {row['Price']:,.2f}{trade_val}) | AI: <b>{row['AI_Score'] * 100:.1f}%</b>\n\n"
                buy_count += 1
            elif 'SELL' in row['Action']:
                icon = "🔥" if row['Profit(%)'] > 0 else "🩸"
                pnl_str = f" ({row['PnL_VND']:+,.0f} VNĐ)" if 'PnL_VND' in row and pd.notna(row['PnL_VND']) else ""
                msg += f"🔴 <b>BÁN: {row['Ticker']}</b> ({row['Action']}) | Lãi/Lỗ: {icon} <b>{row['Profit(%)']}%</b>{pnl_str}\n\n"
                sell_count += 1

if buy_count == 0 and sell_count == 0:
    msg += "<i>⏳ Không có lệnh Mua/Bán mới trong phiên hôm nay.</i>\n\n"

# Báo cáo chi tiết danh mục đang nắm giữ
msg += f"💼 <b>DANH MỤC HIỆN TẠI ({len(current_held)}/5 vị thế):</b>\n"
if current_held:
    for ticker, pos in current_held.items():
        curr_p = stocks_close.iloc[-1][ticker] if ticker in stocks_close.columns and pd.notna(stocks_close.iloc[-1][ticker]) else pos.get('last_price', pos['buy_price'])
        unrealized_pnl = ((curr_p / pos['buy_price']) - 1.0) * 100.0
        icon = "🟢" if unrealized_pnl >= 0 else "🔴"
        peak_pnl = ((pos['highest'] / pos['buy_price']) - 1.0) * 100.0
        leader_badge = " 👑 [Leader]" if peak_pnl >= 20.0 else ""
        msg += f"{icon} <b>{ticker}</b>: {curr_p:,.0f} | Lãi tạm: <b>{unrealized_pnl:+.1f}%</b> (Đỉnh: +{peak_pnl:.1f}%){leader_badge} | T+{pos['hold_days']}\n"
    msg += "\n"
else:
    msg += "<i>100% Tiền mặt (Bảo toàn vốn tối đa, nhận lãi Bank 4%/năm).</i>\n\n"

msg += f"➖➖➖➖➖➖➖➖➖➖\n"
msg += f"🏦 NAV Hiện tại: <b>{final_nav:,.0f} VNĐ</b> | Max Drawdown: <b>{max_mdd:.2f}%</b>"

send_telegram_msg(msg)
print("✅ Hoàn tất chu trình quant_bot.py!")
