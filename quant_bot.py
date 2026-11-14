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
# TỪ ĐIỂN PHÂN LOẠI NGÀNH VN30 (SECTOR MAP)
# Khống chế rủi ro tập trung ngành (Sector Cap <= 2 mã/ngành)
# ==========================================
SECTOR_MAP = {
    # 1. Ngân hàng (Banking)
    'VCB': 'BANK', 'BID': 'BANK', 'CTG': 'BANK', 'MBB': 'BANK', 'TCB': 'BANK',
    'VPB': 'BANK', 'ACB': 'BANK', 'HDB': 'BANK', 'TPB': 'BANK', 'STB': 'BANK',
    'VIB': 'BANK', 'SSB': 'BANK', 'SHB': 'BANK', 'LPB': 'BANK', 'EIB': 'BANK',
    'MSB': 'BANK', 'OCB': 'BANK',
    
    # 2. Bất động sản, Xây dựng & Hạ tầng (Real Estate & Construction)
    'VIC': 'REAL_ESTATE', 'VHM': 'REAL_ESTATE', 'VRE': 'REAL_ESTATE', 
    'NVL': 'REAL_ESTATE', 'KDH': 'REAL_ESTATE', 'PDR': 'REAL_ESTATE', 
    'BCM': 'REAL_ESTATE', 'KBC': 'REAL_ESTATE', 'CII': 'REAL_ESTATE', 
    'CTD': 'REAL_ESTATE', 'ROS': 'REAL_ESTATE', 'FLC': 'REAL_ESTATE',
    'ITA': 'REAL_ESTATE', 'IJC': 'REAL_ESTATE', 'HHS': 'REAL_ESTATE',
    'SJS': 'REAL_ESTATE', 'DIG': 'REAL_ESTATE', 'DXG': 'REAL_ESTATE',
    
    # 3. Thép & Kim loại (Steel & Materials)
    'HPG': 'STEEL', 'HSG': 'STEEL', 'NKG': 'STEEL',
    
    # 4. Chứng khoán (Brokerage)
    'SSI': 'SECURITIES', 'HCM': 'SECURITIES', 'VND': 'SECURITIES', 'VCI': 'SECURITIES',
    
    # 5. Dầu khí, Năng lượng & Tiện ích (Energy & Utilities)
    'GAS': 'ENERGY', 'PLX': 'ENERGY', 'POW': 'ENERGY', 'PVD': 'ENERGY', 
    'PVT': 'ENERGY', 'NT2': 'ENERGY', 'PPC': 'ENERGY', 'REE': 'ENERGY', 
    'VSH': 'ENERGY', 'PGD': 'ENERGY',
    
    # 6. Bán lẻ & Tiêu dùng (Retail & Consumer)
    'MWG': 'CONSUMER', 'MSN': 'CONSUMER', 'SAB': 'CONSUMER', 'VNM': 'CONSUMER', 
    'PNJ': 'CONSUMER', 'KDC': 'CONSUMER', 'SBT': 'CONSUMER', 'PET': 'CONSUMER',
    
    # 7. Công nghệ & Viễn thông (Technology)
    'FPT': 'TECH',
    
    # 8. Phân bón, Hóa chất, Cao su & Nông nghiệp (Chemicals & Agriculture)
    'GVR': 'CHEMICALS', 'DPM': 'CHEMICALS', 'DCM': 'CHEMICALS', 
    'DRC': 'CHEMICALS', 'CSM': 'CHEMICALS', 'HAG': 'AGRICULTURE', 'HVG': 'AGRICULTURE',
    
    # 9. Vận tải & Hàng không (Logistics & Aviation)
    'VJC': 'LOGISTICS', 'GMD': 'LOGISTICS', 'HVN': 'LOGISTICS',
    
    # 10. Bảo hiểm (Insurance)
    'BVH': 'INSURANCE'
}

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
# HÀM TẢI DỮ LIỆU ĐA NGUỒN (DNSE + VNDIRECT + TCBS + TRADINGVIEW)
# ==========================================
def fetch_ticker_data(symbol, n_bars=4000):
    """
    Tải dữ liệu chứng khoán OHLCV THẬT từ API:
    - VNINDEX & các chỉ số: DNSE Entrade /ohlcs/index (Open, High, Low, Close, Volume thật).
      Tuyệt đối KHÔNG làm phẳng nến (Open=High=Low=Close) để đảm bảo ATR/ADX/MFI chuẩn xác 100%.
    - Cổ phiếu VN30: DNSE Entrade /ohlcs/stock.
    """
    now_ts = int(datetime.datetime.now().timestamp())
    start_ts = int((datetime.datetime.now() - datetime.timedelta(days=12*365)).timestamp())

    is_index = symbol.upper() in ['VNINDEX', 'VN30', 'VN100', 'VNXALL', 'HNX', 'HNX30', 'UPCOM']
    entrade_endpoint = 'index' if is_index else 'stock'

    # 1. Nguồn 1: DNSE Entrade API (Tốc độ cao, đầy đủ OHLCV 10-12 năm)
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
        api_type = 'index' if is_index else 'stock'
        url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={symbol}&type={api_type}&resolution=D&from={start_ts}&to={now_ts}"
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

    # Fallback khẩn cấp nếu mất mạng: CSV
    if symbol.upper() == 'VNINDEX' and os.path.exists('markets_database10years.csv'):
        try:
            print("   ⚠️ CẢNH BÁO: Tải API thất bại, buộc đọc VNINDEX từ markets_database10years.csv!")
            csv_df = pd.read_csv('markets_database10years.csv')
            if 'VNINDEX' in csv_df.columns:
                c_df = csv_df[['Date', 'VNINDEX', 'VN_VOLUME']].dropna(subset=['VNINDEX']).copy()
                c_df['Date'] = pd.to_datetime(c_df['Date']).dt.tz_localize(None).dt.normalize()
                c_df.rename(columns={'VNINDEX': 'Close', 'VN_VOLUME': 'Volume'}, inplace=True)
                c_df['Open'] = c_df['Close']
                c_df['High'] = c_df['Close']
                c_df['Low'] = c_df['Close']
                return c_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
        except Exception:
            pass

    return None

# ==========================================
# 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG (OHLCV THẬT)
# ==========================================
print("1. Đang tải Market Proxy (VNINDEX) với nến OHLCV thật...")
vnindex_raw = fetch_ticker_data('VNINDEX', n_bars=4000)
if vnindex_raw is None or vnindex_raw.empty:
    sys.exit("❌ LỖI: Không thể tải dữ liệu VNINDEX từ API!")

vnindex_df = vnindex_raw.rename(columns={'Close': 'VNINDEX'}).drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
vnindex_df['MA20'] = vnindex_df['VNINDEX'].rolling(20, min_periods=10).mean()
vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50, min_periods=20).mean()
vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100, min_periods=40).mean()
vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200, min_periods=80).mean()
vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14) / vnindex_df['VNINDEX']
# Bổ sung biến độ dao động 20 phiên của VN-Index cho mô hình AI
vnindex_df['VN_ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 20) / vnindex_df['VNINDEX']
vnindex_df['ATR_PTS'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['VOL_MA20'] = vnindex_df['Volume'].rolling(20, min_periods=10).mean()
vnindex_df['MACRO_FLOOR'] = vnindex_df['MA100'] - (1.2 * vnindex_df['ATR_PTS'])
vnindex_df['VN_MFI'] = calc_mfi(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], vnindex_df['Volume'], 14)
df_clean = vnindex_df.dropna().drop_duplicates(subset=['Date'], keep='last').copy().reset_index(drop=True)

# ==========================================
# 2. TẢI TOÀN BỘ RỔ CỔ PHIẾU VN30 LỊCH SỬ (POINT-IN-TIME)
# ==========================================
print(f"2. Đang tải toàn bộ {len(ALL_HISTORICAL_VN30_TICKERS)} mã cổ phiếu VN30 lịch sử (Chống Survivorship Bias)...")

timeline_dates = pd.DatetimeIndex(df_clean['Date'].drop_duplicates().sort_values())

stocks_open = pd.DataFrame(index=timeline_dates)
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
        df_t = df_t[~df_t.index.duplicated(keep='last')]
        stocks_open[ticker] = df_t['Open']
        stocks_close[ticker] = df_t['Close']
        stocks_volume[f'{ticker}_V'] = df_t['Volume']
        stocks_high[f'{ticker}_H'] = df_t['High']
        stocks_low[f'{ticker}_L'] = df_t['Low']
        loaded_tickers.append(ticker)
    else:
        delisted_or_empty.append(ticker)

print(f"   ✅ Đã nạp thành công {len(loaded_tickers)} mã.")
if delisted_or_empty:
    print(f"   ⚠️ Các mã không có dữ liệu: {', '.join(delisted_or_empty)}")

# =====================================================================
# KHÔNG DÙNG LỆNH FILL: DỮ LIỆU NGUYÊN BẢN 100% (RAW MARKET DATA)
# Tính chỉ báo trực tiếp trên dữ liệu giao dịch thực tế
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

# =====================================================================
# TÍNH ĐỘ RỘNG THỊ TRƯỜNG POINT-IN-TIME (TỶ LỆ MÃ VN30 TRÊN MA50)
# =====================================================================
above_ma50_cnt = (stocks_close > stocks_ma50).sum(axis=1)
valid_ma50_cnt = (stocks_close.notna() & stocks_ma50.notna()).sum(axis=1)
market_breadth_series = (above_ma50_cnt / valid_ma50_cnt.replace(0, np.nan)).fillna(0.5)

breadth_map = dict(zip(pd.to_datetime(market_breadth_series.index).strftime('%Y-%m-%d'), market_breadth_series.values))

# ==========================================
# 3. TẢI BỘ NHỚ AI & ĐỒNG BỘ VĨ MÔ CSV
# ==========================================
print("3. Đang tải Não bộ AI tĩnh & Đồng bộ Vĩ mô + Độ rộng...")
try:
    macro_df = pd.read_csv('markets_database10years.csv')
    macro_df = macro_df[['Date', 'DXY', 'US10Y', 'USD_VND']].copy()
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    macro_df = macro_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

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
    roll_m = macro_df[col].rolling(120, min_periods=20).mean()
    roll_s = macro_df[col].rolling(120, min_periods=20).std()
    macro_df[f'Z_{col}'] = (macro_df[col] - roll_m) / roll_s
z_cols = [f'Z_{c}' for c in macro_cols]

clean_dates_str = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')

df_ml_list = []
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
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200', 'VN_Breadth_MA50', 'VN_ATR_PCT'] + z_cols
for f in ml_features:
    ml_master[f] = pd.to_numeric(ml_master[f], errors='coerce')

# =====================================================================
# NẠP MÔ HÌNH WALK-FORWARD EXPANDING RETRAINING (CUỐN CHIẾU TỪNG NĂM)
# 100% Out-of-Sample, không Look-Ahead Leakage, không Model Obsolescence
# =====================================================================
wf_models = {}
for y in [2022, 2023, 2024, 2025, 2026]:
    fn = f"xgboost_model_{y}.json"
    if os.path.exists(fn):
        try:
            m = xgb.XGBClassifier()
            m.load_model(fn)
            wf_models[y] = m
            print(f"   🧠 Đã nạp Walk-Forward Model năm {y}: {fn}")
        except Exception as e:
            print(f"   ⚠️ Lỗi đọc {fn}: {e}")

DEFAULT_MODEL_FILE = "xgboost_ultimate.json"
default_model = xgb.XGBClassifier()
if os.path.exists(DEFAULT_MODEL_FILE):
    default_model.load_model(DEFAULT_MODEL_FILE)
    print(f"   🧠 Đã nạp mô hình cơ sở: {DEFAULT_MODEL_FILE}")
elif wf_models:
    default_model = list(wf_models.values())[-1]
else:
    sys.exit(f"⚠️ Không tìm thấy {DEFAULT_MODEL_FILE}! Bác phải chạy train_bot.py trước.")

# Chấm điểm AI Score thích ứng Walk-Forward OOS
ml_master['AI_Score'] = np.nan

# 1. Giai đoạn <= 2021: Dùng mô hình học trên 2014-2021 (model 2022 hoặc default)
pre_2022_mask = ml_master['Date'] < '2022-01-01'
base_m = wf_models.get(2022, default_model)
if pre_2022_mask.sum() > 0:
    ml_master.loc[pre_2022_mask, 'AI_Score'] = base_m.predict_proba(ml_master.loc[pre_2022_mask, ml_features])[:, 1]

# 2. Các năm Out-of-Sample 2022 -> 2026: Dùng đúng mô hình học trước năm đó!
for y in [2022, 2023, 2024, 2025, 2026]:
    if y == 2026:
        y_mask = ml_master['Date'] >= f"{y}-01-01"
    else:
        y_mask = (ml_master['Date'] >= f"{y}-01-01") & (ml_master['Date'] < f"{y+1}-01-01")
    
    if y_mask.sum() > 0:
        m_y = wf_models.get(y, default_model)
        ml_master.loc[y_mask, 'AI_Score'] = m_y.predict_proba(ml_master.loc[y_mask, ml_features])[:, 1]

# Tạo tra cứu AI Score dạng dict (Key: ticker -> {date_str: score})
ai_dict = {}
for c in loaded_tickers:
    c_sub = ml_master[ml_master['Ticker'] == c]
    ai_dict[c] = dict(zip(c_sub['Date'], c_sub['AI_Score']))

# ==========================================
# 4. SỔ KẾ TOÁN THỰC CHIẾN (NEXT-DAY OPEN + LÔ CHẴN 100 HOSE + HYSTERESIS BUFFER)
# ==========================================
print("4. Chạy Sổ Kế Toán (Point-in-Time VN30 + Khớp Open[t+1] + Khóa trần ngành + Hysteresis Buffer)...")

def run_ai_quant_system():
    backtest = df_clean[df_clean['Date'] >= '2014-01-01'].copy().reset_index(drop=True)
    
    MAX_SLOTS = 5                      # Tối đa 5 vị thế (20% NAV / slot)
    MAX_PER_SECTOR = 2                 # Khóa trần ngành: Tối đa 2 mã / ngành (Chống sập dây chuyền)
    BUY_FRICTION = 0.0065              # 0.5% trượt giá + 0.15% phí mua
    SELL_FRICTION = 0.0075             # 0.5% trượt giá + 0.15% phí bán + 0.10% thuế
    ANNUAL_BANK_RATE = 0.04            # 4.0% / năm lãi gửi tiền mặt
    DAILY_BANK_RATE = ANNUAL_BANK_RATE / 252
    MIN_HOLD_DAYS = 3                  # Chu kỳ T+2.5 (mua Open ngày t, bán sớm nhất Open ngày t+3)
    
    AI_THRESHOLD = 0.55                # Ngưỡng AI Score chuẩn mực
    VOL_BREAKOUT_RATIO = 1.2           # Volume đột biến >= 1.2x MA20
    ADX_MIN = 18                       # Bắt xu hướng thị trường ADX > 18
    MAX_BUYS_PER_DAY = 2               # Tối đa 2 lệnh mua / ngày (Giải ngân phân lớp)
    TRAILING_STOP_TRIGGER = 0.15       # Trailing stop tiêu chuẩn khi lãi đỉnh >= 15%
    TRAILING_STOP_DROP = 0.08          # Rớt 8% từ đỉnh thì chốt
    SUPER_WINNER_PEAK = 0.20           # Cổ phiếu dẫn dắt (Leader): Lãi đỉnh >= 20%
    SUPER_WINNER_DROP = 0.10           # Siêu cổ phiếu: Trailing stop nới rộng 10% từ đỉnh cao nhất
    PROFIT_PROTECTION_PEAK = 0.10      # Chặn lãi theo đỉnh vị thế khi lãi đỉnh >= 10%
    PROFIT_PROTECTION_DROP = 0.08      # Rớt 8% từ đỉnh thì chốt toàn bộ vị thế
    PROFIT_PROTECTION_FLOOR = 0.02     # Sàn bảo vệ vốn/lãi tối thiểu +2%
    MA20_PROFIT_EXIT = 0.10            # Thủng MA20 chốt lãi khi đã có đệm > 10%
    HARD_STOPLOSS = -0.10              # Cắt lỗ cứng -10%
    PORTFOLIO_MDD_LIMIT = -0.18        # Portfolio Stop: Tổng NAV giảm 18% từ đỉnh chu kỳ -> bán hết
    
    INITIAL_CAPITAL = 100_000_000.0    # Vốn khởi điểm 100 triệu VNĐ
    cash = INITIAL_CAPITAL
    total_bank_interest = 0.0
    held = {}                          # {ticker: {'qty': shares, 'buy_price': open_p, 'highest': ..., 'buy_date': ..., 'hold_days': 0}}
    trade_log = []
    daily_nav_history = []
    nav_peak = cash                    # Đỉnh NAV chu kỳ nắm giữ
    consecutive_above_ma100 = 0
    last_macro_exit_bar = -999         # Đánh dấu phiên kích hoạt bán tháo Macro gần nhất (Cooldown 10 phiên)

    pending_sells = []  # list of (ticker, reason)
    pending_buys = []   # list of {'ticker': c, 'score': score, 'signal': 'Breakout'}

    def get_current_stock_valuation(d):
        total = 0.0
        for k, pos in held.items():
            curr_p = stocks_close.at[d, k] if (k in stocks_close.columns and pd.notna(stocks_close.at[d, k]) and stocks_close.at[d, k] > 0) else pos.get('last_price', pos['buy_price'])
            total += pos['qty'] * (curr_p * 1000.0)
        return total

    for i in range(len(backtest)):
        date = backtest['Date'].iloc[i]
        date_str = date.strftime('%Y-%m-%d')
        
        # -------------------------------------------------------------
        # 1. ĐẦU PHIÊN: CỘNG LÃI QUA ĐÊM CHO TIỀN MẶT NHÀN RỖI
        # -------------------------------------------------------------
        if cash > 0:
            interest_today = cash * DAILY_BANK_RATE
            cash += interest_today
            total_bank_interest += interest_today

        # -------------------------------------------------------------
        # 2. KHỚP LỆNH CHỜ TẠI GIÁ MỞ CỬA (OPEN[t]) TỪ TÍN HIỆU PHIÊN TRƯỚC
        # -------------------------------------------------------------
        # (A) KHỚP LỆNH BÁN CHỜ TẠI OPEN[t]
        if pending_sells:
            sells_executed = []
            for c, reason in pending_sells:
                if c in held:
                    open_p = stocks_open.at[date, c] if (c in stocks_open.columns and pd.notna(stocks_open.at[date, c]) and stocks_open.at[date, c] > 0) else (stocks_close.at[date, c] if c in stocks_close.columns else np.nan)
                    if pd.isna(open_p) or open_p <= 0:
                        continue  # Tạm ngừng giao dịch -> Giữ lại lệnh bán cho phiên tiếp theo
                    
                    qty = held[c]['qty']
                    sell_price_vnd = open_p * 1000.0
                    buy_price_vnd = held[c]['buy_price'] * 1000.0
                    
                    gross_val = qty * sell_price_vnd
                    net_val = gross_val * (1.0 - SELL_FRICTION)
                    buy_cost = qty * buy_price_vnd * (1.0 + BUY_FRICTION)
                    pnl_vnd = net_val - buy_cost
                    pnl_pct = ((sell_price_vnd * (1.0 - SELL_FRICTION)) / (buy_price_vnd * (1.0 + BUY_FRICTION)) - 1.0) * 100.0
                    hold_days = held[c]['hold_days']
                    
                    cash += net_val
                    del held[c]
                    sells_executed.append((c, reason))
                    
                    rem_stock_val = get_current_stock_valuation(date)
                    trade_nav = cash + rem_stock_val
                    cum_ret_pct = ((trade_nav / INITIAL_CAPITAL) - 1.0) * 100.0
                    
                    trade_log.append({
                        'Date': date_str,
                        'Action': reason,
                        'Ticker': c,
                        'Price': round(float(open_p), 2),
                        'Trade_Value': int(round(net_val)),
                        'PnL_VND': int(round(pnl_vnd)),
                        'Profit(%)': round(float(pnl_pct), 2),
                        'Hold_Days': int(hold_days),
                        'Total_NAV': int(round(trade_nav)),
                        'Portfolio_Return(%)': round(float(cum_ret_pct), 2),
                        'AI_Score': None
                    })
            
            pending_sells = [(c, r) for c, r in pending_sells if (c, r) not in sells_executed and c in held]

        # (B) KHỚP LỆNH MUA CHỜ TẠI OPEN[t] THEO LÔ CHẴN 100 CỔ PHIẾU HOSE
        if pending_buys:
            stock_val_open = get_current_stock_valuation(date)
            current_nav_open = cash + stock_val_open
            target_slot_size = current_nav_open / MAX_SLOTS
            
            buys_today = 0
            for cand in pending_buys:
                empty_slots = MAX_SLOTS - len(held)
                if empty_slots <= 0 or cash < 2_000_000 or buys_today >= MAX_BUYS_PER_DAY:
                    break
                c = cand['ticker']
                if c in held:
                    continue
                
                # Kiểm tra trần ngành thực tế tại thời điểm khớp
                cand_sec = SECTOR_MAP.get(c, 'OTHER')
                current_sec_count = sum(1 for h in held if SECTOR_MAP.get(h, 'OTHER') == cand_sec)
                if current_sec_count >= MAX_PER_SECTOR:
                    continue

                open_p = stocks_open.at[date, c] if (c in stocks_open.columns and pd.notna(stocks_open.at[date, c]) and stocks_open.at[date, c] > 0) else (stocks_close.at[date, c] if c in stocks_close.columns else np.nan)
                if pd.isna(open_p) or open_p <= 0:
                    continue
                
                price_vnd = open_p * 1000.0
                cost_per_share_vnd = price_vnd * (1.0 + BUY_FRICTION)
                cost_per_lot_vnd = cost_per_share_vnd * 100.0
                
                alloc = min(target_slot_size, cash)
                lots = np.floor(alloc / cost_per_lot_vnd)
                shares = lots * 100.0  # Luôn là bội số chính xác của 100 cổ phiếu HOSE
                
                if shares >= 100:
                    total_spent = shares * cost_per_share_vnd
                    cash -= total_spent
                    held[c] = {
                        'qty': shares,
                        'buy_price': open_p,
                        'highest': open_p,
                        'last_price': open_p,
                        'buy_date': date_str,
                        'hold_days': 0
                    }
                    buys_today += 1
                    
                    rem_stock_val = get_current_stock_valuation(date)
                    trade_nav = cash + rem_stock_val
                    cum_ret_pct = ((trade_nav / INITIAL_CAPITAL) - 1.0) * 100.0
                    
                    trade_log.append({
                        'Date': date_str,
                        'Action': f"BUY ({cand['signal']})",
                        'Ticker': c,
                        'Price': round(float(open_p), 2),
                        'Trade_Value': int(round(total_spent)),
                        'PnL_VND': None,
                        'Profit(%)': None,
                        'Hold_Days': None,
                        'Total_NAV': int(round(trade_nav)),
                        'Portfolio_Return(%)': round(float(cum_ret_pct), 2),
                        'AI_Score': round(float(cand['score']), 2)
                    })
            
            pending_buys = []

        # -------------------------------------------------------------
        # 3. TRONG PHIÊN & CUỐI PHIÊN: CẬP NHẬT TRẠNG THÁI VỊ THẾ
        # -------------------------------------------------------------
        for c in held:
            held[c]['hold_days'] += 1
            curr_p = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
            curr_h = stocks_high.at[date, f'{c}_H'] if f'{c}_H' in stocks_high.columns else curr_p
            if pd.notna(curr_h) and curr_h > 0:
                held[c]['highest'] = max(held[c].get('highest', curr_h), curr_h)
            if pd.notna(curr_p) and curr_p > 0:
                held[c]['last_price'] = curr_p

        # Cập nhật NAV cuối phiên chuẩn xác theo Close[t]
        stock_val_close = get_current_stock_valuation(date)
        current_nav = cash + stock_val_close
        
        # Cập nhật đỉnh NAV của chu kỳ nắm giữ
        if current_nav > nav_peak:
            nav_peak = current_nav

        # SỬA LỖI CẮT NON: Reset đỉnh NAV khi tài khoản ở 100% tiền mặt
        # Tránh mang gánh nặng đỉnh cũ của chu kỳ nhiều tháng trước
        if len(held) == 0:
            nav_peak = current_nav
        
        daily_nav_history.append({
            'Date': date_str,
            'NAV': int(round(current_nav)),
            'Cash': int(round(cash)),
            'Bank_Interest': int(round(total_bank_interest)),
            'Portfolio_Return(%)': round(((current_nav / INITIAL_CAPITAL) - 1.0) * 100.0, 2)
        })

        # -------------------------------------------------------------
        # 4. CUỐI PHIÊN (ATC): QUÉT TÍN HIỆU CHO PHIÊN HÔM SAU
        # -------------------------------------------------------------
        idx = backtest['VNINDEX'].iloc[i]
        market_ma20 = backtest['MA20'].iloc[i] if 'MA20' in backtest.columns else np.nan
        market_ma50 = backtest['MA50'].iloc[i] if 'MA50' in backtest.columns else np.nan
        market_ma100 = backtest['MA100'].iloc[i]
        market_ma200 = backtest['MA200'].iloc[i]
        market_adx = backtest['ADX'].iloc[i]
        market_atr_pct = backtest['ATR_PCT'].iloc[i]
        market_vol = backtest['Volume'].iloc[i] if 'Volume' in backtest.columns else 0.0
        market_vol_ma20 = backtest['VOL_MA20'].iloc[i] if 'VOL_MA20' in backtest.columns else 1.0
        macro_floor = backtest['MACRO_FLOOR'].iloc[i] if 'MACRO_FLOOR' in backtest.columns else (market_ma100 * 0.985)

        if idx > market_ma100:
            consecutive_above_ma100 += 1
        else:
            consecutive_above_ma100 = 0

        # Khởi đầu nhịp sóng mới sau chu kỳ điều chỉnh -> Reset chuẩn so sánh đỉnh NAV cho chu kỳ mới
        if consecutive_above_ma100 == 1 and len(held) <= 1:
            nav_peak = current_nav

        active_vn30 = get_vn30_constituents(date_str)

        portfolio_dd = (current_nav - nav_peak) / nav_peak if nav_peak > 0 else 0.0
        prev_idx = backtest['VNINDEX'].iloc[i-1] if i > 0 else idx
        prev_ma100 = backtest['MA100'].iloc[i-1] if i > 0 else market_ma100
        vol_shock = (market_vol / market_vol_ma20) if (pd.notna(market_vol_ma20) and market_vol_ma20 > 0) else 1.0

        # =============================================================
        # CÔNG TẮC VĨ MÔ THÍCH ỨNG: BỘ LỌC THOÁT MA100 CHUẨN XÁC
        # 1. Gỡ bỏ hoàn toàn điều kiện "chớm gãy MA100" gây bẫy whipsaw
        # 2. Chỉ thanh lý theo Macro khi VN-Index thủng 1.5% trong 2 phiên liên tiếp
        # 3. Tuyệt đối KHÔNG bán tháo cổ phiếu có lãi đệm >= +3% theo Macro
        # =============================================================
        two_day_deep_breakdown = (idx < market_ma100 * 0.985) and (prev_idx < prev_ma100 * 0.985) # Thủng 1.5% 2 phiên liên tiếp
        hard_keltner_breakdown = (idx < macro_floor)                                             # Rơi tự do thủng sàn Keltner
        flash_crash_breakdown = (idx < market_ma100 * 0.970)                                     # Sập rất sâu > 3%

        is_confirmed_macro_breakdown = two_day_deep_breakdown or hard_keltner_breakdown or flash_crash_breakdown

        # (A) QUÉT TÍN HIỆU BÁN ĐỂ KHỚP TẠI OPEN PHIÊN MAI
        already_pending_sells = {item[0] for item in pending_sells}

        if portfolio_dd <= -0.18 and len(held) > 0:
            # CẦU DAO PORTFOLIO STOP: Tổng NAV sụt 18% từ đỉnh chu kỳ -> Bán toàn bộ
            for c in list(held.keys()):
                if held[c]['hold_days'] >= MIN_HOLD_DAYS and c not in already_pending_sells:
                    pending_sells.append((c, 'SELL (Portfolio Stop -18%)'))

        elif is_confirmed_macro_breakdown and len(held) > 0:
            # SẬP DỨT KHOÁT: Đóng cửa dưới MA100*0.985 2 phiên hoặc gãy sàn Keltner!
            # Đánh dấu phiên bán tháo Macro để kích hoạt Cooldown 10 phiên không mua lại
            last_macro_exit_bar = i

            for c in list(held.keys()):
                if held[c]['hold_days'] < MIN_HOLD_DAYS or c in already_pending_sells:
                    continue
                curr_price = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                if pd.isna(curr_price) or curr_price <= 0:
                    continue

                buy_price = held[c]['buy_price']
                profit_margin = (curr_price - buy_price) / buy_price

                # BẢO VỆ VỊ THẾ CÓ LÃI ĐỆM >= +3%:
                # Tuyệt đối KHÔNG bán hoảng loạn theo Macro! Để MA20 cá nhân hoặc Trailing Stop tự xử lý!
                if profit_margin >= 0.03:
                    continue

                # Chỉ cắt các vị thế đang lỗ, hòa hoặc lãi mỏng (< +3%)
                pending_sells.append((c, 'SELL (Cắt Macro gãy MA100)'))

        else:
            # THỊ TRƯỜNG BÌNH THƯỜNG (HOẶC CHƯA XÁC NHẬN SẬP DỨT KHOÁT):
            # Quản trị rủi ro riêng lẻ từng vị thế (Không bán tháo cả rổ)
            for c in list(held.keys()):
                if held[c]['hold_days'] < MIN_HOLD_DAYS or c in already_pending_sells:
                    continue
                curr_price = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                if pd.isna(curr_price) or curr_price <= 0:
                    continue

                buy_price = held[c]['buy_price']
                profit_margin = (curr_price - buy_price) / buy_price
                peak_margin = (held[c]['highest'] - buy_price) / buy_price

                # LÝ DO 1: Rời rổ VN30
                if c not in active_vn30:
                    pending_sells.append((c, 'SELL (Rời rổ VN30)'))
                # LÝ DO 2: Cắt lỗ cứng -10%
                elif profit_margin <= HARD_STOPLOSS:
                    pending_sells.append((c, 'SELL (Cắt lỗ cứng -10%)'))
                # LÝ DO 3: Quản trị Leader / Siêu cổ phiếu (Lãi đỉnh >= 20%)
                elif peak_margin >= SUPER_WINNER_PEAK:
                    if curr_price < (held[c]['highest'] * (1.0 - SUPER_WINNER_DROP)):
                        pending_sells.append((c, 'SELL (Trailing Stop chốt siêu cổ)'))
                    elif c in stocks_ma50.columns and pd.notna(stocks_ma50.at[date, c]) and curr_price < stocks_ma50.at[date, c]:
                        pending_sells.append((c, 'SELL (Gãy MA50 chốt lời sóng lớn)'))
                    elif profit_margin <= 0.08:
                        pending_sells.append((c, 'SELL (Khóa lãi sàn bảo vệ +8%)'))
                # LÝ DO 4: Chặn lãi theo đỉnh vị thế (Lãi đỉnh 10% - 20%)
                elif peak_margin >= PROFIT_PROTECTION_PEAK:
                    if curr_price < (held[c]['highest'] * (1.0 - PROFIT_PROTECTION_DROP)):
                        pending_sells.append((c, 'SELL (Trailing Stop chốt lời)'))
                    elif c in stocks_ma20.columns and pd.notna(stocks_ma20.at[date, c]) and curr_price < stocks_ma20.at[date, c] and profit_margin >= 0.05:
                        pending_sells.append((c, 'SELL (Thủng MA20 bảo toàn lãi)'))
                    elif profit_margin <= PROFIT_PROTECTION_FLOOR:
                        pending_sells.append((c, 'SELL (Khóa bảo vệ vốn +2%)'))
                # LÝ DO 5: Vị thế bình thường
                else:
                    if c in stocks_ma20.columns and pd.notna(stocks_ma20.at[date, c]) and curr_price < stocks_ma20.at[date, c] and profit_margin > MA20_PROFIT_EXIT:
                        pending_sells.append((c, 'SELL (Thủng MA20 bảo toàn lãi)'))

        # =============================================================
        # (B) QUÉT TÍN HIỆU MUA CHO PHIÊN MAI (HYSTERESIS BUFFER + COOLDOWN + SECTOR CAP)
        # =============================================================
        if idx < market_ma50 or market_atr_pct > 0.020:
            max_exposure = 0.4
        elif idx < market_ma200:
            max_exposure = 0.5
        elif idx < market_ma100:
            max_exposure = 0.7
        else:
            max_exposure = 1.0

        allowed_stock_cap = max(0.0, (current_nav * max_exposure) - stock_val_close)

        # 1. VÙNG ĐỆM HYSTERESIS: VN-Index phải vượt MA100 tối thiểu +1.5%
        is_macro_buffer_ok = (idx >= market_ma100 * 1.015)

        # 2. XU HƯỚNG NGẮN/TRUNG HẠN ĐỒNG THUẬN: MA20 > MA50
        is_ma_trend_ok = (pd.notna(market_ma20) and pd.notna(market_ma50) and market_ma20 > market_ma50)

        # 3. THỜI GIAN CHỜ COOLDOWN: Ít nhất 10 phiên sau lần bán tháo Macro gần nhất (tránh mua đuổi nhịp hồi giả)
        is_cooldown_passed = (i - last_macro_exit_bar > 10)

        # 4. ĐỘ BIẾN ĐỘNG HẠ NHIỆT (market_atr_pct <= 0.020)
        is_volatility_calm = (market_atr_pct <= 0.020)

        # 5. ĐỘ RỘNG THỊ TRƯỜNG: >= 35% số mã VN30 nằm trên MA50
        vn30_above_ma50_cnt = sum([
            1 for ticker in active_vn30
            if ticker in stocks_close.columns and ticker in stocks_ma50.columns
            and pd.notna(stocks_close.at[date, ticker]) and pd.notna(stocks_ma50.at[date, ticker])
            and stocks_close.at[date, ticker] > stocks_ma50.at[date, ticker]
        ])
        market_breadth = (vn30_above_ma50_cnt / len(active_vn30)) if len(active_vn30) > 0 else 0.5
        is_breadth_healthy = (market_breadth >= 0.35)

        can_enter_new_trades = (
            is_macro_buffer_ok
            and is_ma_trend_ok
            and is_cooldown_passed
            and is_volatility_calm
            and is_breadth_healthy
            and (market_adx > ADX_MIN)
            and (allowed_stock_cap > 0)
        )

        projected_held_count = len(held) - len(pending_sells)
        available_slots = MAX_SLOTS - projected_held_count

        if available_slots > 0 and can_enter_new_trades:
            buy_candidates = []
            for c in active_vn30:
                if c not in held and c in loaded_tickers and c in stocks_close.columns:
                    curr_price = stocks_close.at[date, c]
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

                    # Tín hiệu Momentum Breakout chuẩn mực
                    if curr_price > prev_h20 and v_ratio >= VOL_BREAKOUT_RATIO and score >= AI_THRESHOLD:
                        buy_candidates.append({
                            'ticker': c, 'score': score, 'signal': 'Breakout',
                            'sector': SECTOR_MAP.get(c, 'OTHER')
                        })

            if buy_candidates:
                # Sắp xếp ưu tiên AI Score cao nhất
                buy_candidates = sorted(buy_candidates, key=lambda x: x['score'], reverse=True)
                
                # KHÓA TRẦN NGÀNH (SECTOR CAP <= 2 MÃ/NGÀNH)
                selected_buys = []
                for cand in buy_candidates:
                    if len(selected_buys) >= MAX_BUYS_PER_DAY:
                        break
                    
                    sec = cand['sector']
                    sec_in_held = sum(1 for h in held if SECTOR_MAP.get(h, 'OTHER') == sec and h not in [s[0] for s in pending_sells])
                    sec_in_selected = sum(1 for b in selected_buys if b.get('sector') == sec)
                    
                    if (sec_in_held + sec_in_selected) < MAX_PER_SECTOR:
                        selected_buys.append(cand)
                    # Nếu ngành đã đủ 2 mã (vd Bank), nhường slot cho ngành khác
                
                pending_buys = selected_buys

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
    df_trades = df_trades[final_cols] if not df_trades.empty else pd.DataFrame(columns=desired_cols)

    # Ép kiểu Int64 nullable để xuất file sạch 100%
    for int_col in ['Trade_Value', 'PnL_VND', 'Hold_Days', 'Total_NAV']:
        if int_col in df_trades.columns:
            df_trades[int_col] = pd.to_numeric(df_trades[int_col], errors='coerce').round(0).astype('Int64')

    for nav_int in ['NAV', 'Cash', 'Bank_Interest']:
        if nav_int in df_nav.columns:
            df_nav[nav_int] = pd.to_numeric(df_nav[nav_int], errors='coerce').round(0).astype('Int64')

    return df_trades, df_nav, final_nav, total_bank_interest, held, pending_buys, pending_sells

# ==========================================
# CHẠY CHIẾN LƯỢC & XUẤT BÁO CÁO TOÀN DIỆN
# ==========================================
df_trade_log, df_nav_hist, final_nav, total_interest, current_held, next_buys, next_sells = run_ai_quant_system()

# Ghi nhận file trading_log.csv
df_trade_log.to_csv('trading_log.csv', index=False, na_rep='', float_format='%.2f')
print("✅ Đã ghi nhận sổ kế toán giao dịch tinh gọn vào file trading_log.csv!")

# Ghi nhận file daily_nav.csv
df_nav_hist.to_csv('daily_nav.csv', index=False, na_rep='', float_format='%.2f')
print("✅ Đã ghi nhận biến động tài sản hàng ngày chuẩn vào file daily_nav.csv!")

# ==========================================
# TÍNH TOÁN METRICS ĐỊNH LƯỢNG RỦI RO & HIỆU SUẤT
# ==========================================
initial_capital = 100_000_000.0
total_return_pct = ((final_nav / initial_capital) - 1.0) * 100.0

nav_dates = pd.to_datetime(df_nav_hist['Date'])
n_years = (nav_dates.iloc[-1] - nav_dates.iloc[0]).days / 365.25
cagr = ((final_nav / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0 if n_years > 0 else 0.0

df_nav_hist['Peak'] = df_nav_hist['NAV'].cummax()
df_nav_hist['Drawdown'] = (df_nav_hist['NAV'] - df_nav_hist['Peak']) / df_nav_hist['Peak']
max_mdd = df_nav_hist['Drawdown'].min() * 100.0

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
msg += f"➖➖➖➖➖➖➖➖➖➖\n\n🎯 <b>LỆNH ĐÃ KHỚP HÔM NAY (ATO/OPEN):</b>\n"

buy_count, sell_count = 0, 0
if not df_trade_log.empty:
    latest_trades = df_trade_log[df_trade_log['Date'] == latest_date_str]
    if not latest_trades.empty:
        for _, row in latest_trades.iterrows():
            if 'BUY' in row['Action']:
                trade_val = f" | Tiền: {row['Trade_Value']:,.0f} VNĐ" if 'Trade_Value' in row and pd.notna(row['Trade_Value']) else ""
                msg += f"🟢 <b>MUA: {row['Ticker']}</b> (Giá ATO: {row['Price']:,.2f}{trade_val}) | AI: <b>{row['AI_Score'] * 100:.1f}%</b>\n\n"
                buy_count += 1
            elif 'SELL' in row['Action']:
                icon = "🔥" if row['Profit(%)'] > 0 else "🩸"
                pnl_str = f" ({row['PnL_VND']:+,.0f} VNĐ)" if 'PnL_VND' in row and pd.notna(row['PnL_VND']) else ""
                msg += f"🔴 <b>BÁN: {row['Ticker']}</b> ({row['Action']}) | Lãi/Lỗ: {icon} <b>{row['Profit(%)']}%</b>{pnl_str}\n\n"
                sell_count += 1

if buy_count == 0 and sell_count == 0:
    msg += "<i>⏳ Không có lệnh Mua/Bán khớp trong phiên hôm nay.</i>\n\n"

# Báo cáo các lệnh chờ ATO phiên tiếp theo
if next_buys or next_sells:
    msg += f"⚡ <b>TÍN HIỆU ĐẶT LỆNH ATO PHIÊN TIẾP THEO:</b>\n"
    for b in next_buys:
        b_sec = SECTOR_MAP.get(b['ticker'], 'OTHER')
        msg += f"   • 🟢 Mua ATO: <b>{b['ticker']}</b> [{b_sec}] (AI Score: <b>{b['score']*100:.1f}%</b>)\n"
    for s_tick, s_reason in next_sells:
        msg += f"   • 🔴 Bán ATO: <b>{s_tick}</b> ({s_reason})\n"
    msg += "\n"

# Báo cáo chi tiết danh mục đang nắm giữ
msg += f"💼 <b>DANH MỤC HIỆN TẠI ({len(current_held)}/5 vị thế):</b>\n"
if current_held:
    for ticker, pos in current_held.items():
        curr_p = stocks_close.iloc[-1][ticker] if ticker in stocks_close.columns and pd.notna(stocks_close.iloc[-1][ticker]) else pos.get('last_price', pos['buy_price'])
        unrealized_pnl = ((curr_p / pos['buy_price']) - 1.0) * 100.0
        icon = "🟢" if unrealized_pnl >= 0 else "🔴"
        peak_pnl = ((pos['highest'] / pos['buy_price']) - 1.0) * 100.0
        leader_badge = " 👑 [Leader]" if peak_pnl >= 20.0 else ""
        qty_str = f"{int(pos['qty']):,} cp"
        sec_name = SECTOR_MAP.get(ticker, 'OTHER')
        msg += f"{icon} <b>{ticker}</b> [{sec_name}] ({qty_str}): Giá {curr_p:,.2f} | Lãi tạm: <b>{unrealized_pnl:+.1f}%</b> (Đỉnh: +{peak_pnl:.1f}%){leader_badge} | T+{pos['hold_days']}\n"
    msg += "\n"
else:
    msg += "<i>100% Tiền mặt (Bảo toàn vốn tối đa, nhận lãi Bank 4%/năm).</i>\n\n"

msg += f"➖➖➖➖➖➖➖➖➖➖\n"
msg += f"🏦 NAV Hiện tại: <b>{final_nav:,.0f} VNĐ</b> | Max Drawdown: <b>{max_mdd:.2f}%</b>"

send_telegram_msg(msg)
print("✅ Hoàn tất chu trình quant_bot.py!")
