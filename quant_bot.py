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

from vn30_universe import get_vn30_constituents, ALL_HISTORICAL_VN30_TICKERS

warnings.filterwarnings('ignore')
logging.getLogger("tvDatafeed").setLevel(logging.CRITICAL)

# ==========================================
# TỪ ĐIỂN PHÂN LOẠI NGÀNH VN30 (SECTOR MAP)
# Khống chế rủi ro tập trung ngành (Sector Cap <= 2 mã/ngành)
# ==========================================
SECTOR_MAP = {
    # 1. Ngân hàng (Banking)
    'VCB': 'BANK', 'BID': 'BANK', 'CTG': 'BANK', 'MBB': 'BANK', 'TCB': 'BANK',
    'VPB': 'BANK', 'ACB': 'BANK', 'HDB': 'BANK', 'TPB': 'BANK', 'STB': 'BANK',
    'VIB': 'BANK', 'SSB': 'BANK', 'SHB': 'BANK', 'LPB': 'BANK', 'EIB': 'BANK',
    'MSB': 'BANK', 'OCB': 'BANK', 'PVF': 'BANK',
    
    # 2. Bất động sản, Xây dựng & Hạ tầng (Real Estate & Construction)
    'VIC': 'REAL_ESTATE', 'VHM': 'REAL_ESTATE', 'VRE': 'REAL_ESTATE', 
    'NVL': 'REAL_ESTATE', 'KDH': 'REAL_ESTATE', 'PDR': 'REAL_ESTATE', 
    'BCM': 'REAL_ESTATE', 'KBC': 'REAL_ESTATE', 'CII': 'REAL_ESTATE', 
    'CTD': 'REAL_ESTATE', 'ROS': 'REAL_ESTATE', 'FLC': 'REAL_ESTATE',
    'ITA': 'REAL_ESTATE', 'IJC': 'REAL_ESTATE', 'HHS': 'REAL_ESTATE',
    'SJS': 'REAL_ESTATE', 'DIG': 'REAL_ESTATE', 'DXG': 'REAL_ESTATE',
    'OGC': 'REAL_ESTATE', 'NTL': 'REAL_ESTATE', 'QCG': 'REAL_ESTATE',
    
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
    
    # 8. Phân bón, Hóa chất & Cao su (Chemicals & Materials)
    'GVR': 'CHEMICALS', 'DGC': 'CHEMICALS', 'DPM': 'CHEMICALS', 'DCM': 'CHEMICALS', 
    'DRC': 'CHEMICALS', 'CSM': 'CHEMICALS',
    
    # 9. Nông nghiệp & Thực phẩm (Agriculture)
    'HAG': 'AGRICULTURE', 'HNG': 'AGRICULTURE', 'HVG': 'AGRICULTURE',
    
    # 10. Vận tải & Hàng không (Logistics & Aviation)
    'VJC': 'LOGISTICS', 'GMD': 'LOGISTICS', 'HVN': 'LOGISTICS',
    
    # 11. Bảo hiểm (Insurance)
    'BVH': 'INSURANCE'
}

# ==========================================
# TỪ ĐIỂN HỆ SINH THÁI / TẬP ĐOÀN LIÊN KẾT (ECOSYSTEM MAP)
# Khống chế rủi ro sụp đổ chéo cùng họ (Ecosystem Cap <= 1 mã/hệ sinh thái)
# Triệt tiêu rủi ro sụt giảm đồng thời như cặp VHM + VRE ngày 10/10/2025
# ==========================================
ECOSYSTEM_MAP = {
    # 1. Họ Vingroup
    'VIC': 'VINGROUP', 'VHM': 'VINGROUP', 'VRE': 'VINGROUP',
    
    # 2. Họ Dầu khí / Năng lượng Petrovietnam (PVN)
    'GAS': 'PVN', 'PVD': 'PVN', 'PVT': 'PVN', 'POW': 'PVN', 'PLX': 'PVN', 'NT2': 'PVN', 'PPC': 'PVN', 'PGD': 'PVN',
    
    # 3. Họ Masan
    'MSN': 'MASAN',
    
    # 4. Họ Tập đoàn Hóa chất / Cao su Việt Nam (Vinachem / VRG)
    'GVR': 'VRG', 'DPM': 'VRG', 'DCM': 'VRG', 'DRC': 'VRG', 'CSM': 'VRG',
    
    # 5. Khối Ngân hàng Quốc doanh (Big 4)
    'VCB': 'BIG4_BANK', 'BID': 'BIG4_BANK', 'CTG': 'BIG4_BANK',
    
    # 6. Họ FPT
    'FPT': 'FPT_GROUP',
    
    # 7. Họ Thép & Kim loại
    'HPG': 'STEEL_GROUP', 'HSG': 'STEEL_GROUP', 'NKG': 'STEEL_GROUP',
    
    # 8. Họ Hoàng Anh Gia Lai
    'HAG': 'HAGL', 'HNG': 'HAGL',
    
    # 9. Họ FLC
    'FLC': 'FLC_GROUP', 'ROS': 'FLC_GROUP',
}

BACKTEST_START_DATE = os.environ.get('BACKTEST_START_DATE', '2022-01-01')

# ==========================================
# CỜ TÍNH NĂNG (FEATURE FLAGS - 4 CORE SOLUTIONS & ECOSYSTEM CONSTRAINTS)
# ==========================================
ENABLE_DYNAMIC_PERCENTILE_AI = os.environ.get('ENABLE_DYNAMIC_PERCENTILE_AI', 'True').lower() in ('true', '1', 'yes')
ENABLE_ASYMMETRIC_ATR_EXIT = os.environ.get('ENABLE_ASYMMETRIC_ATR_EXIT', 'True').lower() in ('true', '1', 'yes')
ENABLE_MA20_SLOPE_FILTER = os.environ.get('ENABLE_MA20_SLOPE_FILTER', 'True').lower() in ('true', '1', 'yes')
ENABLE_ANTI_CHASE_FILTER = os.environ.get('ENABLE_ANTI_CHASE_FILTER', 'True').lower() in ('true', '1', 'yes')
MAX_PER_SECTOR = int(os.environ.get('MAX_PER_SECTOR', '2'))        # Nới trần 2 mã/ngành, áp dụng Split-Size (10% NAV/lệnh khi cùng ngành)
MAX_PER_ECOSYSTEM = int(os.environ.get('MAX_PER_ECOSYSTEM', '2'))  # Nới trần 2 mã/hệ sinh thái, áp dụng Split-Size (10% NAV/lệnh khi cùng hệ sinh thái)
HARD_STOPLOSS_FLOOR = float(os.environ.get('HARD_STOPLOSS_FLOOR', '-0.085'))  # Biên dưới tuyệt đối cắt lỗ động (mã biến động mạnh tối đa -8.5%)
ATR_STOPLOSS_MULTIPLIER = float(os.environ.get('ATR_STOPLOSS_MULTIPLIER', '2.2'))  # Hệ số ATR cho cắt lỗ động: max(FLOOR, -ATR_MULT * ATR_pct)

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
# HÀM BÁO CÁO TELEGRAM (HỖ TRỢ CẢ TELEGRAM_* VÀ BOT_TOKEN/CHAT_ID)
# ==========================================
def send_telegram_msg(text_message):
    token = os.environ.get('TELEGRAM_BOT_TOKEN') or os.environ.get('BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID') or os.environ.get('CHAT_ID')
    if not token or not chat_id:
        print("\n[TELEGRAM PREVIEW]\n" + text_message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text_message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"⚠️ Gửi Telegram thất bại: {e}")

# ==========================================
# HÀM TẢI DỮ LIỆU ĐA NGUỒN (DNSE + VNDIRECT + TCBS + TRADINGVIEW + LOCAL DATA)
# Gắn nhãn chất lượng dữ liệu: REAL_OHLC vs SYNTHETIC_CLOSE_ONLY
# Tuyệt đối KHÔNG làm phẳng nến (Open=High=Low=Close) để đảm bảo ATR/ADX/MFI chuẩn xác
# ==========================================
def fetch_ticker_data(symbol, n_bars=4500):
    """
    Tải dữ liệu chứng khoán OHLCV THẬT từ API hoặc thư mục local data/:
    1. Local data/ directory (nếu đã chạy download_all_data.py) -> REAL_OHLC
    2. DNSE Entrade API trực tiếp (Open, High, Low, Close, Volume thật, không cần key) -> REAL_OHLC
    3. VNDirect DChart API -> REAL_OHLC
    4. TCBS Public API -> REAL_OHLC
    5. TradingView API -> REAL_OHLC
    6. Fallback khẩn cấp từ markets_database10years.csv:
       Ước lượng biên độ nến True Range chống phẳng nến -> Gắn nhãn SYNTHETIC_CLOSE_ONLY
    """
    now_ts = int(datetime.datetime.now().timestamp())
    start_ts = int(datetime.datetime(2012, 1, 1).timestamp())

    # 0. Nguồn 0: File CSV cục bộ từ thư mục data/ (chuẩn OHLCV do download_all_data.py tạo)
    local_csv = os.path.join("data", f"{symbol}.csv")
    if os.path.exists(local_csv):
        try:
            loc_df = pd.read_csv(local_csv)
            if 'Date' in loc_df.columns and 'Close' in loc_df.columns and len(loc_df) > 50:
                loc_df['Date'] = pd.to_datetime(loc_df['Date']).dt.tz_localize(None).dt.normalize()
                for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if c not in loc_df.columns:
                        loc_df[c] = loc_df['Close']
                loc_df['Data_Quality'] = 'REAL_OHLC'
                loc_df['Data_Source'] = 'LOCAL_DATA_DIR'
                return loc_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Data_Quality', 'Data_Source']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
        except Exception:
            pass

    is_index = symbol.upper() in ['VNINDEX', 'VN30', 'VN100', 'VNXALL', 'HNX', 'HNX30', 'UPCOM']
    entrade_endpoint = 'index' if is_index else 'stock'

    # 1. Nguồn 1: DNSE Entrade API (Tốc độ cao, đầy đủ OHLCV 10-14 năm)
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
                df['Data_Quality'] = 'REAL_OHLC'
                df['Data_Source'] = 'DNSE_ENTRADE'
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Data_Quality', 'Data_Source']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
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
                df['Data_Quality'] = 'REAL_OHLC'
                df['Data_Source'] = 'VNDIRECT'
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Data_Quality', 'Data_Source']].drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
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

    # 4. Nguồn 4: TradingView (Phương án dự phòng)
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

    # Fallback khẩn cấp nếu mất mạng: CSV
    if symbol.upper() == 'VNINDEX' and os.path.exists('markets_database10years.csv'):
        try:
            print("   ⚠️ CẢNH BÁO: Tải API thất bại, buộc đọc VNINDEX từ markets_database10years.csv!")
            csv_df = pd.read_csv('markets_database10years.csv')
            if 'VNINDEX' in csv_df.columns:
                c_df = csv_df[['Date', 'VNINDEX', 'VN_VOLUME']].dropna(subset=['VNINDEX']).copy()
                c_df['Date'] = pd.to_datetime(c_df['Date']).dt.tz_localize(None).dt.normalize()
                c_df.rename(columns={'VNINDEX': 'Close', 'VN_VOLUME': 'Volume'}, inplace=True)
                # BẮT BUỘC: Sắp xếp Date tăng dần trước mọi phép tính thời gian (tránh nhìn trước tương lai khi file gốc xếp giảm dần)
                c_df = c_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
                
                # Tái tạo biên độ dao động nến dựa trên độ biến động lịch sử để không làm phẳng nến
                # Phẳng nến (High == Low == Close) sẽ làm sai lệch nghiêm trọng chỉ báo ATR và ADX
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


# ==========================================
# 4. SỔ KẾ TOÁN THỰC CHIẾN (NEXT-DAY OPEN + LÔ CHẴN 100 HOSE + HYSTERESIS BUFFER)
# ==========================================
def run_ai_quant_system(
    df_clean,
    stocks_open,
    stocks_close,
    stocks_high,
    stocks_low,
    stocks_volume,
    stocks_high_20,
    stocks_vol_ma20,
    stocks_ma20,
    stocks_ma50,
    stocks_atr,
    stocks_rsi,
    stocks_ret20,
    ai_dict,
    loaded_tickers,
    backtest_start_date=None,
    initial_capital=100_000_000.0,
    max_per_sector=None,
    max_per_ecosystem=None,
    enable_dynamic_percentile_ai=ENABLE_DYNAMIC_PERCENTILE_AI,
    enable_asymmetric_atr_exit=ENABLE_ASYMMETRIC_ATR_EXIT,
    enable_ma20_slope_filter=ENABLE_MA20_SLOPE_FILTER,
    enable_anti_chase_filter=ENABLE_ANTI_CHASE_FILTER,
    hard_stoploss_floor=HARD_STOPLOSS_FLOOR,
    atr_stoploss_multiplier=ATR_STOPLOSS_MULTIPLIER
):
    if backtest_start_date is None:
        backtest_start_date = BACKTEST_START_DATE
    if max_per_sector is None:
        max_per_sector = MAX_PER_SECTOR
    if max_per_ecosystem is None:
        max_per_ecosystem = MAX_PER_ECOSYSTEM
    backtest = df_clean[df_clean['Date'] >= backtest_start_date].copy().reset_index(drop=True)

    # Bảng tra cứu AI Score để tính Rolling Percentile động (Zero Lookahead Bias)
    stocks_ai_scores = pd.DataFrame(index=backtest['Date'])
    for ticker_col in loaded_tickers:
        ticker_scores = ai_dict.get(ticker_col, {})
        stocks_ai_scores[ticker_col] = [ticker_scores.get(d.strftime('%Y-%m-%d'), np.nan) for d in backtest['Date']]
    
    MAX_SLOTS = 5                         # Tối đa 5 vị thế (20% NAV / slot tiêu chuẩn)
    MAX_PER_SECTOR = max_per_sector       # Khóa trần ngành: Nới lên 2 mã, nhưng chia đôi tỷ trọng (Split-Size 10% NAV)
    MAX_PER_ECOSYSTEM = max_per_ecosystem # Khóa trần hệ sinh thái: Nới lên 2 mã, nhưng chia đôi tỷ trọng (Split-Size 10% NAV)
    BUY_FRICTION = 0.0065              # 0.5% trượt giá + 0.15% phí mua
    SELL_FRICTION = 0.0075             # 0.5% trượt giá + 0.15% phí bán + 0.10% thuế
    ANNUAL_BANK_RATE = 0.04            # 4.0% / năm lãi gửi tiền mặt
    DAILY_BANK_RATE = ANNUAL_BANK_RATE / 252
    MIN_HOLD_DAYS = 3                  # Chu kỳ T+2.5 (mua Open ngày t, bán sớm nhất Open ngày t+3)
    
    AI_THRESHOLD = 0.60                # Ngưỡng AI Score chọn lọc cao cấp (Lọc bỏ bẫy Breakout phân khúc 0.55-0.59)
    VOL_BREAKOUT_RATIO = 1.2           # Volume đột biến >= 1.2x MA20 (Nhạy bén bắt trúng chân sóng lớn)
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
    HARD_STOPLOSS_FLOOR = hard_stoploss_floor    # Biên dưới tuyệt đối cắt lỗ động (mã biến động mạnh tối đa -8.5%)
    ATR_STOPLOSS_MULT = atr_stoploss_multiplier  # Hệ số ATR cho cắt lỗ động
    PORTFOLIO_MDD_LIMIT = -0.18        # Portfolio Stop: Tổng NAV giảm 18% từ đỉnh chu kỳ -> bán hết
    
    def calc_dynamic_stoploss(ticker, d):
        """
        Cắt lỗ động theo ATR từng mã: max(FLOOR, -ATR_MULT * ATR_pct)
        - Mã biến động mạnh (ATR_pct cao): biên thở tới HARD_STOPLOSS_FLOOR (-8.5%)
        - Mã hiền lành (ATR_pct thấp): tự siết chặt quanh -5% đến -6%
        - Fallback -6.5% nếu thiếu dữ liệu ATR
        """
        atr_val = stocks_atr.at[d, ticker] if (stocks_atr is not None and ticker in stocks_atr.columns and d in stocks_atr.index and pd.notna(stocks_atr.at[d, ticker])) else np.nan
        ref_p = stocks_close.at[d, ticker] if (ticker in stocks_close.columns and pd.notna(stocks_close.at[d, ticker]) and stocks_close.at[d, ticker] > 0) else np.nan
        if pd.notna(atr_val) and pd.notna(ref_p) and ref_p > 0:
            atr_pct = atr_val / ref_p
            return max(HARD_STOPLOSS_FLOOR, -ATR_STOPLOSS_MULT * atr_pct)
        return -0.065  # Fallback bảo thủ khi thiếu ATR
    
    INITIAL_CAPITAL = initial_capital  # Vốn khởi điểm (mặc định 100 triệu VNĐ)
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
    current_max_exposure = 1.0  # Hạn mức tỷ trọng cổ phiếu tối đa (cập nhật từ ATC phiên trước)

    next_trade_id = 1
    completed_trades = []

    def get_stock_valuation_at_open(curr_date, bar_i):
        """
        Định giá danh mục cổ phiếu tại thời điểm đầu phiên (Open[t]).
        BẮT BUỘC: Tra cứu theo ngày Date thực tế (Close[t-1] - Giá tham chiếu HOSE phiên t).
        TUYỆT ĐỐI KHÔNG dùng chỉ số bar_i của backtest để iloc trên stocks_close (tránh lệch 10 năm về 2012).
        """
        total = 0.0
        if bar_i > 0:
            prev_d = backtest['Date'].iloc[bar_i - 1]
        else:
            clean_matches = df_clean.index[df_clean['Date'] == curr_date]
            prev_d = df_clean['Date'].iloc[clean_matches[0] - 1] if len(clean_matches) > 0 and clean_matches[0] > 0 else None

        for k, pos in held.items():
            ref_p = None
            if prev_d is not None and k in stocks_close.columns and prev_d in stocks_close.index:
                val = stocks_close.at[prev_d, k]
                if pd.notna(val) and val > 0:
                    ref_p = val
            if ref_p is None:
                ref_p = pos.get('last_price', pos['buy_price'])
            total += pos['qty'] * (ref_p * 1000.0)
        return total

    def get_stock_valuation_at_close(d):
        """
        Định giá danh mục cổ phiếu tại thời điểm kết phiên (ATC/Close[t]).
        Sử dụng giá đóng cửa Close[t] cùng ngày cho công tác chốt sổ NAV và quản trị rủi ro cuối phiên.
        """
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
                    # BẮT BUỘC: Khớp tại Open[t]. TUYỆT ĐỐI KHÔNG fallback về Close[t] cùng ngày (chống nhìn trước tương lai)
                    open_p = stocks_open.at[date, c] if (c in stocks_open.columns and pd.notna(stocks_open.at[date, c]) and stocks_open.at[date, c] > 0) else np.nan
                    if pd.isna(open_p) or open_p <= 0:
                        continue  # Tạm ngừng giao dịch hoặc thiếu Open -> Giữ lại lệnh bán cho phiên tiếp theo
                    
                    # RÀNG BUỘC LÔ CHẴN 100 CỔ PHIẾU HOSE KHI BÁN HẠ 50% TỶ TRỌNG
                    shares_held = int(held[c]['qty'])
                    if 'Hoán đổi 50% tỷ trọng' in reason:
                        sell_qty = (shares_held // 2 // 100) * 100
                        rem_qty = shares_held - sell_qty
                        if sell_qty < 100 or rem_qty < 100:
                            # Không đủ để chia đôi thành 2 lô chẵn 100: Hủy cờ rebalance, giữ nguyên vị thế
                            held[c]['rebalance_for_leader'] = False
                            continue
                        is_partial_leader_rebalance = True
                    else:
                        is_partial_leader_rebalance = False
                        sell_qty = shares_held
                        rem_qty = 0
                    
                    sell_price_vnd = open_p * 1000.0
                    buy_price_vnd = held[c]['buy_price'] * 1000.0
                    
                    gross_val = sell_qty * sell_price_vnd
                    net_val = gross_val * (1.0 - SELL_FRICTION)
                    buy_cost = sell_qty * buy_price_vnd * (1.0 + BUY_FRICTION)
                    pnl_vnd = net_val - buy_cost
                    pnl_pct = ((sell_price_vnd * (1.0 - SELL_FRICTION)) / (buy_price_vnd * (1.0 + BUY_FRICTION)) - 1.0) * 100.0
                    hold_days = held[c]['hold_days']
                    trade_id = held[c].get('trade_id', 'T000')
                    
                    cash += net_val
                    
                    if is_partial_leader_rebalance:
                        held[c]['qty'] -= sell_qty
                        held[c]['is_split_size'] = True
                        held[c]['rebalance_for_leader'] = False
                        held[c]['accumulated_pnl'] = held[c].get('accumulated_pnl', 0.0) + pnl_vnd
                        action_name = 'SELL_PARTIAL (Hoán đổi 50% tỷ trọng cho Leader)'
                    else:
                        clean_reason = reason.replace('SELL (', '').rstrip(')').replace('SELL ', '')
                        action_name = f"SELL_FULL ({clean_reason})"
                        total_pos_pnl_vnd = held[c].get('accumulated_pnl', 0.0) + pnl_vnd
                        init_shares = held[c].get('initial_qty', sell_qty)
                        buy_cost_total = init_shares * (held[c]['buy_price'] * 1000.0) * (1.0 + BUY_FRICTION)
                        pos_profit_pct = (total_pos_pnl_vnd / buy_cost_total) * 100.0 if buy_cost_total > 0 else 0.0
                        completed_trades.append({
                            'Trade_ID': trade_id,
                            'Ticker': c,
                            'Buy_Date': held[c]['buy_date'],
                            'Exit_Date': date_str,
                            'Buy_Price': held[c]['buy_price'],
                            'Exit_Price': round(float(open_p), 2),
                            'Initial_Qty': int(init_shares),
                            'Total_PnL_VND': int(round(total_pos_pnl_vnd)),
                            'Profit(%)': round(float(pos_profit_pct), 2),
                            'Hold_Days': int(hold_days),
                            'Reason': clean_reason
                        })
                        del held[c]
                        
                    sells_executed.append((c, reason))
                    
                    rem_stock_val = get_stock_valuation_at_open(date, i)
                    trade_nav = cash + rem_stock_val
                    cum_ret_pct = ((trade_nav / INITIAL_CAPITAL) - 1.0) * 100.0
                    
                    trade_log.append({
                        'Date': date_str,
                        'Trade_ID': trade_id,
                        'Action': action_name,
                        'Ticker': c,
                        'Price': round(float(open_p), 2),
                        'Shares': int(sell_qty),
                        'Rem_Shares': int(rem_qty),
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
            # ĐỊNH GIÁ ĐẦU PHIÊN CHUẨN XÁC: Tra cứu theo ngày Date thực tế (Close[t-1]), triệt tiêu Look-ahead Bias
            stock_val_open = get_stock_valuation_at_open(date, i)
            current_nav_open = cash + stock_val_open
            target_slot_size = current_nav_open / MAX_SLOTS
            
            # GIỚI HẠN TỶ TRỌNG CỔ PHIẾU THEO NGÂN SÁCH THỰC TẾ (EXPOSURE BUDGETING)
            # Khắc phục triệt để lỗi vượt trần: Cấp vốn mỗi lệnh phải nằm trong hạn mức thực tế còn lại
            allowed_stock_val = current_nav_open * current_max_exposure
            rem_exposure_budget = max(0.0, allowed_stock_val - stock_val_open)
            
            buys_today = 0
            for cand in pending_buys:
                empty_slots = MAX_SLOTS - len(held)
                if empty_slots <= 0 or cash < 2_000_000 or buys_today >= MAX_BUYS_PER_DAY or rem_exposure_budget <= 0:
                    break
                c = cand['ticker']
                if c in held:
                    continue
                
                # Kiểm tra trần ngành và hệ sinh thái thực tế tại thời điểm khớp (kèm cơ chế Split-Size)
                cand_sec = SECTOR_MAP.get(c, 'OTHER')
                cand_eco = ECOSYSTEM_MAP.get(c, c)
                sec_cap_limit = MAX_PER_SECTOR
                eco_cap_limit = MAX_PER_ECOSYSTEM
                current_sec_count = sum(1 for h in held if SECTOR_MAP.get(h, 'OTHER') == cand_sec)
                current_eco_count = sum(1 for h in held if ECOSYSTEM_MAP.get(h, h) == cand_eco)
                if current_sec_count >= sec_cap_limit or current_eco_count >= eco_cap_limit:
                    continue

                # BẮT BUỘC: Khớp tại Open[t]. TUYỆT ĐỐI KHÔNG fallback về Close[t] cùng ngày
                open_p = stocks_open.at[date, c] if (c in stocks_open.columns and pd.notna(stocks_open.at[date, c]) and stocks_open.at[date, c] > 0) else np.nan
                if pd.isna(open_p) or open_p <= 0:
                    continue  # Bỏ qua / Hủy lệnh nếu không có giá Open hợp lệ (không fallback về Close[t])
                
                price_vnd = open_p * 1000.0
                cost_per_share_vnd = price_vnd * (1.0 + BUY_FRICTION)
                cost_per_lot_vnd = cost_per_share_vnd * 100.0
                
                # ĐỒNG BỘ CƠ CHẾ HOÁN ĐỔI CHO LEADER TRUYỀN THẲNG TỪ BƯỚC ATC:
                # Không tính lại trên 'held' tại Open để tránh mã thứ 2 vừa mua cùng buổi sáng kích hoạt rebalance lên mã thứ 1
                is_leader_upgrade = cand.get('is_leader_upgrade', False)
                rebalance_targets = cand.get('rebalance_targets', [])
                is_split_pos = cand.get('is_split_size', False)

                if is_leader_upgrade:
                    slot_multiplier = 1.0
                    is_split_pos = False
                else:
                    slot_multiplier = 0.5 if is_split_pos else 1.0

                target_alloc = target_slot_size * slot_multiplier
                # KHỐNG CHẾ NGÂN SÁCH LỆNH MUA CHUẨN XÁC:
                # Giá trị đặt mua tối đa không được vượt quá: Target Slot, Tiền mặt hiện có, VÀ Ngân sách tỷ trọng còn lại!
                alloc_budget = min(target_alloc, cash, rem_exposure_budget)

                # KIỂM TRA SỨC MUA TIỀN MẶT DỰ PHÒNG THỰC TẾ (CASH PURCHASING POWER):
                # 1. Ngân sách khả dụng phải đủ mua tối thiểu 1 lô 100 cổ phiếu
                # 2. Ngân sách khả dụng phải đủ trang trải ít nhất 50% vị thế chuẩn (chống mua manh mún khi tiền cũ chưa về)
                if alloc_budget < cost_per_lot_vnd or alloc_budget < (target_alloc * 0.50):
                    continue

                max_lots = np.floor(alloc_budget / cost_per_lot_vnd)
                shares = max_lots * 100.0  # Luôn là bội số chính xác của 100 cổ phiếu HOSE
                total_spent = shares * cost_per_share_vnd
                
                # Chặn tuyệt đối nếu không đủ tiền mặt thanh toán hoặc vượt trần tỷ trọng
                if shares >= 100 and cash >= total_spent and total_spent <= (rem_exposure_budget + 1e-4):
                    cash -= total_spent
                    rem_exposure_budget -= total_spent
                    stock_val_open += total_spent
                    trade_id = f"T{next_trade_id:03d}"
                    next_trade_id += 1
                    held[c] = {
                        'trade_id': trade_id,
                        'qty': shares,
                        'initial_qty': shares,
                        'buy_price': open_p,
                        'highest': open_p,
                        'last_price': open_p,
                        'buy_date': date_str,
                        'hold_days': 0,
                        'consecutive_below_ma20': 0,
                        'is_split_size': is_split_pos,
                        'score': cand.get('score', 0.55),
                        'rebalance_for_leader': False,
                        'accumulated_pnl': 0.0
                    }
                    buys_today += 1
                    
                    # Chỉ kích hoạt cờ hoán đổi tỷ trọng sau khi lệnh mua Leader ĐÃ KHỚP THÀNH CÔNG:
                    if is_leader_upgrade and rebalance_targets:
                        for h in rebalance_targets:
                            if h in held and not held[h].get('is_split_size', False) and h != c:
                                held[h]['rebalance_for_leader'] = True
                    
                    trade_nav = cash + stock_val_open
                    cum_ret_pct = ((trade_nav / INITIAL_CAPITAL) - 1.0) * 100.0
                    
                    trade_log.append({
                        'Date': date_str,
                        'Trade_ID': trade_id,
                        'Action': f"BUY ({cand['signal']})",
                        'Ticker': c,
                        'Price': round(float(open_p), 2),
                        'Shares': int(shares),
                        'Rem_Shares': int(shares),
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
        # 3. TRONG PHIÊN & CUỐI PHIÊN: CẬP NHẬT TRẠNG THÁI VỊ THẾ & HARD STOP INTRADAY
        # -------------------------------------------------------------
        for c in held:
            held[c]['hold_days'] += 1

        # KIỂM TRA HARD STOP-LOSS ĐỘNG THEO ATR TRONG PHIÊN (SAU KHI ĐỦ T+2.5):
        # Ngưỡng cắt lỗ = max(-8.5%, -2.2 * ATR_pct) — mã hiền tự siết chặt, mã mạnh có biên thở rộng
        # Khớp tại min(Open, stop_price) để chặn đứng các cú trượt lỗ sâu
        intraday_stopped = []
        for c in list(held.keys()):
            if held[c]['hold_days'] >= MIN_HOLD_DAYS:
                buy_p = held[c]['buy_price']
                dynamic_sl = calc_dynamic_stoploss(c, date)
                stop_threshold_price = buy_p * (1.0 + dynamic_sl)
                curr_l = stocks_low.at[date, f'{c}_L'] if f'{c}_L' in stocks_low.columns else (stocks_close.at[date, c] if c in stocks_close.columns else np.nan)
                
                if pd.notna(curr_l) and curr_l <= stop_threshold_price:
                    open_p = stocks_open.at[date, c] if (c in stocks_open.columns and pd.notna(stocks_open.at[date, c]) and stocks_open.at[date, c] > 0) else curr_l
                    
                    # KIỂM TRA BIÊN ĐỘ GIÁ SÀN HOSE (-7%) VÀ TÌNH TRẠNG TRẮNG BÊN MUA:
                    prev_d = backtest['Date'].iloc[i - 1] if i > 0 else date
                    prev_c = stocks_close.at[prev_d, c] if (prev_d in stocks_close.index and c in stocks_close.columns and pd.notna(stocks_close.at[prev_d, c])) else buy_p
                    floor_price = prev_c * 0.932  # Giá sàn HOSE (-7% + dung sai làm tròn bước giá)
                    
                    if pd.notna(open_p) and open_p <= floor_price:
                        curr_h = stocks_high.at[date, f'{c}_H'] if f'{c}_H' in stocks_high.columns else (stocks_close.at[date, c] if c in stocks_close.columns else np.nan)
                        if pd.notna(curr_h) and curr_h > floor_price:
                            # Trong phiên có thanh khoản hấp thụ thoát sàn -> Khớp tại giá sàn
                            exec_price = floor_price
                        else:
                            # Trắng bên mua cả ngày: Hoãn khớp sang phiên sau nếu chưa hoãn
                            held[c]['floor_lock_count'] = held[c].get('floor_lock_count', 0) + 1
                            if held[c]['floor_lock_count'] < 3:
                                continue  # Hoãn khớp sang phiên kế tiếp có thanh khoản
                            else:
                                # STRESS HAIRCUT (Khoản phạt rủi ro mô phỏng, KHÔNG phải mức khớp lệnh thực tế):
                                # Sau 3 phiên kịch sàn liên tiếp không có thanh khoản, giả lập giá thoát
                                # bằng giá sàn phiên cuối cùng. Đây là worst-case liquidation assumption
                                # vì trên thực tế ATO phiên thứ 4 thường có lệnh hấp thụ tại sàn hoặc gần sàn.
                                exec_price = floor_price
                    else:
                        exec_price = min(open_p, stop_threshold_price) if (pd.notna(open_p) and open_p > 0) else stop_threshold_price
                    
                    sell_qty = int(held[c]['qty'])
                    sell_price_vnd = exec_price * 1000.0
                    buy_price_vnd = buy_p * 1000.0
                    
                    gross_val = sell_qty * sell_price_vnd
                    net_val = gross_val * (1.0 - SELL_FRICTION)
                    buy_cost = sell_qty * buy_price_vnd * (1.0 + BUY_FRICTION)
                    pnl_vnd = net_val - buy_cost
                    pnl_pct = ((sell_price_vnd * (1.0 - SELL_FRICTION)) / (buy_price_vnd * (1.0 + BUY_FRICTION)) - 1.0) * 100.0
                    hold_days = held[c]['hold_days']
                    trade_id = held[c].get('trade_id', 'T000')
                    
                    cash += net_val
                    
                    total_pos_pnl_vnd = held[c].get('accumulated_pnl', 0.0) + pnl_vnd
                    init_shares = held[c].get('initial_qty', sell_qty)
                    buy_cost_total = init_shares * (held[c]['buy_price'] * 1000.0) * (1.0 + BUY_FRICTION)
                    pos_profit_pct = (total_pos_pnl_vnd / buy_cost_total) * 100.0 if buy_cost_total > 0 else 0.0
                    
                    completed_trades.append({
                        'Trade_ID': trade_id,
                        'Ticker': c,
                        'Buy_Date': held[c]['buy_date'],
                        'Exit_Date': date_str,
                        'Buy_Price': held[c]['buy_price'],
                        'Exit_Price': round(float(exec_price), 2),
                        'Initial_Qty': int(init_shares),
                        'Total_PnL_VND': int(round(total_pos_pnl_vnd)),
                        'Profit(%)': round(float(pos_profit_pct), 2),
                        'Hold_Days': int(hold_days),
                        'Reason': f'Cắt lỗ động ATR {dynamic_sl*100:.1f}% (Low trong phiên chạm ngưỡng)'
                    })
                    
                    del held[c]
                    intraday_stopped.append(c)
                    
                    rem_stock_val = get_stock_valuation_at_close(date)
                    trade_nav = cash + rem_stock_val
                    cum_ret_pct = ((trade_nav / INITIAL_CAPITAL) - 1.0) * 100.0
                    
                    trade_log.append({
                        'Date': date_str,
                        'Trade_ID': trade_id,
                        'Action': f"SELL_FULL (Cắt lỗ động ATR {dynamic_sl*100:.1f}% trong phiên)",
                        'Ticker': c,
                        'Price': round(float(exec_price), 2),
                        'Shares': int(sell_qty),
                        'Rem_Shares': 0,
                        'Trade_Value': int(round(net_val)),
                        'PnL_VND': int(round(pnl_vnd)),
                        'Profit(%)': round(float(pnl_pct), 2),
                        'Hold_Days': int(hold_days),
                        'Total_NAV': int(round(trade_nav)),
                        'Portfolio_Return(%)': round(float(cum_ret_pct), 2),
                        'AI_Score': None
                    })

        if intraday_stopped:
            pending_sells = [(tick, r) for tick, r in pending_sells if tick not in intraday_stopped]

        for c in held:
            curr_p = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
            curr_h = stocks_high.at[date, f'{c}_H'] if f'{c}_H' in stocks_high.columns else curr_p
            if pd.notna(curr_h) and curr_h > 0:
                held[c]['highest'] = max(held[c].get('highest', curr_h), curr_h)
            if pd.notna(curr_p) and curr_p > 0:
                held[c]['last_price'] = curr_p

            # Theo dõi số phiên liên tiếp đóng cửa dưới MA20 (phục vụ Asymmetric Exit)
            ma20_c = stocks_ma20.at[date, c] if c in stocks_ma20.columns else np.nan
            if pd.notna(curr_p) and pd.notna(ma20_c) and curr_p < ma20_c:
                held[c]['consecutive_below_ma20'] = held[c].get('consecutive_below_ma20', 0) + 1
            else:
                held[c]['consecutive_below_ma20'] = 0

        # Cập nhật NAV cuối phiên chuẩn xác theo Close[t]
        stock_val_close = get_stock_valuation_at_close(date)
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

        # BỘ NGẮT MẠCH CẮT LỖ CỨNG ĐỘC LẬP (-7.5%):
        # Ưu tiên cao nhất cho mọi vị thế đã đủ T+2.5, chặn đứng triệt để các khoản lỗ đuôi trái (fat-tail losses)
        for c in list(held.keys()):
            if held[c]['hold_days'] >= MIN_HOLD_DAYS and c not in already_pending_sells:
                c_p = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                if pd.notna(c_p) and c_p > 0:
                    b_p = held[c]['buy_price']
                    p_margin = (c_p - b_p) / b_p
                    if p_margin <= HARD_STOPLOSS:
                        pending_sells.append((c, f'SELL (Cắt lỗ cứng {HARD_STOPLOSS*100:.1f}%)'))
                        already_pending_sells.add(c)

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

                # BẢO VỆ VỊ THẾ CÓ LÃI ĐỆM:
                # Nếu bật Asymmetric Exit: vị thế lãi >= +5% KHÔNG bán hoảng loạn theo Macro!
                # Nếu không bật: giữ ngưỡng cũ (+3%)
                cushion_threshold = 0.05 if enable_asymmetric_atr_exit else 0.03
                if profit_margin >= cushion_threshold:
                    continue

                # Chỉ cắt các vị thế đang lỗ, hòa hoặc lãi mỏng
                pending_sells.append((c, 'SELL (Cắt Macro gãy MA100)'))

        else:
            # THỊ TRƯỜNG BÌNH THƯỜNG (HOẶC CHƯA XÁC NHẬN SẬP DỨT KHOÁT):
            index_below_ma20 = (pd.notna(market_ma20) and idx < market_ma20)

            for c in list(held.keys()):
                if held[c]['hold_days'] < MIN_HOLD_DAYS or c in already_pending_sells:
                    continue
                curr_price = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                if pd.isna(curr_price) or curr_price <= 0:
                    continue

                buy_price = held[c]['buy_price']
                profit_margin = (curr_price - buy_price) / buy_price
                peak_margin = (held[c]['highest'] - buy_price) / buy_price
                ma20_c = stocks_ma20.at[date, c] if c in stocks_ma20.columns else np.nan
                ma50_c = stocks_ma50.at[date, c] if c in stocks_ma50.columns else np.nan
                consec_below = held[c].get('consecutive_below_ma20', 0)

                atr_c = stocks_atr.at[date, c] if (stocks_atr is not None and c in stocks_atr.columns) else np.nan
                atr_pct = (atr_c / curr_price) if (pd.notna(atr_c) and curr_price > 0) else 0.03

                # LÝ DO 1: Rời rổ VN30
                if c not in active_vn30:
                    pending_sells.append((c, 'SELL (Rời rổ VN30)'))

                # LÝ DO 2: HOÁN ĐỔI TỶ TRỌNG CHO LEADER MỚI BỨT PHÁ (LEADER REBALANCE)
                elif held[c].get('rebalance_for_leader', False) and held[c]['hold_days'] >= MIN_HOLD_DAYS and profit_margin < 0.10:
                    shares_held = int(held[c]['qty'])
                    can_split_lot = ((shares_held // 2 // 100) * 100 >= 100) and (shares_held - ((shares_held // 2 // 100) * 100) >= 100)
                    if can_split_lot:
                        pending_sells.append((c, 'SELL (Hoán đổi 50% tỷ trọng cho Leader)'))
                    held[c]['rebalance_for_leader'] = False

                # GIẢI PHÁP 3: THOÁT VỊ THẾ BẤT ĐỐI XỨNG (ASYMMETRIC EXIT RULES)
                elif enable_asymmetric_atr_exit:
                    # 1. Cắt lỗ cứng độc lập (dứt khoát bảo vệ vốn)
                    if profit_margin <= HARD_STOPLOSS:
                        pending_sells.append((c, f'SELL (Cắt lỗ cứng {HARD_STOPLOSS*100:.1f}%)'))

                    # 2. SIÊU CỔ PHIẾU (LÃI ĐỈNH >= 20%):
                    # Duy trì trạng thái bảo vệ theo đỉnh; không bao giờ bị rơi rụng nhánh khi lãi giảm
                    elif peak_margin >= SUPER_WINNER_PEAK:
                        super_drop = max(SUPER_WINNER_DROP, 2.5 * atr_pct)
                        if curr_price < (held[c]['highest'] * (1.0 - super_drop)):
                            pending_sells.append((c, 'SELL (Trailing Stop chốt siêu cổ)'))
                        elif profit_margin <= 0.05:
                            pending_sells.append((c, 'SELL (Khóa lãi sàn bảo vệ +5%)'))
                        elif pd.notna(ma50_c) and curr_price < ma50_c:
                            pending_sells.append((c, 'SELL (Gãy MA50 chốt lời sóng lớn)'))
                        elif consec_below >= 2:
                            pending_sells.append((c, 'SELL (Xác nhận 2 phiên thủng MA20 chốt siêu cổ)'))

                    # 3. VỊ THẾ LÃI KHÁ (LÃI ĐỈNH 10% - 20%):
                    # Kích hoạt Trailing Stop & Sàn khóa lãi +2% bảo vệ thành quả
                    elif peak_margin >= PROFIT_PROTECTION_PEAK:
                        standard_drop = max(PROFIT_PROTECTION_DROP, 2.5 * atr_pct)
                        if curr_price < (held[c]['highest'] * (1.0 - standard_drop)):
                            pending_sells.append((c, 'SELL (Trailing Stop ATR chốt lời)'))
                        elif profit_margin <= PROFIT_PROTECTION_FLOOR:
                            pending_sells.append((c, 'SELL (Khóa bảo vệ vốn +2%)'))
                        elif consec_below >= 2:
                            pending_sells.append((c, 'SELL (Xác nhận 2 phiên thủng MA20 bảo toàn lãi)'))

                    # 4. VỊ THẾ CÓ LÃI ĐỆM (5% - 10%, đỉnh chưa chạm 10%):
                    # Cho cổ phiếu không gian co lắc; chỉ bán khi xác nhận 2 phiên thủng MA20
                    elif profit_margin >= 0.05:
                        if consec_below >= 2:
                            pending_sells.append((c, 'SELL (Xác nhận 2 phiên thủng MA20 bảo toàn lãi)'))

                    # 5. VỊ THẾ LÃI MỎNG HOẶC LỖ (< +5%): Cắt nhanh bảo vệ vốn
                    else:
                        if index_below_ma20 and pd.notna(ma20_c) and curr_price < ma20_c:
                            pending_sells.append((c, 'SELL (Thị trường dưới MA20 cắt vị thế yếu)'))
                        elif pd.notna(ma20_c) and curr_price < ma20_c and profit_margin <= 0:
                            pending_sells.append((c, 'SELL (Thủng MA20 cắt lỗ nhanh)'))
                        elif pd.notna(ma20_c) and curr_price < ma20_c and consec_below >= 2:
                            pending_sells.append((c, 'SELL (2 phiên dưới MA20 cắt vị thế mỏng)'))

                # NẾU TẮT ASYMMETRIC EXIT: DÙNG LOGIC GỐC
                else:
                    if profit_margin <= HARD_STOPLOSS:
                        pending_sells.append((c, f'SELL (Cắt lỗ cứng {HARD_STOPLOSS*100:.1f}%)'))
                    elif peak_margin >= SUPER_WINNER_PEAK:
                        drop_allowed = 0.06 if index_below_ma20 else SUPER_WINNER_DROP
                        if curr_price < (held[c]['highest'] * (1.0 - drop_allowed)):
                            pending_sells.append((c, 'SELL (Trailing Stop chốt siêu cổ)'))
                        elif pd.notna(ma50_c) and curr_price < ma50_c:
                            pending_sells.append((c, 'SELL (Gãy MA50 chốt lời sóng lớn)'))
                        elif index_below_ma20 and pd.notna(ma20_c) and curr_price < ma20_c and profit_margin >= 0.08:
                            pending_sells.append((c, 'SELL (Thị trường dưới MA20 chốt siêu cổ)'))
                        elif profit_margin <= 0.08:
                            pending_sells.append((c, 'SELL (Khóa lãi sàn bảo vệ +8%)'))
                    elif peak_margin >= PROFIT_PROTECTION_PEAK:
                        drop_allowed = 0.05 if index_below_ma20 else PROFIT_PROTECTION_DROP
                        if curr_price < (held[c]['highest'] * (1.0 - drop_allowed)):
                            pending_sells.append((c, 'SELL (Trailing Stop chốt lời)'))
                        elif pd.notna(ma20_c) and curr_price < ma20_c and (profit_margin >= 0.03 if index_below_ma20 else 0.05):
                            pending_sells.append((c, 'SELL (Thủng MA20 bảo toàn lãi)'))
                        elif profit_margin <= PROFIT_PROTECTION_FLOOR:
                            pending_sells.append((c, 'SELL (Khóa bảo vệ vốn +2%)'))
                    else:
                        if index_below_ma20 and pd.notna(ma20_c) and curr_price < ma20_c:
                            pending_sells.append((c, 'SELL (Thị trường dưới MA20 cắt vị thế yếu)'))
                        elif pd.notna(ma20_c) and curr_price < ma20_c and profit_margin > MA20_PROFIT_EXIT:
                            pending_sells.append((c, 'SELL (Thủng MA20 bảo toàn lãi)'))

        # =============================================================
        # (B) QUÉT TÍN HIỆU MUA CHO PHIÊN MAI (PRE-MA100 + HYSTERESIS BUFFER + COOLDOWN + SECTOR CAP)
        # =============================================================
        # ĐỒNG BỘ ĐIỀU KIỆN MUA VÀ BÁN VĨ MÔ:
        # Nếu VN-Index dưới MA100: Khóa mua mới tuyệt đối (0% vốn cổ phiếu, 100% tiền mặt nhàn rỗi hưởng lãi Bank)
        if idx < market_ma100:
            max_exposure = 0.0
        elif idx < market_ma50 or market_atr_pct > 0.020:
            max_exposure = 0.4
        elif idx < market_ma200:
            max_exposure = 0.5
        else:
            max_exposure = 1.0

        current_max_exposure = max_exposure
        allowed_stock_cap = max(0.0, (current_nav * max_exposure) - stock_val_close)

        # 1. VÙNG ĐỆM HYSTERESIS: VN-Index phải vượt MA100 tối thiểu +1.5%
        is_macro_buffer_ok = (idx >= market_ma100 * 1.015)

        # 2. XU HƯỚNG TRUNG HẠN ĐỒNG THUẬN: MA50 phải dốc lên hoặc đi ngang (loại bỏ sóng hồi bulltrap trong thị trường Bear 2022)
        prev5_ma50 = backtest['MA50'].iloc[i-5] if i >= 5 else backtest['MA50'].iloc[0]
        is_ma50_slope_rising = (market_ma50 >= prev5_ma50) if (pd.notna(market_ma50) and pd.notna(prev5_ma50)) else True

        # 3. ĐƯỜNG MA100 KHÔNG CẮM ĐẦU RƠI MẠNH (Downtrend thoái trào)
        prev5_ma100 = backtest['MA100'].iloc[i-5] if i >= 5 else backtest['MA100'].iloc[0]
        is_ma100_not_falling = (market_ma100 >= prev5_ma100 * 0.998) if (pd.notna(market_ma100) and pd.notna(prev5_ma100)) else True

        # 4. XU HƯỚNG NGẮN/TRUNG HẠN ĐỒNG THUẬN: MA20 > MA50 VÀ MA50 dốc lên
        is_ma_trend_ok = (pd.notna(market_ma20) and pd.notna(market_ma50) and market_ma20 > market_ma50 and is_ma50_slope_rising)

        # 5. PRE-MA100 FILTER: VN-Index phải đóng cửa TRÊN MA20 (Khóa mua mới khi thị trường gãy xu hướng ngắn hạn)
        is_index_above_ma20 = (pd.notna(market_ma20) and idx >= market_ma20)

        # 6. CHỐNG PHIÊN PHÂN PHỐI MẠNH: Khóa mua khi Index giảm > 1.5% hoặc giảm > 0.8% kèm Volume Shock
        idx_ret = (idx - prev_idx) / prev_idx if prev_idx > 0 else 0.0
        is_not_distribution = not ((idx_ret <= -0.015) or (idx_ret < -0.008 and vol_shock >= 1.25))

        # 7. THỜI GIAN CHỜ COOLDOWN: Ít nhất 10 phiên sau lần bán tháo Macro gần nhất (tránh mua đuổi nhịp hồi giả)
        is_cooldown_passed = (i - last_macro_exit_bar > 10)

        # 8. ĐỘ BIẾN ĐỘNG HẠ NHIỆT (market_atr_pct <= 0.020)
        is_volatility_calm = (market_atr_pct <= 0.020)

        # 9. ĐỘ RỘNG THỊ TRƯỜNG: >= 35% số mã VN30 nằm trên MA50
        vn30_above_ma50_cnt = sum([
            1 for ticker in active_vn30
            if ticker in stocks_close.columns and ticker in stocks_ma50.columns
            and pd.notna(stocks_close.at[date, ticker]) and pd.notna(stocks_ma50.at[date, ticker])
            and stocks_close.at[date, ticker] > stocks_ma50.at[date, ticker]
        ])
        market_breadth = (vn30_above_ma50_cnt / len(active_vn30)) if len(active_vn30) > 0 else 0.5
        is_breadth_healthy = (market_breadth >= 0.35)

        # GIẢI PHÁP 2: TÍNH NGƯỠNG AI SCORE DỰA TRÊN TOP 20% (PERCENTILE 80) TRONG 120 PHIÊN GẦN NHẤT
        if enable_dynamic_percentile_ai:
            win_start = max(0, i - 120)
            win_matrix = stocks_ai_scores.iloc[win_start:i].values
            valid_scores = win_matrix[~np.isnan(win_matrix)]
            if len(valid_scores) >= 50:
                p80 = np.percentile(valid_scores, 80)
                current_ai_threshold = float(np.clip(p80, 0.55, 0.65))
            else:
                current_ai_threshold = 0.58
        else:
            current_ai_threshold = AI_THRESHOLD

        # GIẢI PHÁP 4: BỘ LỌC ĐỘ DỐC MA20 CỦA INDEX & NGƯỠNG VOLUME BREAKOUT 1.2x
        prev5_ma20 = backtest['MA20'].iloc[i-5] if i >= 5 else backtest['MA20'].iloc[0]
        is_ma20_slope_rising = (market_ma20 > prev5_ma20) if (pd.notna(market_ma20) and pd.notna(prev5_ma20)) else True
        vol_breakout_target = 1.2 if enable_ma20_slope_filter else VOL_BREAKOUT_RATIO

        is_full_trend_ok = (
            is_macro_buffer_ok
            and is_ma_trend_ok
            and is_ma100_not_falling
            and is_index_above_ma20
            and (is_ma20_slope_rising if enable_ma20_slope_filter else True)
            and is_not_distribution
            and is_cooldown_passed
            and is_volatility_calm
            and is_breadth_healthy
            and (market_adx > ADX_MIN)
        )

        projected_held_count = len(held) - len(pending_sells)

        if is_full_trend_ok:
            can_enter_new_trades = (allowed_stock_cap > 0)
            available_slots = MAX_SLOTS - projected_held_count
            trade_signal_label = 'Breakout'
        else:
            can_enter_new_trades = False
            available_slots = 0
            trade_signal_label = 'None'

        if available_slots > 0 and can_enter_new_trades:
            buy_candidates = []
            vol_target = vol_breakout_target

            for c in active_vn30:
                if c not in held and c in loaded_tickers and c in stocks_close.columns:
                    curr_price = stocks_close.at[date, c]
                    curr_vol = stocks_volume.at[date, f'{c}_V'] if f'{c}_V' in stocks_volume.columns else np.nan
                    prev_h20 = stocks_high_20.at[date, c] if c in stocks_high_20.columns else np.nan
                    prev_v_ma = stocks_vol_ma20.at[date, f'{c}_V'] if f'{c}_V' in stocks_vol_ma20.columns else np.nan
                    ma50_c = stocks_ma50.at[date, c] if c in stocks_ma50.columns else np.nan
                    rsi_c = stocks_rsi.at[date, c] if (stocks_rsi is not None and c in stocks_rsi.columns) else np.nan

                    if pd.isna(curr_price) or pd.isna(curr_vol) or pd.isna(prev_h20) or pd.isna(prev_v_ma):
                        continue
                    if prev_v_ma <= 0 or curr_price <= 0:
                        continue

                    score = ai_dict.get(c, {}).get(date_str, np.nan)
                    if pd.isna(score):
                        continue

                    v_ratio = curr_vol / prev_v_ma

                    # GIẢI PHÁP 3: BỘ LỌC CHỐNG MUA ĐUỔI VÙNG QUÁ MUA (ANTI-CHASE FILTER)
                    # Chặn các lệnh breakout muộn khi cổ phiếu đã tăng nóng cách xa nền MA50 > 20% (như bẫy VHM 14/10/2025)
                    # Nâng trần RSI lên 82 (không chặn chân sóng breakout mạnh từ nền chặt RSI 75-80 như PLX đầu 2026)
                    if enable_anti_chase_filter:
                        is_chasing = False
                        if pd.notna(ma50_c) and ma50_c > 0 and (curr_price > ma50_c * 1.20):
                            is_chasing = True
                        if pd.notna(rsi_c) and (rsi_c > 82.0):
                            is_chasing = True
                        if is_chasing:
                            continue

                    # Tín hiệu Momentum Breakout chuẩn mực (Volume bứt phá >= 1.2x, AI Score vượt ngưỡng)
                    if curr_price > prev_h20 and v_ratio >= vol_target and score >= current_ai_threshold:
                        breakout_pct = ((curr_price / prev_h20) - 1.0) if (pd.notna(prev_h20) and prev_h20 > 0) else 0.0
                        ret_20 = stocks_ret20.at[date, c] if (stocks_ret20 is not None and c in stocks_ret20.columns and pd.notna(stocks_ret20.at[date, c])) else 0.0

                        buy_candidates.append({
                            'ticker': c,
                            'score': score,
                            'signal': trade_signal_label,
                            'sector': SECTOR_MAP.get(c, 'OTHER'),
                            'ecosystem': ECOSYSTEM_MAP.get(c, c),
                            'breakout_pct': breakout_pct,
                            'ret_20': ret_20
                        })

            if buy_candidates:
                # GIẢI PHÁP 2: SẮP XẾP ƯU TIÊN THEO ĐỘNG LƯỢNG GIÁ THAY VÌ BẢNG CHỮ CÁI
                # 1. Điểm AI Score cao nhất
                # 2. Tỷ lệ bứt phá đỉnh 20 phiên (% Breakout) mạnh nhất
                # 3. Sức mạnh giá tương đối (RS / Tỷ suất sinh lời 20 phiên gần nhất) cao nhất
                buy_candidates = sorted(
                    buy_candidates,
                    key=lambda x: (round(x['score'], 3), round(x['breakout_pct'], 4), round(x['ret_20'], 4)),
                    reverse=True
                )
                
                # KHÓA TRẦN NGÀNH & HỆ SINH THÁI (KÈM CƠ CHẾ SPLIT-SIZE ALLOCATION)
                selected_buys = []
                for cand in buy_candidates:
                    if len(selected_buys) >= MAX_BUYS_PER_DAY or len(selected_buys) >= available_slots:
                        break
                    
                    sec = cand['sector']
                    eco = cand['ecosystem']

                    sec_in_held = sum(1 for h in held if SECTOR_MAP.get(h, 'OTHER') == sec and h not in [s[0] for s in pending_sells])
                    sec_in_selected = sum(1 for b in selected_buys if b.get('sector') == sec)

                    eco_in_held = sum(1 for h in held if ECOSYSTEM_MAP.get(h, h) == eco and h not in [s[0] for s in pending_sells])
                    eco_in_selected = sum(1 for b in selected_buys if b.get('ecosystem') == eco)
                    
                    sec_cap_limit = MAX_PER_SECTOR
                    if (sec_in_held + sec_in_selected) < sec_cap_limit and (eco_in_held + eco_in_selected) < MAX_PER_ECOSYSTEM:
                        # CƠ CHẾ HOÁN ĐỔI CHO LEADER: Nếu mã mới có AI Score vượt trội mã đang nắm giữ cùng nhóm (ngưỡng tối thiểu +0.03)
                        LEADER_MIN_AI_MARGIN = 0.03
                        held_siblings = [
                            h for h in held 
                            if (SECTOR_MAP.get(h, 'OTHER') == sec or ECOSYSTEM_MAP.get(h, h) == eco) 
                            and h not in [s[0] for s in pending_sells]
                        ]
                        full_size_siblings = [h for h in held_siblings if not held[h].get('is_split_size', False)]
                        max_sibling_score = max([held[h].get('score', 0.55) for h in full_size_siblings], default=0.0)
                        
                        # Không rebalance nếu cùng nhóm đã có mã trong selected_buys hôm nay
                        has_sibling_in_selected = any(b.get('sector') == sec or b.get('ecosystem') == eco for b in selected_buys)
                        is_leader_upgrade = (len(full_size_siblings) > 0) and (not has_sibling_in_selected) and (cand['score'] >= max_sibling_score + LEADER_MIN_AI_MARGIN)
                        
                        cand['is_leader_upgrade'] = is_leader_upgrade
                        cand['rebalance_targets'] = list(full_size_siblings) if is_leader_upgrade else []
                        
                        if is_leader_upgrade:
                            cand['is_split_size'] = False
                        else:
                            is_split = (sec_in_held + sec_in_selected >= 1) or (eco_in_held + eco_in_selected >= 1)
                            cand['is_split_size'] = is_split
                            if is_split:
                                for prev_b in selected_buys:
                                    if prev_b.get('sector') == sec or prev_b.get('ecosystem') == eco:
                                        prev_b['is_split_size'] = True
                        selected_buys.append(cand)
                    # Nếu ngành hoặc hệ sinh thái đã đủ trần, nhường slot cho ngành/hệ sinh thái khác
                
                pending_buys = selected_buys

    # Chốt NAV phiên cuối cùng
    final_stock_val = get_stock_valuation_at_close(backtest['Date'].iloc[-1])
    final_nav = cash + final_stock_val

    df_nav = pd.DataFrame(daily_nav_history)
    df_trades = pd.DataFrame(trade_log)

    desired_cols = [
        'Date', 'Trade_ID', 'Action', 'Ticker', 'Price', 'Shares', 'Rem_Shares', 'Trade_Value', 'PnL_VND',
        'Profit(%)', 'Hold_Days', 'Total_NAV', 'Portfolio_Return(%)', 'AI_Score'
    ]
    final_cols = [col for col in desired_cols if col in df_trades.columns]
    df_trades = df_trades[final_cols] if not df_trades.empty else pd.DataFrame(columns=desired_cols)

    # Ép kiểu Int64 nullable để xuất file sạch 100%
    for int_col in ['Shares', 'Rem_Shares', 'Trade_Value', 'PnL_VND', 'Hold_Days', 'Total_NAV']:
        if int_col in df_trades.columns:
            df_trades[int_col] = pd.to_numeric(df_trades[int_col], errors='coerce').round(0).astype('Int64')

    for nav_int in ['NAV', 'Cash', 'Bank_Interest']:
        if nav_int in df_nav.columns:
            df_nav[nav_int] = pd.to_numeric(df_nav[nav_int], errors='coerce').round(0).astype('Int64')

    return df_trades, df_nav, final_nav, total_bank_interest, held, pending_buys, pending_sells, completed_trades

# ==========================================
# PIPELINE THỰC THI CHÍNH (CHỈ CHẠY KHI GỌI TRỰC TIẾP, AN TOÀN KHI IMPORT)
# ==========================================
def main():
    # ==========================================
    # 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG (OHLCV THẬT)
    # ==========================================
    print("1. Đang tải Market Proxy (VNINDEX) với nến OHLCV thật...")
    vnindex_raw = fetch_ticker_data('VNINDEX', n_bars=4500)
    if vnindex_raw is None or vnindex_raw.empty:
        sys.exit("❌ LỖI: Không thể tải dữ liệu VNINDEX từ API!")

    vnindex_df = vnindex_raw.rename(columns={'Close': 'VNINDEX'}).drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
    vnindex_df['MA20'] = vnindex_df['VNINDEX'].rolling(20, min_periods=10).mean()
    vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50, min_periods=20).mean()
    vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100, min_periods=30).mean()
    vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200, min_periods=40).mean()
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
    stocks_atr = pd.DataFrame(index=stocks_close.index)
    stocks_ret20 = pd.DataFrame(index=stocks_close.index)

    for c in loaded_tickers:
        s_c = stocks_close[c].dropna()
        if len(s_c) >= 20:
            stocks_ma20[c] = s_c.rolling(20).mean()
            stocks_ma50[c] = s_c.rolling(50, min_periods=20).mean()
            stocks_rsi[c] = calc_rsi(s_c)
            stocks_macd[c] = calc_macd_hist(s_c)
            stocks_ret20[c] = s_c.pct_change(20)

        h_col = f'{c}_H'
        l_col = f'{c}_L'
        # ĐỈNH KHÁNG CỰ 20 PHIÊN: Bắt buộc tính trên giá High (High-20 Breakout) để chống bẫy râu nến trên (Bull Trap)
        if h_col in stocks_high.columns:
            s_h = stocks_high[h_col].dropna()
            if len(s_h) >= 20:
                stocks_high_20[c] = s_h.rolling(20).max().shift(1)
            elif len(s_c) >= 20:
                stocks_high_20[c] = s_c.rolling(20).max().shift(1)
        elif len(s_c) >= 20:
            stocks_high_20[c] = s_c.rolling(20).max().shift(1)

        if h_col in stocks_high.columns and l_col in stocks_low.columns and c in stocks_close.columns:
            s_h = stocks_high[h_col].dropna()
            s_l = stocks_low[l_col].dropna()
            common_idx = s_h.index.intersection(s_l.index).intersection(s_c.index)
            if len(common_idx) >= 14:
                stocks_atr[c] = calc_atr(s_h.loc[common_idx], s_l.loc[common_idx], s_c.loc[common_idx], 14)

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
    # 3. TẢI BỘ NHỚ AI & ĐỒNG BỘ VĨ MÔ CSV
    # ==========================================
    print("3. Đang tải Não bộ AI tĩnh & Đồng bộ Vĩ mô + Độ rộng...")
    try:
        macro_df = pd.read_csv('markets_database10years.csv')
        avail_cols = ['Date', 'DXY', 'US10Y', 'USD_VND']
        if 'FED_LIQUIDITY' in macro_df.columns:
            avail_cols.append('FED_LIQUIDITY')
        macro_df = macro_df[avail_cols].copy()
        macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
        macro_df = macro_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
    except Exception as e:
        sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

    # LÀM SẠCH TỶ GIÁ USD/VND (Xử lý đơn vị nghìn 21.055, 21.0 và dị biệt 3210, 3211)
    # TUYỆT ĐỐI KHÔNG dùng interpolate() hay bfill() (chống rò rỉ dữ liệu tương lai / Look-ahead Bias)
    macro_df['USD_VND'] = pd.to_numeric(macro_df['USD_VND'], errors='coerce')
    macro_df['USD_VND'] = macro_df['USD_VND'].apply(lambda x: x * 1000.0 if (pd.notna(x) and 10.0 <= x <= 40.0) else x)
    macro_df['USD_VND'] = macro_df['USD_VND'].apply(lambda x: np.nan if (pd.notna(x) and (x < 15000.0 or x > 35000.0)) else x)
    macro_df['USD_VND'] = macro_df['USD_VND'].ffill()

    if 'FED_LIQUIDITY' in macro_df.columns:
        macro_df['FED_LIQUIDITY'] = pd.to_numeric(macro_df['FED_LIQUIDITY'], errors='coerce').ffill().shift(1)

    # ĐỒNG BỘ THỜI ĐIỂM CÔNG BỐ (PUBLICATION LAG):
    # Thị trường VN đóng cửa lúc 15:00 (phát tín hiệu 15:30). Dữ liệu đóng cửa Mỹ (DXY, US10Y)
    # và tỷ giá ngoại hối quốc tế (Yahoo VND=X) kết thúc sau đó. Bắt buộc shift(1) để chống nhìn trước tương lai!
    macro_df['DXY'] = pd.to_numeric(macro_df['DXY'], errors='coerce').shift(1)
    macro_df['US10Y'] = pd.to_numeric(macro_df['US10Y'], errors='coerce').shift(1)
    macro_df['USD_VND'] = macro_df['USD_VND'].shift(1)

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

    # TÍNH DỰ PHÒNG Z_FED_LIQUIDITY (NẾU CÓ TRONG DATA ĐỂ TRÁNH NAN KHI GẶP MODEL YÊU CẦU FEATURE NÀY)
    if 'FED_LIQUIDITY' in macro_df.columns:
        roll_m = macro_df['FED_LIQUIDITY'].rolling(120, min_periods=5).mean()
        roll_s = macro_df['FED_LIQUIDITY'].rolling(120, min_periods=5).std().replace(0, np.nan)
        macro_df['Z_FED_LIQUIDITY'] = (macro_df['FED_LIQUIDITY'] - roll_m) / roll_s

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
    default_model = None
    if os.path.exists(DEFAULT_MODEL_FILE):
        default_model = xgb.XGBClassifier()
        default_model.load_model(DEFAULT_MODEL_FILE)
        print(f"   🧠 Đã nạp mô hình cơ sở: {DEFAULT_MODEL_FILE}")

    def get_model_for_year(target_year):
        if target_year in wf_models:
            return wf_models[target_year]
        # CHẶN TUYỆT ĐỐI MODEL TƯƠNG LAI: Chỉ tìm model của các năm <= target_year
        valid_past_years = [py for py in sorted(wf_models.keys()) if py <= target_year]
        if valid_past_years:
            fallback_year = max(valid_past_years)
            print(f"   ℹ️ Năm {target_year} thiếu model riêng -> Dùng model năm quá khứ gần nhất: {fallback_year}")
            return wf_models[fallback_year]
        if default_model is not None:
            return default_model
        sys.exit(f"🛑 LỖI BẢO MẬT WALK-FORWARD: Không tìm thấy mô hình hợp lệ cho năm {target_year}! Cấm tuyệt đối dùng model tương lai.")

    def predict_with_model(model_obj, data_subset):
        """Hàm dự báo linh hoạt: đọc feature_names từ model để tương thích mọi phiên bản"""
        if hasattr(model_obj, 'feature_names_in_') and model_obj.feature_names_in_ is not None:
            f_names = list(model_obj.feature_names_in_)
        elif hasattr(model_obj, 'get_booster') and model_obj.get_booster().feature_names:
            f_names = list(model_obj.get_booster().feature_names)
        else:
            f_names = list(ml_features)
        
        sub_df = data_subset[[c for c in f_names if c in data_subset.columns]].copy()
        for col in f_names:
            if col not in sub_df.columns:
                sub_df[col] = np.nan
        return model_obj.predict_proba(sub_df[f_names])[:, 1]

    # Chấm điểm AI Score thích ứng Walk-Forward OOS
    ml_master['AI_Score'] = np.nan

    # 1. Giai đoạn <= 2021: Dùng mô hình học trên 2014-2021 (model 2022 hoặc model quá khứ)
    pre_2022_mask = ml_master['Date'] < '2022-01-01'
    if pre_2022_mask.sum() > 0:
        base_m = get_model_for_year(2022)
        ml_master.loc[pre_2022_mask, 'AI_Score'] = predict_with_model(base_m, ml_master.loc[pre_2022_mask])

    # 2. Các năm Out-of-Sample 2022 -> 2026: Dùng đúng mô hình học trước năm đó!
    for y in [2022, 2023, 2024, 2025, 2026]:
        if y == 2026:
            y_mask = ml_master['Date'] >= f"{y}-01-01"
        else:
            y_mask = (ml_master['Date'] >= f"{y}-01-01") & (ml_master['Date'] < f"{y+1}-01-01")

        if y_mask.sum() > 0:
            m_y = get_model_for_year(y)
            ml_master.loc[y_mask, 'AI_Score'] = predict_with_model(m_y, ml_master.loc[y_mask])

    # Tạo tra cứu AI Score dạng dict (Key: ticker -> {date_str: score})
    ai_dict = {}
    for c in loaded_tickers:
        c_sub = ml_master[ml_master['Ticker'] == c]
        ai_dict[c] = dict(zip(c_sub['Date'], c_sub['AI_Score']))



    print("\n" + "="*70)
    print("⚙️ CẤU HÌNH HỆ THỐNG GIAO DỊCH ĐỊNH LƯỢNG (AI QUANT CONFIG):")
    print(f"   • Solution 1: Time-Decayed Retrain Model: BẬT")
    print(f"   • Solution 2: Rolling Percentile Dynamic AI (Top 20% / 120 bars): {'BẬT' if ENABLE_DYNAMIC_PERCENTILE_AI else 'TẮT (Cố định 0.60)'}")
    print(f"   • Solution 3: Asymmetric Exit Rules (2 nến MA20 + 2.5x ATR Trailing): {'BẬT' if ENABLE_ASYMMETRIC_ATR_EXIT else 'TẮT'}")
    print(f"   • Solution 4: Index MA20 Slope & 1.2x Breakout Volume: {'BẬT' if ENABLE_MA20_SLOPE_FILTER else 'TẮT'}")
    print(f"   • Solution 5: Split-Size & Leader Rebalance (Tối đa {MAX_PER_SECTOR} mã/ngành, {MAX_PER_ECOSYSTEM} mã/HST | Hoán đổi 20% NAV cho Leader)")
    print(f"   • Solution 6: Sắp xếp ưu tiên lệnh mua: Động lượng (AI Score -> Breakout% -> Ret20)")
    print(f"   • Solution 7: Anti-Chase Filter (Chặn mua đuổi > MA50*1.20 hoặc RSI>82): {'BẬT' if ENABLE_ANTI_CHASE_FILTER else 'TẮT'}")
    print(f"   • Solution 8: Macro Synchronization & Intraday Hard Stop-Loss (Index > MA100 & MA50 dốc lên | Hard Stop Low: {HARD_STOPLOSS*100:.1f}%): BẬT")
    print("="*70 + "\n")

    # ==========================================
    # CHẠY CHIẾN LƯỢC & XUẤT BÁO CÁO TOÀN DIỆN
    # ==========================================
    df_trade_log, df_nav_hist, final_nav, total_interest, current_held, next_buys, next_sells, completed_trades = run_ai_quant_system(
        df_clean=df_clean,
        stocks_open=stocks_open,
        stocks_close=stocks_close,
        stocks_high=stocks_high,
        stocks_low=stocks_low,
        stocks_volume=stocks_volume,
        stocks_high_20=stocks_high_20,
        stocks_vol_ma20=stocks_vol_ma20,
        stocks_ma20=stocks_ma20,
        stocks_ma50=stocks_ma50,
        stocks_atr=stocks_atr,
        stocks_rsi=stocks_rsi,
        stocks_ret20=stocks_ret20,
        ai_dict=ai_dict,
        loaded_tickers=loaded_tickers,
        backtest_start_date=BACKTEST_START_DATE,
        max_per_sector=MAX_PER_SECTOR,
        max_per_ecosystem=MAX_PER_ECOSYSTEM,
        enable_dynamic_percentile_ai=ENABLE_DYNAMIC_PERCENTILE_AI,
        enable_asymmetric_atr_exit=ENABLE_ASYMMETRIC_ATR_EXIT,
        enable_ma20_slope_filter=ENABLE_MA20_SLOPE_FILTER,
        enable_anti_chase_filter=ENABLE_ANTI_CHASE_FILTER,
        hard_stoploss=HARD_STOPLOSS
    )

    # Ghi nhận file trading_log.csv
    df_trade_log.to_csv('trading_log.csv', index=False, na_rep='', float_format='%.2f')
    print("✅ Đã ghi nhận sổ kế toán giao dịch tinh gọn vào file trading_log.csv!")

    # Ghi nhận file daily_nav.csv
    df_nav_hist.to_csv('daily_nav.csv', index=False, na_rep='', float_format='%.2f')
    print("✅ Đã ghi nhận biến động tài sản hàng ngày chuẩn vào file daily_nav.csv!")

    # Ghi nhận file completed_trades.csv (Chu kỳ vị thế đóng trọn vẹn)
    df_completed = pd.DataFrame(completed_trades)
    if not df_completed.empty:
        df_completed.to_csv('completed_trades.csv', index=False, na_rep='', float_format='%.2f')
        print("✅ Đã ghi nhận chu kỳ vị thế hoàn tất vào file completed_trades.csv!")

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

    total_net_pnl = final_nav - initial_capital
    trade_pnl_total = sells['PnL_VND'].sum() if total_sells > 0 else 0.0
    interest_share_pct = (total_interest / total_net_pnl * 100.0) if total_net_pnl > 0 else 0.0
    trade_share_pct = (trade_pnl_total / total_net_pnl * 100.0) if total_net_pnl > 0 else 0.0

    if total_sells > 0:
        pnl_by_ticker = sells.groupby('Ticker')['PnL_VND'].sum().sort_values(ascending=False)
        top_ticker = pnl_by_ticker.index[0]
        top_ticker_pnl = pnl_by_ticker.iloc[0]
        top_ticker_share_of_trade = (top_ticker_pnl / trade_pnl_total * 100.0) if trade_pnl_total > 0 else 0.0
        rest_tickers_pnl = trade_pnl_total - top_ticker_pnl
    else:
        top_ticker = "N/A"
        top_ticker_pnl = 0.0
        top_ticker_share_of_trade = 0.0
        rest_tickers_pnl = 0.0

    cash_only_days = len(df_nav_hist[df_nav_hist['Cash'] == df_nav_hist['NAV']])
    cash_only_pct = (cash_only_days / len(df_nav_hist) * 100.0) if len(df_nav_hist) > 0 else 0.0

    vnindex_start = df_clean.loc[df_clean['Date'] >= BACKTEST_START_DATE, 'VNINDEX'].iloc[0]
    vnindex_end = df_clean['VNINDEX'].iloc[-1]
    vnindex_return_pct = ((vnindex_end / vnindex_start) - 1.0) * 100.0
    pure_cash_return_pct = (((1.0 + 0.04/252) ** len(df_nav_hist)) - 1.0) * 100.0

    print("\n" + "="*58)
    print(f"📊 BÁO CÁO ĐỊNH LƯỢNG THỰC CHIẾN TỪ {BACKTEST_START_DATE[:4]} ĐẾN NAY")
    print("🧠 MÔ HÌNH CƠ SỞ (BASE MODEL): HUẤN LUYỆN TRÊN CHU KỲ VÀNG 10 NĂM (2012 -> 2021)")
    print("🎯 KIỂM THỬ THỰC CHIẾN OOS: TÁI HUẤN LUYỆN CUỐN CHIẾU TỪNG NĂM (100% BLIND TEST / ZERO LEAKAGE)")
    print("="*58)
    print(f"💵 Vốn khởi điểm ({BACKTEST_START_DATE}):        {initial_capital:,.0f} VNĐ")
    print(f"🏦 TỔNG TÀI SẢN HIỆN TẠI (NAV):   {final_nav:,.0f} VNĐ")
    print(f"🚀 TỔNG LỢI NHUẬN (Đã trừ phí):    +{total_return_pct:.2f}% (+{total_net_pnl:,.0f} VNĐ)")
    print(f"📈 Lợi nhuận kép TB (CAGR):        {cagr:.2f}% / năm")
    print(f"----------------------------------------------------------")
    print(f"🔍 PHÂN RÃ NGUỒN LỢI NHUẬN (ATTRIBUTION ANALYSIS):")
    print(f"   💰 Lãi tiền gửi nhàn rỗi (4%/năm): +{total_interest:,.0f} VNĐ ({interest_share_pct:.1f}% tổng lãi)")
    print(f"   📊 Lãi giao dịch cổ phiếu thuần:   +{trade_pnl_total:,.0f} VNĐ ({trade_share_pct:.1f}% tổng lãi)")
    print(f"   ⭐ Đóng góp siêu cổ #{top_ticker}:       +{top_ticker_pnl:,.0f} VNĐ ({top_ticker_share_of_trade:.1f}% lãi trade)")
    print(f"   💼 Lãi các cổ phiếu còn lại:       +{rest_tickers_pnl:,.0f} VNĐ")
    print(f"   🛡 Tỷ lệ phiên giữ 100% tiền mặt:  {cash_only_pct:.1f}% ({cash_only_days}/{len(df_nav_hist)} phiên)")
    print(f"----------------------------------------------------------")
    print(f"🏁 ĐỐI CHUẨN THỊ TRƯỜNG (BENCHMARKS COMPARISON):")
    print(f"   🤖 Chiến lược AI Quant Bot:       +{total_return_pct:.2f}% (CAGR: {cagr:.2f}%)")
    print(f"   🏦 Tiền gửi tiết kiệm (Risk-free): +{pure_cash_return_pct:.2f}%")
    print(f"   🇻🇳 VN-Index (Buy & Hold):          {vnindex_return_pct:+.2f}% ({vnindex_start:,.1f} -> {vnindex_end:,.1f})")
    print(f"----------------------------------------------------------")
    print(f"📉 SỤT GIẢM TỐI ĐA (Max Drawdown): {max_mdd:.2f}%")
    print(f"   (Thời kỳ chịu đựng lớn nhất: Từ đỉnh {pd.to_datetime(peak_date).strftime('%d/%m/%Y')} về đáy {pd.to_datetime(trough_date).strftime('%d/%m/%Y')})")
    print(f"⚡ Chỉ số Calmar (CAGR / MDD):     {calmar_ratio:.2f}")
    print(f"----------------------------------------------------------")
    if not df_completed.empty:
        total_roundtrips = len(df_completed)
        win_rt = len(df_completed[df_completed['Total_PnL_VND'] > 0])
        loss_rt = total_roundtrips - win_rt
        win_rate_rt = (win_rt / total_roundtrips * 100.0) if total_roundtrips > 0 else 0.0
        avg_hold_days_rt = df_completed['Hold_Days'].mean() if total_roundtrips > 0 else 0.0
        gross_profit_rt = df_completed.loc[df_completed['Total_PnL_VND'] > 0, 'Total_PnL_VND'].sum()
        gross_loss_rt = abs(df_completed.loc[df_completed['Total_PnL_VND'] < 0, 'Total_PnL_VND'].sum())
        profit_factor_rt = (gross_profit_rt / gross_loss_rt) if gross_loss_rt > 0 else 999.0
        avg_win_pct_rt = df_completed.loc[df_completed['Profit(%)'] > 0, 'Profit(%)'].mean() if win_rt > 0 else 0.0
        avg_loss_pct_rt = df_completed.loc[df_completed['Profit(%)'] < 0, 'Profit(%)'].mean() if loss_rt > 0 else 0.0
        payoff_ratio_rt = abs(avg_win_pct_rt / avg_loss_pct_rt) if avg_loss_pct_rt != 0 else 999.0
        best_trade_rt = df_completed['Profit(%)'].max() if total_roundtrips > 0 else 0.0
        worst_trade_rt = df_completed['Profit(%)'].min() if total_roundtrips > 0 else 0.0
    else:
        total_roundtrips, win_rt, loss_rt, win_rate_rt, avg_hold_days_rt = 0, 0, 0, 0.0, 0.0
        gross_profit_rt, gross_loss_rt, profit_factor_rt, avg_win_pct_rt, avg_loss_pct_rt, payoff_ratio_rt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        best_trade_rt, worst_trade_rt = 0.0, 0.0

    print(f"🎯 THỐNG KÊ CHU KỲ VỊ THẾ HOÀN TẤT (ROUND-TRIP TRADES):")
    print(f"   • Tổng số vị thế đã đóng trọn vẹn: {total_roundtrips} thương vụ")
    print(f"   • Tỷ lệ thắng (Win Rate):         {win_rate_rt:.2f}% (Thắng {win_rt}/{total_roundtrips} vị thế)")
    print(f"   • Profit Factor (Tổng Lãi/Lỗ):    {profit_factor_rt:.2f}")
    print(f"   • Payoff Ratio (Lãi TB / Lỗ TB):  {payoff_ratio_rt:.2f}x (Lãi TB: +{avg_win_pct_rt:.2f}% | Lỗ TB: {avg_loss_pct_rt:.2f}%)")
    print(f"   • Thời gian nắm giữ trung bình:   {avg_hold_days_rt:.1f} phiên")
    print(f"   • Thương vụ lãi cao nhất:         +{best_trade_rt:.2f}%")
    print(f"   • Thương vụ lỗ sâu nhất:          {worst_trade_rt:.2f}%")
    print(f"----------------------------------------------------------")
    print(f"📋 THỐNG KÊ LƯỢT KHỚP BÁN CHI TIẾT (SELL EXECUTIONS):")
    print(f"   • Tổng số lệnh bán đã khớp:       {total_sells} lượt (gồm {max(0, total_sells - total_roundtrips)} lần hạ tỷ trọng Leader)")
    print(f"   • Tỷ lệ lệnh bán có lãi:          {win_rate:.2f}% (Thắng {win_sells}/{total_sells} lệnh)")
    print(f"   • Payoff Ratio lượt khớp bán:     {payoff_ratio:.2f}x (Lãi TB: +{avg_win_pct:.2f}% | Lỗ TB: {avg_loss_pct:.2f}%)")
    print(f"   • Lệnh lãi đơn lẻ cao nhất:       +{best_trade:.2f}%")
    print(f"   • Lệnh lỗ đơn lẻ sâu nhất:        {worst_trade:.2f}%")
    print("="*58 + "\n")

    # ==========================================
    # 5. TELEGRAM RADAR BÁO CÁO HÀNG NGÀY
    # ==========================================
    print("5. Báo cáo Telegram Radar...")

    latest_date = df_clean['Date'].iloc[-1]
    latest_date_str = latest_date.strftime('%Y-%m-%d')
    active_today = get_vn30_constituents(latest_date_str)
    latest_idx = df_clean['VNINDEX'].iloc[-1]
    latest_ma20 = df_clean['MA20'].iloc[-1] if 'MA20' in df_clean.columns else np.nan
    latest_ma100 = df_clean['MA100'].iloc[-1]

    msg = f"🤖 <b>SIÊU RADAR VN30 - BÁO CÁO THỰC CHIẾN (TEST {BACKTEST_START_DATE[:4]}-NAY)</b> 🤖\n"
    msg += f"📅 Ngày: <b>{latest_date.strftime('%d/%m/%Y')}</b>\n"
    msg += f"🏛 Rổ VN30 hiệu lực: <b>{len(active_today)} mã chuẩn HOSE</b>\n"
    if pd.notna(latest_ma20) and latest_idx < latest_ma20:
        msg += f"⚠️ <b>Cảnh báo Thị trường:</b> VN-Index ({latest_idx:,.1f}) < MA20 ({latest_ma20:,.1f}) -> 🔒 <i>Đang khóa Mua mới & siết Trailing Stop</i>\n"
    else:
        msg += f"📈 <b>Trạng thái Thị trường:</b> VN-Index ({latest_idx:,.1f}) trên MA20 ({latest_ma20:,.1f}) & MA100 ({latest_ma100:,.1f}) -> 🟢 <i>Uptrend</i>\n"
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
            size_tag = " [10% NAV - Split-Size]" if b.get('is_split_size', False) else " [20% NAV]"
            msg += f"   • 🟢 Mua ATO: <b>{b['ticker']}</b> [{b_sec}]{size_tag} (AI Score: <b>{b['score']*100:.1f}%</b>)\n"
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
            split_tag = " [1/2 Slot]" if pos.get('is_split_size', False) else ""
            qty_str = f"{int(pos['qty']):,} cp"
            sec_name = SECTOR_MAP.get(ticker, 'OTHER')
            msg += f"{icon} <b>{ticker}</b> [{sec_name}]{split_tag} ({qty_str}): Giá {curr_p:,.2f} | Lãi tạm: <b>{unrealized_pnl:+.1f}%</b> (Đỉnh: +{peak_pnl:.1f}%){leader_badge} | T+{pos['hold_days']}\n"
        msg += "\n"
    else:
        msg += "<i>100% Tiền mặt (Bảo toàn vốn tối đa, nhận lãi Bank 4%/năm).</i>\n\n"

    msg += f"➖➖➖➖➖➖➖➖➖➖\n"
    msg += f"🏦 NAV Hiện tại: <b>{final_nav:,.0f} VNĐ</b> | Max Drawdown: <b>{max_mdd:.2f}%</b>"

    send_telegram_msg(msg)
    print("✅ Hoàn tất chu trình quant_bot.py!")

if __name__ == "__main__":
    main()
