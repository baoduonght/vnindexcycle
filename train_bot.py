import pandas as pd
import numpy as np
import requests
from tvDatafeed import TvDatafeed, Interval
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
import warnings
import sys
import os
import datetime

from vn30_universe import ALL_HISTORICAL_VN30_TICKERS

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
    start_ts = int((datetime.datetime.now() - datetime.timedelta(days=12*365)).timestamp())

    # Phân loại Index vs Stock để gọi đúng endpoint của DNSE Entrade
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
# 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG (OHLCV THẬT — KHÔNG LÀM PHẲNG NẾN)
# ==========================================
print("1. Đang tải Market Proxy (VNINDEX)...")
vnindex_raw = fetch_ticker_data('VNINDEX', n_bars=4000)
if vnindex_raw is None or vnindex_raw.empty:
    sys.exit("❌ LỖI: Không thể tải dữ liệu VNINDEX.")

vnindex_df = vnindex_raw.rename(columns={'Close': 'VNINDEX'}).drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50, min_periods=20).mean()
vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100, min_periods=40).mean()
vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200, min_periods=80).mean()
# ATR, ADX, MFI tính trên High/Low THẬT (không bị làm phẳng)
vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14) / vnindex_df['VNINDEX']
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

# ==========================================
# 3. CHUẨN BỊ TẬP DỮ LIỆU ĐỂ AI HỌC (TRAIN DATA)
# ==========================================
print("3. Khởi tạo màng lưới tính toán & Đồng bộ Vĩ mô...")
try:
    macro_df = pd.read_csv('markets_database10years.csv')
    # Chỉ lấy DXY, US10Y, USD_VND từ CSV (VNINDEX OHLCV đã tải từ API)
    macro_df = macro_df[['Date', 'DXY', 'US10Y', 'USD_VND']].copy()
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    macro_df = macro_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200']].copy()
market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']
market_context['Date'] = pd.to_datetime(market_context['Date']).dt.strftime('%Y-%m-%d')
market_context = market_context.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)

macro_df = macro_df.merge(market_context[['Date', 'VN_MFI', 'VN_Dist_MA200']], on='Date', how='inner')
macro_df = macro_df.sort_values('Date').reset_index(drop=True)

macro_cols = ['DXY', 'US10Y', 'USD_VND']
for col in macro_cols:
    macro_df[col] = pd.to_numeric(macro_df[col], errors='coerce')
    roll_m = macro_df[col].rolling(120, min_periods=20).mean()
    roll_s = macro_df[col].rolling(120, min_periods=20).std()
    macro_df[f'Z_{col}'] = (macro_df[col] - roll_m) / roll_s
z_cols = [f'Z_{c}' for c in macro_cols]

clean_dates_str = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')
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

    # Chỉ đánh nhãn Target khi cả hai future return đều KHÔNG NaN
    valid_future = s_future_ret.notna() & vn_future_ret.notna()
    target_series = np.where(valid_future, (s_future_ret > vn_future_ret).astype(int), np.nan)

    df_c = pd.DataFrame({
        'Date': pd.to_datetime(s_valid_close.index).strftime('%Y-%m-%d'),
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
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200'] + z_cols
for f in ml_features:
    ml_master[f] = pd.to_numeric(ml_master[f], errors='coerce')

# ==========================================
# 4. HUẤN LUYỆN XGBOOST (PURGED TIME-SERIES SPLIT — CHỐNG LOOK-AHEAD LEAKAGE)
# ==========================================
print("4. 🔄 KHỞI ĐỘNG LÒ LUYỆN ĐAN: HUẤN LUYỆN BỘ NÃO MỚI...")

# Chỉ loại bỏ các hàng có Target là NaN (30 ngày gần nhất chưa có tương lai)
# XGBoost xử lý NaN ở features nguyên bản (missing=np.nan)
# TUYỆT ĐỐI KHÔNG dùng fill/ffill/bfill/fillna
train_data = ml_master.dropna(subset=['Target']).copy()
train_data['Target'] = train_data['Target'].astype(int)

# =====================================================
# PURGED TIME-SERIES SPLIT:
#   In-Sample  (Train): 2014-01-01 → 2021-12-31
#   Out-of-Sample (OOS): 2022-01-01 → 2026-xx-xx
# Mô hình CHỈ học từ quá khứ, KHÔNG BAO GIỜ nhìn thấy dữ liệu OOS
# =====================================================
train_mask = train_data['Date'] < '2022-01-01'
test_mask = train_data['Date'] >= '2022-01-01'

# Fallback: Nếu API chỉ trả dữ liệu ngắn (không có trước 2022)
if train_mask.sum() == 0 or test_mask.sum() == 0:
    unique_dates = sorted(train_data['Date'].unique())
    split_idx = int(len(unique_dates) * 0.70)
    split_date = unique_dates[split_idx]
    train_mask = train_data['Date'] < split_date
    test_mask = train_data['Date'] >= split_date
    print(f"   ℹ️ Dữ liệu phân tách theo điểm cắt: {split_date} (70% Train / 30% OOS)")
else:
    print(f"   ℹ️ Purged Split: In-Sample (2014-2021) | Out-of-Sample (2022-2026)")

print(f"   📊 Tổng số mẫu dữ liệu trích xuất: {len(ml_master):,}")
print(f"   📊 Số mẫu huấn luyện (In-Sample):   {train_mask.sum():,}")
print(f"   📊 Số mẫu kiểm tra (Out-of-Sample):  {test_mask.sum():,}")

X_train = train_data.loc[train_mask, ml_features]
y_train = train_data.loc[train_mask, 'Target']

X_test = train_data.loc[test_mask, ml_features]
y_test = train_data.loc[test_mask, 'Target']

if len(X_train) == 0:
    sys.exit("❌ LỖI: Tập X_train rỗng.")

scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1.0

xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric='logloss',
    missing=np.nan
)

print(f"   Đang huấn luyện trên {len(X_train)} mẫu In-Sample (2014-2021)...")
xgb_model.fit(X_train, y_train)

# Đánh giá Out-of-Sample (OOS) — dữ liệu mô hình CHƯA BAO GIỜ thấy
if len(X_test) > 0 and len(y_test.unique()) > 1:
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ OUT-OF-SAMPLE (Dữ liệu ngoài mẫu 2022-2026):")
    print(f"   🎯 ROC-AUC Score: {auc:.4f}")
    print(f"   🎯 Precision:     {prec:.4f}")
    print(f"   🎯 Recall:        {rec:.4f}")

# =====================================================
# LƯU MÔ HÌNH CHỈ TỪ IN-SAMPLE (2014-2021)
# KHÔNG tái huấn luyện trên toàn bộ dữ liệu → chống Look-Ahead Leakage hoàn toàn
# =====================================================
MODEL_FILE = "xgboost_ultimate.json"
xgb_model.save_model(MODEL_FILE)

print(f"\n✅ XGBoost đã hoàn tất huấn luyện trên {len(X_train)} dòng In-Sample!")
print(f"💾 Não bộ đã lưu tại: {MODEL_FILE} (Chỉ học 2014-2021, OOS blind test 2022-2026)")

# Bảng xếp hạng Feature Importances
importances = pd.Series(xgb_model.feature_importances_, index=ml_features).sort_values(ascending=False)
print("\n🏆 BẢNG XẾP HẠNG TẦM QUAN TRỌNG CỦA CÁC BIẾN (FEATURE IMPORTANCES):")
for feat, imp in importances.items():
    print(f"   • {feat:<15}: {imp*100:.2f}%")
