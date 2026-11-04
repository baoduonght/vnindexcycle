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
# HÀM TẢI DỮ LIỆU ĐA NGUỒN (TRADINGVIEW + VNDIRECT FALLBACK)
# ==========================================
def fetch_ticker_data(symbol, n_bars=4000):
    try:
        df = tv.get_hist(symbol=symbol, exchange='HOSE', interval=Interval.in_daily, n_bars=n_bars)
        if df is not None and len(df) > 50:
            df = df.reset_index()
            df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    try:
        df = tv.get_hist(symbol=symbol, exchange='UPCOM', interval=Interval.in_daily, n_bars=n_bars)
        if df is not None and len(df) > 50:
            df = df.reset_index()
            df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    try:
        now_ts = int(datetime.datetime.now().timestamp())
        start_ts = int((datetime.datetime.now() - datetime.timedelta(days=12*365)).timestamp())
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={start_ts}&to={now_ts}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=12)
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
                return df.sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    return None

# ==========================================
# 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG
# ==========================================
print("1. Đang tải Market Proxy (VNINDEX)...")
vnindex_raw = fetch_ticker_data('VNINDEX', n_bars=4000)
if vnindex_raw is None or vnindex_raw.empty:
    sys.exit("❌ LỖI: Không thể tải dữ liệu VNINDEX.")

vnindex_df = vnindex_raw.rename(columns={'Close': 'VNINDEX'})
vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50).mean()
vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100).mean()
vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200).mean()
vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14) / vnindex_df['VNINDEX']
vnindex_df['VN_MFI'] = calc_mfi(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], vnindex_df['Volume'], 14)
df_clean = vnindex_df.dropna().copy().reset_index(drop=True)

# ==========================================
# 2. TẢI TOÀN BỘ RỔ CỔ PHIẾU VN30 LỊCH SỬ (POINT-IN-TIME)
# ==========================================
print(f"2. Đang tải dữ liệu {len(ALL_HISTORICAL_VN30_TICKERS)} mã VN30 lịch sử để AI học tập toàn diện...")

stocks_close = pd.DataFrame({'Date': df_clean['Date']}).set_index('Date')
stocks_volume = pd.DataFrame({'Date': df_clean['Date']}).set_index('Date')

loaded_tickers = []
for ticker in ALL_HISTORICAL_VN30_TICKERS:
    df_t = fetch_ticker_data(ticker, n_bars=4000)
    if df_t is not None and not df_t.empty:
        df_t.set_index('Date', inplace=True)
        stocks_close[ticker] = df_t['Close']
        stocks_volume[f'{ticker}_V'] = df_t['Volume']
        loaded_tickers.append(ticker)

print(f"   ✅ Đã nạp thành công {len(loaded_tickers)} mã vào tập huấn luyện.")

# KHÔNG DÙNG LỆNH FILL: Tính chỉ báo trực tiếp trên dữ liệu giao dịch thực tế của từng mã
stocks_ma50 = pd.DataFrame(index=stocks_close.index)
stocks_vol_ma20 = pd.DataFrame(index=stocks_close.index)
stocks_rsi = pd.DataFrame(index=stocks_close.index)
stocks_macd = pd.DataFrame(index=stocks_close.index)

for c in loaded_tickers:
    s_c = stocks_close[c].dropna()
    if len(s_c) >= 20:
        stocks_ma50[c] = s_c.rolling(50).mean()
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
    macro_df = macro_df[['Date', 'DXY', 'US10Y', 'USD_VND']]
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    fresh_us_dates = macro_df.dropna(subset=['DXY', 'US10Y', 'USD_VND'])['Date'].tolist()
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200']].copy()
market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']
market_context['Date'] = pd.to_datetime(market_context['Date']).dt.strftime('%Y-%m-%d')

macro_df = macro_df.merge(market_context[['Date', 'VN_MFI', 'VN_Dist_MA200']], on='Date', how='left').ffill()

macro_cols = ['DXY', 'US10Y', 'USD_VND']
for col in macro_cols:
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()
z_cols = [f'Z_{c}' for c in macro_cols]

clean_dates_str = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')
vn_close_aligned = vnindex_df.set_index('Date')['VNINDEX'].reindex(stocks_close.index).ffill()
# Lợi nhuận 30 phiên tới của VNINDEX
vn_future_return = vn_close_aligned.shift(-30) / vn_close_aligned - 1

df_ml_list = []
for c in loaded_tickers:
    s_close = stocks_close[c]
    s_vol = stocks_volume[f'{c}_V']
    s_vol_ma = stocks_vol_ma20[f'{c}_V']
    s_ma50 = stocks_ma50[c]

    valid_mask = s_close.notna() & (s_close > 0)
    if valid_mask.sum() < 60:
        continue

    # Lợi nhuận 30 phiên tới của cổ phiếu
    s_future_ret = s_close.shift(-30) / s_close - 1
    
    # KHẮC PHỤC TRIỆT ĐỂ LỖI LABEL POISONING:
    # Chỉ đánh nhãn Target khi cả s_future_ret VÀ vn_future_return đều KHÔNG NaN.
    # 30 phiên gần nhất không có dữ liệu tương lai sẽ là NaN và bị loại bỏ khỏi tập train,
    # thay vì bị ép về Target=0 khiến AI hiểu lầm thị trường sập!
    valid_future = s_future_ret.notna() & vn_future_return.notna()
    target_series = np.where(valid_future, (s_future_ret > vn_future_return).astype(int), np.nan)

    df_c = pd.DataFrame({
        'Date': clean_dates_str[valid_mask],
        'Ticker': c,
        'Vol_Ratio': (s_vol[valid_mask] / s_vol_ma[valid_mask].replace(0, np.nan)).values,
        'Dist_MA50': (s_close[valid_mask] / s_ma50[valid_mask]).values,
        'RSI': stocks_rsi[c][valid_mask].values,
        'RSI_Momentum': pd.Series(stocks_rsi[c][valid_mask]).diff(3).values,
        'MACD_Hist': stocks_macd[c][valid_mask].values,
        'Target': target_series[valid_mask]
    })
    df_ml_list.append(df_c)

ml_master = pd.concat(df_ml_list, ignore_index=True).merge(macro_df, on='Date', how='left')
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200'] + z_cols

# Chỉ lấy ngày có dữ liệu vĩ mô và tính năng hợp lệ
ml_master = ml_master[ml_master['Date'].isin(fresh_us_dates)]
ml_master_clean = ml_master.dropna(subset=ml_features).copy()

# ==========================================
# 4. HUẤN LUYỆN XGBOOST ALPHA ENGINE
# ==========================================
print("4. 🔄 KHỞI ĐỘNG LÒ LUYỆN ĐAN: HUẤN LUYỆN BỘ NÃO MỚI...")

# Lọc bỏ hoàn toàn các hàng có Target là NaN (30 ngày gần nhất)
train_data = ml_master_clean.dropna(subset=['Target']).copy()
train_data['Target'] = train_data['Target'].astype(int)

# Đánh trọng số thời gian: Dữ liệu gần (từ 2023) có trọng số x2.5
train_data['Weight'] = np.where(train_data['Date'] >= '2023-01-01', 2.5, 1.0)

# Chia tập Out-Of-Time (OOT) để kiểm tra độ tin cậy thực chiến:
# Train trên dữ liệu trước 2023, Test trên 2023 - 2024
train_mask = train_data['Date'] < '2023-01-01'
test_mask = train_data['Date'] >= '2023-01-01'

X_train = train_data.loc[train_mask, ml_features]
y_train = train_data.loc[train_mask, 'Target']
w_train = train_data.loc[train_mask, 'Weight']

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
    eval_metric='logloss'
)

print(f"   Đang huấn luyện trên {len(X_train)} mẫu dữ liệu lịch sử...")
xgb_model.fit(X_train, y_train, sample_weight=w_train)

# Đánh giá Out-Of-Time (OOT)
if len(X_test) > 0 and len(y_test.unique()) > 1:
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ OUT-OF-TIME (Dữ liệu ngoài mẫu 2023 - 2024):")
    print(f"   🎯 ROC-AUC Score: {auc:.4f}")
    print(f"   🎯 Precision:     {prec:.4f}")
    print(f"   🎯 Recall:        {rec:.4f}")

# Huấn luyện lại trên TOÀN BỘ dữ liệu để lưu vào file sản xuất
print("\nĐang tối ưu hóa bộ não trên toàn bộ tập dữ liệu tổng...")
xgb_model.fit(train_data[ml_features], train_data['Target'], sample_weight=train_data['Weight'])

MODEL_FILE = "xgboost_ultimate.json"
xgb_model.save_model(MODEL_FILE)

print(f"✅ XGBoost đã hoàn tất huấn luyện qua {len(train_data)} dòng dữ liệu!")
print(f"💾 Não bộ mới đã được lưu thành công tại: {MODEL_FILE}")

# Bảng xếp hạng các nhân tố quan trọng nhất
importances = pd.Series(xgb_model.feature_importances_, index=ml_features).sort_values(ascending=False)
print("\n🏆 BẢNG XẾP HẠNG TẦM QUAN TRỌNG CỦA CÁC BIẾN (FEATURE IMPORTANCES):")
for feat, imp in importances.items():
    print(f"   • {feat:<15}: {imp*100:.2f}%")
