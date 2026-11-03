import pandas as pd
import numpy as np
import requests
from tvDatafeed import TvDatafeed, Interval
import xgboost as xgb
import warnings
import sys
import os

warnings.filterwarnings('ignore')
tv = TvDatafeed()

# ==========================================
# HÀM TÍNH TOÁN CHỈ BÁO NỘI BỘ
# ==========================================
def calc_adx(high, low, close, n=14):
    up = high - high.shift()
    down = low.shift() - low
    pdm = np.where((up > down) & (up > 0), up, 0)
    ndm = np.where((down > up) & (down > 0), down, 0)
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr = pd.Series(tr).rolling(n).mean()
    pdi = 100 * pd.Series(pdm).rolling(n).mean() / atr
    ndi = 100 * pd.Series(ndm).rolling(n).mean() / atr
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
# 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG
# ==========================================
print("1. Đang tải Market Proxy (VNINDEX)...")
vnindex_df = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=4000)
if vnindex_df is None or vnindex_df.empty:
    sys.exit("❌ LỖI: API TradingView không trả về dữ liệu VNINDEX.")
vnindex_df = vnindex_df.reset_index()
vnindex_df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'VNINDEX', 'volume': 'Volume'}, inplace=True)
vnindex_df['Date'] = pd.to_datetime(vnindex_df['Date']).dt.tz_localize(None).dt.normalize()

vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50).mean()
vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100).mean()
vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200).mean()
vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14) / vnindex_df['VNINDEX']
vnindex_df['VN_MFI'] = calc_mfi(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], vnindex_df['Volume'], 14)

# ==========================================
# 2. TẢI DỮ LIỆU VN30 & CHỈ BÁO KỸ THUẬT
# ==========================================
print("2. Đang tải rổ VN30...")
stock_tickers = ['FPT', 'HPG', 'SSI', 'MWG', 'PNJ', 'TCB', 'MBB', 'VPB', 'ACB', 'VCB']
stocks_close, stocks_volume = pd.DataFrame(), pd.DataFrame()

for ticker in stock_tickers:
    df_temp = tv.get_hist(symbol=ticker, exchange='HOSE', interval=Interval.in_daily, n_bars=4000)
    if df_temp is not None and not df_temp.empty:
        df_temp = df_temp.reset_index()
        df_temp.rename(columns={'datetime': 'Date', 'close': ticker, 'volume': f'{ticker}_V'}, inplace=True)
        df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.tz_localize(None).dt.normalize()
        df_temp.set_index('Date', inplace=True)
        if stocks_close.empty:
            stocks_close, stocks_volume = df_temp[[ticker]], df_temp[[f'{ticker}_V']]
        else:
            stocks_close = stocks_close.join(df_temp[[ticker]], how='outer')
            stocks_volume = stocks_volume.join(df_temp[[f'{ticker}_V']], how='outer')

stocks_close = stocks_close.ffill().bfill()
stocks_volume = stocks_volume.ffill().fillna(0)
stocks_high_20 = stocks_close.rolling(20).max().shift(1)
stocks_ma20 = stocks_close.rolling(20).mean()
stocks_ma50 = stocks_close.rolling(50).mean()
stocks_vol_ma20 = stocks_volume.rolling(20).mean().shift(1)

# ==========================================
# 3. CHUẨN BỊ BỘ DỮ LIỆU ĐỂ AI HỌC (TRAIN DATA)
# ==========================================
print("3. Khởi tạo màng lưới tính toán (Bản Sniper)...")
try:
    # Đọc file tổng hợp từ hệ thống Cào Vĩ mô của bác
    macro_df = pd.read_csv('markets_database10years.csv')
    macro_df = macro_df[['Date', 'DXY', 'US10Y', 'USD_VND']]
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    fresh_us_dates = macro_df.dropna(subset=['DXY', 'US10Y', 'USD_VND'])['Date'].tolist()
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}. Bác nhớ cập nhật file CSV lên Github nhé!")

market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200']].copy()
market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']
market_context['Date'] = pd.to_datetime(market_context['Date']).dt.strftime('%Y-%m-%d')

macro_df = macro_df.merge(market_context[['Date', 'VN_MFI', 'VN_Dist_MA200']], on='Date', how='left').ffill()

# Tính Z-Score cho 3 biến (Kể cả Tỷ giá)
macro_cols = ['DXY', 'US10Y', 'USD_VND']
for col in macro_cols:
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()
z_cols = [f'Z_{c}' for c in macro_cols]

df_ml_list = []
clean_dates = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')
# Căn chuẩn giá VNINDEX theo lịch giao dịch của rổ VN30
vn_close_aligned = vnindex_df.set_index('Date')['VNINDEX'].reindex(stocks_close.index).ffill()
# Tính lợi nhuận tương lai 30 phiên của chính VNINDEX
vn_future_return = vn_close_aligned.shift(-30) / vn_close_aligned - 1

df_ml_list = []
clean_dates = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')
for c in stock_tickers:
    df_c = pd.DataFrame({'Date': clean_dates, 'Ticker': c})
    df_c['Vol_Ratio'] = (stocks_volume[f'{c}_V'] / stocks_vol_ma20[f'{c}_V']).values
    df_c['Dist_MA50'] = (stocks_close[c] / stocks_ma50[c]).values
    df_c['RSI'] = calc_rsi(stocks_close[c]).values
    df_c['RSI_Momentum'] = pd.Series(df_c['RSI']).diff(3).values 
    df_c['MACD_Hist'] = calc_macd_hist(stocks_close[c]).values
    
    # PHƯƠNG PHÁP ALPHA: ĐÁNH BẠI VNINDEX
    stock_future_return = stocks_close[c].shift(-30) / stocks_close[c] - 1
    # Đánh nhãn (Target = 1) CHỈ KHI lợi nhuận mã này LỚN HƠN lợi nhuận VNINDEX
    df_c['Target'] = (stock_future_return > vn_future_return).astype(int).values
    df_ml_list.append(df_c)

ml_master = pd.concat(df_ml_list).merge(macro_df, on='Date', how='left')
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200'] + z_cols
ml_master = ml_master[ml_master['Date'].isin(fresh_us_dates)]
ml_master_clean = ml_master.dropna(subset=ml_features).copy()

if ml_master_clean.empty:
    sys.exit("⚠️ LỖI ĐỒNG BỘ: Tập dữ liệu bị rỗng.")

# ==========================================
# 4. ÉP AI HỌC VÀ LƯU BỘ NHỚ (HYBRID TRAINING)
# ==========================================
print("4. 🔄 KHỞI ĐỘNG LÒ LUYỆN (THỨ 7): ĐÚC BỘ NÃO MỚI...")
train_data = ml_master_clean.dropna(subset=['Target']).copy()

# Đánh trọng số x3 cho dữ liệu thực chiến từ 2024
train_data['Weight'] = np.where(train_data['Date'] >= '2024-01-01', 3.0, 1.0)

X_train = train_data[ml_features]
y_train = train_data['Target']
w_train = train_data['Weight']

if len(X_train) == 0:
    sys.exit("❌ LỖI TRÍ MẠNG: Tập X_train rỗng.")

scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1

# Tinh chỉnh lại lõi AI (Học sâu hơn một chút để xử lý 10 biến)
xgb_model = xgb.XGBClassifier(
    n_estimators=150,              
    max_depth=5, # Tăng nhẹ độ sâu để AI hiểu liên kết chéo giữa Tỷ giá và Trái phiếu                  
    learning_rate=0.05,            
    subsample=0.8,                 
    colsample_bytree=0.8,          
    scale_pos_weight=scale_weight, 
    random_state=42,
    eval_metric='logloss'
)

print("Đang nạp dữ liệu vào Mạng nơ-ron XGBoost (Kết hợp Trọng số thời gian x3)...")
xgb_model.fit(X_train, y_train, sample_weight=w_train)

# Xuất bộ não ra file JSON để file quant_bot.py móc ra xài hàng ngày
MODEL_FILE = "xgboost_ultimate.json"
xgb_model.save_model(MODEL_FILE)

print(f"✅ XGBoost đã tiến hóa xong qua {len(X_train)} dòng dữ liệu!")
print(f"💾 Não bộ mới đã được lưu thành công tại: {MODEL_FILE}")
