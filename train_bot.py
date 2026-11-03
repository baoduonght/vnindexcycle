import pandas as pd
import numpy as np
import warnings
import sys
import os
from tvDatafeed import TvDatafeed, Interval
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings('ignore')
tv = TvDatafeed()

# ==========================================
# HÀM TÍNH TOÁN CHỈ BÁO NỘI BỘ (CHUẨN HÓA INDEX)
# ==========================================
def calc_adx(high, low, close, n=14):
    up = high - high.shift()
    down = low.shift() - low
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(n).mean()
    pdi = 100 * pdm.rolling(n).mean() / atr.replace(0, np.nan)
    ndi = 100 * ndm.rolling(n).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.rolling(n).mean()

def calc_atr(high, low, close, n=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_mfi(high, low, close, volume, n=14):
    tp = (high + low + close) / 3
    rmf = tp * volume
    diff = tp.diff()
    pos_mf = rmf.where(diff > 0, 0.0).rolling(n).sum()
    neg_mf = rmf.where(diff < 0, 0.0).rolling(n).sum()
    return 100 - (100 / (1 + (pos_mf / neg_mf.replace(0, np.nan))))

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
# 2. TẢI DỮ LIỆU VN30 & CHỈ BÁO KỸ THUẬT (GIỮ TRỌN VẸN LỊCH SỬ)
# ==========================================
print("2. Đang tải rổ VN30...")
raw_tickers = ['FPT', 'HPG', 'SSI', 'MWG', 'PNJ', 'TCB', 'MBB', 'VPB', 'ACB', 'VCB']
stocks_close, stocks_volume = pd.DataFrame(), pd.DataFrame()

valid_tickers = []
for ticker in raw_tickers:
    try:
        df_temp = tv.get_hist(symbol=ticker, exchange='HOSE', interval=Interval.in_daily, n_bars=4000)
        if df_temp is not None and not df_temp.empty:
            df_temp = df_temp.reset_index()
            df_temp.rename(columns={'datetime': 'Date', 'close': ticker, 'volume': f'{ticker}_V'}, inplace=True)
            df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.tz_localize(None).dt.normalize()
            df_temp.set_index('Date', inplace=True)
            
            if stocks_close.empty:
                stocks_close = df_temp[[ticker]]
                stocks_volume = df_temp[[f'{ticker}_V']]
            else:
                stocks_close = stocks_close.join(df_temp[[ticker]], how='outer')
                stocks_volume = stocks_volume.join(df_temp[[f'{ticker}_V']], how='outer')
            valid_tickers.append(ticker)
    except Exception as e:
        print(f"⚠️ Bỏ qua {ticker} do lỗi kết nối: {e}")

# Giữ nguyên toàn bộ lịch sử (Không dropna làm mất quá khứ của các mã lên sàn sớm)
stock_tickers = [c for c in valid_tickers if c in stocks_close.columns]

# ==========================================
# 3. CHUẨN BỊ BỘ DỮ LIỆU ĐỂ AI HỌC (TRAIN DATA ĐỘC LẬP TỪNG MÃ)
# ==========================================
print("3. Khởi tạo màng lưới tính toán (Giữ trọn vẹn 100% lịch sử từng mã)...")
try:
    macro_df = pd.read_csv('markets_database10years.csv')
    macro_df = macro_df[['Date', 'DXY', 'US10Y', 'USD_VND']]
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.tz_localize(None).dt.normalize()
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200']].copy()
market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']

macro_df = macro_df.merge(market_context[['Date', 'VN_MFI', 'VN_Dist_MA200']], on='Date', how='left').ffill()

macro_cols = ['DXY', 'US10Y', 'USD_VND']
for col in macro_cols:
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()
z_cols = [f'Z_{c}' for c in macro_cols]

# Đồng bộ VNINDEX theo dòng thời gian
vn_series = vnindex_df.set_index('Date')['VNINDEX'].sort_index()
vn_future_return = vn_series.shift(-30) / vn_series - 1

df_ml_list = []
for c in stock_tickers:
    # Lấy đúng chuỗi ngày mà mã c THỰC TẾ CÓ GIAO DỊCH TRONG LỊCH SỬ
    s_price = stocks_close[c].dropna()
    if len(s_price) < 100: 
        continue  # Mã mới lên sàn dưới 100 ngày thì bỏ qua
        
    valid_dates = s_price.index
    s_vol = stocks_volume[f'{c}_V'].reindex(valid_dates).fillna(0)
    
    # Tính toán chỉ báo kỹ thuật riêng cho mã c trên đúng dòng thời gian của nó
    s_ma50 = s_price.rolling(50).mean()
    s_vol_ma20 = s_vol.rolling(20).mean().shift(1)
    s_rsi = calc_rsi(s_price)
    s_macd = calc_macd_hist(s_price)
    
    df_c = pd.DataFrame({'Date': valid_dates, 'Ticker': c})
    df_c['Vol_Ratio'] = (s_vol / s_vol_ma20.replace(0, np.nan)).values
    df_c['Dist_MA50'] = (s_price / s_ma50.replace(0, np.nan)).values
    df_c['RSI'] = s_rsi.values
    df_c['RSI_Momentum'] = pd.Series(df_c['RSI']).diff(3).values 
    df_c['MACD_Hist'] = s_macd.values
    
    # Lợi nhuận tương lai của cổ phiếu vs VNINDEX
    stock_future_return = s_price.shift(-30) / s_price - 1
    vn_future_aligned = vn_future_return.reindex(valid_dates)
    
    valid_mask = stock_future_return.notna() & vn_future_aligned.notna()
    target_series = pd.Series(np.nan, index=valid_dates)
    target_series[valid_mask] = (stock_future_return[valid_mask] > vn_future_aligned[valid_mask]).astype(int)
    
    df_c['Target'] = target_series.values
    df_ml_list.append(df_c)

ml_master = pd.concat(df_ml_list, ignore_index=True).merge(macro_df, on='Date', how='left')
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200'] + z_cols

# Bỏ hàng thiếu Feature hoặc thiếu Target (chỉ bỏ 30 ngày gần nhất chưa có kết quả)
ml_master_clean = ml_master.dropna(subset=ml_features + ['Target']).copy()
ml_master_clean['Target'] = ml_master_clean['Target'].astype(int)

if ml_master_clean.empty:
    sys.exit("⚠️ LỖI ĐỒNG BỘ: Tập dữ liệu bị rỗng.")

print(f"📊 Tổng số mẫu dữ liệu AI được học (Đầy đủ lịch sử): {len(ml_master_clean)} dòng!")
# ==========================================
# 4. HUẤN LUYỆN & KIỂM THỬ ĐỘC LẬP
# ==========================================
print("4. 🔄 KHỞI ĐỘNG LÒ LUYỆN (THỨ 7): ĐÚC BỘ NÃO MỚI...")

# Trọng số x3 cho dữ liệu thực chiến từ 2024
ml_master_clean['Weight'] = np.where(ml_master_clean['Date'] >= '2024-01-01', 3.0, 1.0)

# Cắt 15% phiên gần nhất làm tập kiểm thử độc lập (Out-of-sample)
unique_dates = np.sort(ml_master_clean['Date'].unique())
split_idx = int(len(unique_dates) * 0.85)
split_date = unique_dates[split_idx]

train_df = ml_master_clean[ml_master_clean['Date'] < split_date]
val_df = ml_master_clean[ml_master_clean['Date'] >= split_date]

X_train, y_train, w_train = train_df[ml_features], train_df['Target'], train_df['Weight']
X_val, y_val = val_df[ml_features], val_df['Target']

scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1

xgb_model = xgb.XGBClassifier(
    n_estimators=200,              
    max_depth=5,                 
    learning_rate=0.03,            
    subsample=0.8,                 
    colsample_bytree=0.8,          
    scale_pos_weight=scale_weight, 
    random_state=42,
    eval_metric='logloss'
)

# Kiểm tra chất lượng mô hình trên dữ liệu kiểm định
xgb_model.fit(X_train, y_train, sample_weight=w_train)
val_preds_proba = xgb_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds_proba)
val_acc = accuracy_score(y_val, (val_preds_proba >= 0.5).astype(int))

print(f"\n📊 KẾT QUẢ THI ĐẤU ĐỘC LẬP (Dữ liệu từ {pd.to_datetime(split_date).strftime('%d/%m/%Y')}):")
print(f"🎯 ROC-AUC Score: {val_auc:.3f} (Chuẩn: > 0.55 là có Alpha)")
print(f"🎯 Accuracy: {val_acc * 100:.2f}%")

# Tối ưu hóa 100% dữ liệu để xuất file bộ não tốt nhất
print("\nĐang tối ưu hóa toàn bộ bộ nhớ để đóng gói...")
xgb_model.fit(ml_master_clean[ml_features], ml_master_clean['Target'], sample_weight=ml_master_clean['Weight'])

MODEL_FILE = "xgboost_ultimate.json"
xgb_model.save_model(MODEL_FILE)

print(f"✅ Đã lưu bộ não chuẩn tại: {MODEL_FILE}")
