import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tvDatafeed import TvDatafeed, Interval
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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

# 🟢 HÀM MỚI: TÍNH ĐỘNG LƯỢNG MACD HISTOGRAM
def calc_macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal

# ==========================================
# 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG
# ==========================================
print("1. Đang tải Market Proxy (VNINDEX) và tính toán Dòng tiền MFI...")
vnindex_df = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=4000)
if vnindex_df is None or vnindex_df.empty:
    sys.exit("❌ LỖI: API TradingView không trả về dữ liệu VNINDEX. Hãy chạy lại sau.")
vnindex_df = vnindex_df.reset_index()
vnindex_df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'VNINDEX', 'volume': 'Volume'}, inplace=True)
vnindex_df['Date'] = pd.to_datetime(vnindex_df['Date']).dt.tz_localize(None).dt.normalize()

vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50).mean()
vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100).mean()
vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200).mean()
vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14) / vnindex_df['VNINDEX']
vnindex_df['VN_MFI'] = calc_mfi(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], vnindex_df['Volume'], 14)

df_clean = vnindex_df.dropna().copy()

# ==========================================
# 2. TẢI DỮ LIỆU VN30 & CHỈ BÁO KỸ THUẬT CÁ NHÂN
# ==========================================
print("2. Đang tải rổ VN30 & Tính toán màng lọc...")
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

stocks_rsi = pd.DataFrame(index=stocks_close.index)
for c in stock_tickers:
    stocks_rsi[c] = calc_rsi(stocks_close[c])

import xgboost as xgb

import xgboost as xgb
from sklearn.metrics import classification_report

# ==========================================
# 3. NHÚNG AI - HUẤN LUYỆN & SÁT HẠCH (BẢN XGBOOST SNIPER)
# ==========================================
print("3. Nâng cấp Não bộ AI (Nhúng La bàn VNINDEX & Động cơ XGBoost)...")

try:
    macro_df = pd.read_csv('global_markets_database10years.csv')
    macro_df = macro_df[['Date', 'DXY', 'US10Y']]
    macro_df['DXY'] = pd.to_numeric(macro_df['DXY'], errors='coerce')
    macro_df['US10Y'] = pd.to_numeric(macro_df['US10Y'], errors='coerce')
    macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    fresh_us_dates = macro_df.dropna(subset=['DXY', 'US10Y'])['Date'].tolist()
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

# 🟢 TÍCH HỢP LA BÀN VNINDEX VÀ DÒNG TIỀN MFI
market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200']].copy()
market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']
market_context['Date'] = pd.to_datetime(market_context['Date']).dt.strftime('%Y-%m-%d')

macro_df = macro_df.merge(market_context[['Date', 'VN_MFI', 'VN_Dist_MA200']], on='Date', how='left')
macro_df['VN_MFI'] = macro_df['VN_MFI'].ffill()
macro_df['VN_Dist_MA200'] = macro_df['VN_Dist_MA200'].ffill()

macro_cols = ['DXY', 'US10Y']
for col in macro_cols:
    macro_df[col] = macro_df[col].ffill()
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()
z_cols = [f'Z_{c}' for c in macro_cols]

df_ml_list = []
clean_dates = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')

for c in stock_tickers:
    df_c = pd.DataFrame({'Date': clean_dates})
    df_c['Ticker'] = c
    df_c['Vol_Ratio'] = (stocks_volume[f'{c}_V'] / stocks_vol_ma20[f'{c}_V']).values
    df_c['Dist_MA50'] = (stocks_close[c] / stocks_ma50[c]).values
    df_c['RSI'] = calc_rsi(stocks_close[c]).values
    df_c['MACD_Hist'] = calc_macd_hist(stocks_close[c]).values
    
    # 🟢 TARGET MỚI TỪ BÁC: Kỳ vọng tăng > 5% trong vòng 30 phiên (T+30)
    future_return = stocks_close[c].shift(-30) / stocks_close[c] - 1
    df_c['Target'] = (future_return > 0.02).astype(int).values
    df_ml_list.append(df_c)

concat_df = pd.concat(df_ml_list)
ml_master = concat_df.merge(macro_df, on='Date', how='left')

# 🟢 MẮT THẦN: Đầy đủ 8 chỉ báo
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200'] + z_cols

ml_master = ml_master[ml_master['Date'].isin(fresh_us_dates)]
ml_master_clean = ml_master.dropna(subset=ml_features).copy()

if ml_master_clean.empty:
    sys.exit("⚠️ LỖI ĐỒNG BỘ: Bot đứng ngoài do thiếu Data VN-US.")

train_data_all = ml_master_clean.dropna(subset=['Target'])
train_data = train_data_all[train_data_all['Date'] < '2024-01-01']

X_train = train_data[ml_features]
y_train = train_data['Target']

if len(X_train) == 0:
    sys.exit("❌ LỖI TRÍ MẠNG: Tập X_train rỗng.")

# 🟢 TÍNH TOÁN TRỌNG SỐ LỆNH MUA (Chống mất cân bằng dữ liệu)
scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1

# 🟢 KHỞI ĐỘNG ĐỘNG CƠ XGBOOST
xgb_model = xgb.XGBClassifier(
    n_estimators=150,              
    max_depth=4,                   
    learning_rate=0.05,            
    subsample=0.8,                 
    colsample_bytree=0.8,          # Ép bốc thăm ngẫu nhiên chỉ báo để trị bệnh ngáo DXY
    scale_pos_weight=scale_weight, 
    random_state=42,
    eval_metric='logloss'
)

print("Đang nạp dữ liệu vào Mạng nơ-ron XGBoost...")
xgb_model.fit(X_train, y_train)
print(f"✅ XGBoost đã tiến hóa xong qua {len(X_train)} dòng dữ liệu!")

# ==========================================
# BẢNG ĐIỂM SÁT HẠCH XGBOOST (CHẾ ĐỘ SNIPER)
# ==========================================
test_data = train_data_all[train_data_all['Date'] >= '2024-01-01']
if not test_data.empty:
    y_test = test_data['Target']
    
    # Lấy điểm xác suất (Probability) thay vì kết quả 0/1 mặc định
    y_probs = xgb_model.predict_proba(test_data[ml_features])[:, 1]
    
    # 🟢 KÍCH HOẠT SNIPER MODE: Ép ngưỡng tự tin lên 75% mới báo MUA
    SNIPER_THRESHOLD = 0.75
    y_pred_sniper = (y_probs >= SNIPER_THRESHOLD).astype(int)
    
    print("\n" + "="*50)
    print(f"📊 BẢNG ĐIỂM XGBOOST - SNIPER MODE (NGƯỠNG {SNIPER_THRESHOLD*100}%)")
    print("="*50)
    print(classification_report(y_test, y_pred_sniper, target_names=['Bỏ qua (0)', 'Đạt chuẩn MUA (1)']))
    
    print("\n🔍 TỶ TRỌNG SỬ DỤNG CHỈ BÁO (FEATURE IMPORTANCE):")
    importances = xgb_model.feature_importances_
    ranked_features = sorted(zip(ml_features, importances), key=lambda x: x[1], reverse=True)
    for feature, imp in ranked_features:
        print(f" 🔹 {feature:<14}: {imp*100:>5.2f}%")
    print("="*50 + "\n")

# 🟢 CHỐT ĐUÔI: ĐIỀN ĐIỂM SỐ ĐỂ XUẤT RA TELEGRAM (Không được xóa)
ml_master['AI_Score'] = float('nan')
valid_rows = ml_master[ml_features].notna().all(axis=1)
if valid_rows.sum() > 0:
    ml_master.loc[valid_rows, 'AI_Score'] = xgb_model.predict_proba(ml_master.loc[valid_rows, ml_features])[:, 1]

ai_dict = {c: ml_master[ml_master['Ticker'] == c].set_index('Date')['AI_Score'] for c in stock_tickers}

# ==========================================
# 4. CHẠY BACKTEST (SỔ KẾ TOÁN)
# ==========================================
print("4. Chạy Sổ Kế Toán & Trích xuất Lịch sử Mua/Bán của AI...")
def run_ai_quant_system_with_log():
    backtest = df_clean[['Date', 'VNINDEX', 'MA50', 'MA100', 'MA200', 'ADX', 'ATR_PCT', 'VN_MFI']].copy()
    MAX_SLOTS = 5
    LIQUIDITY_FEE = 0.003
    DAILY_INTEREST = 0.05 / 252
    cash = 1000000000
    held = {}
    nav_history = np.zeros(len(backtest))
    trade_log = [] 

    for i in range(len(backtest)):
        date = backtest['Date'].iloc[i]
        idx = backtest['VNINDEX'].iloc[i]
        market_ma50 = backtest['MA50'].iloc[i]
        market_ma100 = backtest['MA100'].iloc[i]
        market_ma200 = backtest['MA200'].iloc[i]
        market_adx = backtest['ADX'].iloc[i]
        market_atr_pct = backtest['ATR_PCT'].iloc[i]
        market_mfi = backtest['VN_MFI'].iloc[i] 

        if date not in stocks_close.index:
            nav_history[i] = nav_history[i-1] if i > 0 else cash
            continue

        cash = cash * (1 + DAILY_INTEREST)

        if idx < market_ma100:
            for c in list(held.keys()):
                sell_price = stocks_close.at[date, c]
                cash += held[c]['qty'] * sell_price * (1 - LIQUIDITY_FEE)
                profit_pct = (sell_price - held[c]['buy_price']) / held[c]['buy_price'] * 100
                trade_log.append({'Date': date, 'Action': 'SELL (Cắt giảm)', 'Ticker': c, 'Price': sell_price, 'Profit(%)': round(profit_pct, 2)})
            held.clear()
        else:
            to_remove = []
            for c in held.keys():
                curr_price = stocks_close.at[date, c]
                buy_price = held[c]['buy_price']
                held[c]['highest'] = max(held[c]['highest'], curr_price)

                profit_margin = (curr_price - buy_price) / buy_price
                peak_margin = (held[c]['highest'] - buy_price) / buy_price

                if profit_margin <= -0.08: to_remove.append(c)
                elif profit_margin >= 0.20: to_remove.append(c)
                elif peak_margin >= 0.12 and curr_price < held[c]['highest'] * 0.94: to_remove.append(c)
                elif curr_price < stocks_ma20.at[date, c] and profit_margin > 0.05: to_remove.append(c)

            for c in to_remove:
                sell_price = stocks_close.at[date, c]
                cash += held[c]['qty'] * sell_price * (1 - LIQUIDITY_FEE)
                profit_pct = (sell_price - held[c]['buy_price']) / held[c]['buy_price'] * 100
                trade_log.append({'Date': date, 'Action': 'SELL', 'Ticker': c, 'Price': sell_price, 'Profit(%)': round(profit_pct, 2)})
                del held[c]

            max_exposure = 0.3 if (idx < market_ma50 or market_atr_pct > 0.018) else 1.0
            current_total_nav = cash + sum([held[c]['qty'] * stocks_close.at[date, c] for c in held.keys()])
            allowed_capital_to_invest = (current_total_nav * max_exposure) - sum([held[c]['qty'] * stocks_close.at[date, c] for c in held.keys()])

            empty = MAX_SLOTS - len(held)
            if empty > 0 and (idx > market_ma200) and (market_adx > 22) and (allowed_capital_to_invest > 0):
                buy_candidates = []
                for c in stock_tickers:
                    if c not in held:
                        curr_price = stocks_close.at[date, c]
                        curr_vol = stocks_volume.at[date, f'{c}_V']
                        prev_high_20 = stocks_high_20.at[date, c]
                        prev_vol_ma20 = stocks_vol_ma20.at[date, f'{c}_V']

                        if pd.isna(prev_high_20) or pd.isna(prev_vol_ma20): continue

                        if curr_price > prev_high_20 and curr_vol > (prev_vol_ma20 * 1.5):
                            ai_score = ai_dict[c].get(date, 0)
                            if pd.isna(ai_score): ai_score = 0
                            
                            # 🟢 SIẾT KỶ LUẬT THÉP: Điểm AI phải tuyệt đối tự tin (> 65%) mới xét mua
                            if ai_score >= 0.65:
                                vol_ratio = curr_vol / prev_vol_ma20 if prev_vol_ma20 > 0 else 0
                                stock_rsi = stocks_rsi.at[date, c]
                                buy_candidates.append((c, ai_score, vol_ratio, stock_rsi))

                if buy_candidates:
                    invest_amt = min(allowed_capital_to_invest / empty, cash)
                    for c, score, v_ratio, s_rsi in buy_candidates:
                        if invest_amt > 0 and stocks_close.at[date, c] > 0 and cash >= invest_amt:
                            cash -= invest_amt
                            buy_price = stocks_close.at[date, c]
                            qty = (invest_amt * (1 - LIQUIDITY_FEE)) / buy_price
                            held[c] = {'qty': qty, 'buy_price': buy_price, 'highest': buy_price}
                            
                            trade_log.append({
                                'Date': date, 'Action': 'BUY', 'Ticker': c, 'Price': buy_price, 
                                'AI_Score': round(score, 2), 'Vol_Ratio': round(v_ratio, 1),
                                'RSI': round(s_rsi, 1), 'VN_MFI': round(market_mfi, 1)
                            })
                            empty -= 1
                        if empty == 0 or cash <= 0: break

        nav_history[i] = cash + sum([held[c]['qty'] * stocks_close.at[date, c] for c in held.keys()])

    return nav_history, pd.DataFrame(trade_log)

nav_history, df_trade_log = run_ai_quant_system_with_log()
backtest_result = df_clean[['Date', 'VNINDEX']].copy()
backtest_result['NAV_AI_Driven'] = nav_history

# ==========================================
# 5. MODULE BÁO CÁO TÍN HIỆU QUA TELEGRAM
# ==========================================
print("5. Đang trích xuất tín hiệu mới nhất và đẩy lên Telegram...")
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_msg(text_message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Bỏ qua gửi Telegram vì chưa có TOKEN. Log tin nhắn:\n")
        print(text_message)
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text_message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def process_and_send_signals(df_log):
    latest_date = stocks_close.index[-1]
    
    msg = f"🤖 <b>SIÊU RADAR VN30 - BÁO CÁO HÀNG NGÀY</b> 🤖\n"
    msg += f"📅 Ngày: <b>{latest_date.strftime('%d/%m/%Y')}</b>\n"
    msg += "➖➖➖➖➖➖➖➖➖➖\n\n"
    
    msg += "🎯 <b>LỆNH HÀNH ĐỘNG HÔM NAY:</b>\n"
    buy_count, sell_count = 0, 0
    
    if not df_log.empty:
        latest_trades = df_log[df_log['Date'] == df_log['Date'].max()]
        if not latest_trades.empty and df_log['Date'].max() == latest_date:
            for _, row in latest_trades.iterrows():
                ticker = row['Ticker']
                action = row['Action']
                price = row['Price']
                
                if 'BUY' in action:
                    ai_score = row.get('AI_Score', 0)
                    msg += f"🟢 <b>MUA ĐỘT PHÁ: {ticker}</b> (Giá: {price:,.0f})\n"
                    msg += f"   + AI dự báo Win: <b>{ai_score * 100:.1f}%</b>\n\n"
                    buy_count += 1
                elif 'SELL' in action:
                    profit = row.get('Profit(%)', 0)
                    icon = "🔥" if profit > 0 else "🩸"
                    msg += f"🔴 <b>BÁN TẤT TOÁN: {ticker}</b> (Lãi/Lỗ: {icon} {profit}%)\n\n"
                    sell_count += 1

    if buy_count == 0 and sell_count == 0:
        msg += "<i>⏳ Hệ thống nằm im quan sát, không có lệnh mới.</i>\n\n"

    msg += "➖➖➖➖➖➖➖➖➖➖\n"
    msg += "<i>⚠️ Khuyến nghị từ AI Random Forest.</i>"
    
    send_telegram_msg(msg)

process_and_send_signals(df_trade_log)

if not df_trade_log.empty:
    df_trade_log.to_csv('trading_log.csv', index=False)

print("6. Vẽ biểu đồ...")
try:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=backtest_result['Date'], y=backtest_result['VNINDEX'], name="VN-Index", mode='lines', line=dict(color='#B2B5BE', width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=backtest_result['Date'], y=backtest_result['NAV_AI_Driven'], name="NAV (AI Filter + Sizing)", mode='lines', line=dict(color='#00E676', width=2.5)), secondary_y=True)
    fig.update_layout(title='AI Quant System', template='plotly_dark')
    print("✅ Hoàn tất toàn bộ chu trình!")
except:
    pass
  # ==========================================
# 5. MÔ PHỎNG P&L: TOP-DOWN MACRO (MAX 10 SLOT + LỌC TRÙNG + LÃI KÉP + GỬI BANK TỰ ĐỘNG)
# ==========================================
print("\n" + "="*50)
print("🛡 MÔ PHỎNG P&L: QUẢN TRỊ DANH MỤC TOÀN DIỆN (CỔ PHIẾU + GỬI BANK 5%/NĂM)")
print("="*50)

if 'MA50' not in vnindex_df.columns:
    vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50).mean()

sim_data = test_data.copy()
sim_data['AI_Score'] = xgb_model.predict_proba(sim_data[ml_features])[:, 1]

# Chuẩn bị Bách khoa toàn thư Vĩ mô
vn_trend = vnindex_df[['Date', 'VNINDEX', 'MA50', 'VN_MFI']].copy()
vn_trend['Date'] = pd.to_datetime(vn_trend['Date']).dt.strftime('%Y-%m-%d')
macro_temp = macro_df[['Date', 'Z_DXY']].copy()
vn_trend = vn_trend.merge(macro_temp, on='Date', how='left')
vn_trend['Z_DXY'] = vn_trend['Z_DXY'].ffill()
vn_trend_idx = vn_trend.set_index('Date')

sim_data = sim_data.merge(vn_trend[['Date']], on='Date', how='left') 
buy_signals = sim_data[sim_data['AI_Score'] >= 0.75].copy()
buy_signals = buy_signals.sort_values('Date')

# 🟢 ĐIỀU CHỈNH LUẬT CHƠI
INITIAL_CAPITAL = 1000000000
MAX_SLOTS = 10
FEE_TAX = 0.003                         
BANK_RATE_YEARLY = 0.05                 # Lãi suất NH rảnh rỗi: 5%/năm
DAILY_RATE = BANK_RATE_YEARLY / 365     # Lãi suất theo ngày

# Ngày bắt đầu và kết thúc mô phỏng
sim_start_date = pd.to_datetime(test_data['Date'].min())
sim_end_date = pd.to_datetime(test_data['Date'].max())

# Quản lý 10 khe vốn (Thêm 'last_active_date' để tính ngày rảnh rỗi gửi NH)
slot_status = [{
    'free_date': sim_start_date, 
    'ticker': None, 
    'capital': INITIAL_CAPITAL / MAX_SLOTS,
    'last_active_date': sim_start_date 
} for _ in range(MAX_SLOTS)]

trade_logs = []
total_bank_interest = 0

for idx, row in buy_signals.iterrows():
    ticker = row['Ticker']
    buy_date = pd.to_datetime(row['Date'])
    buy_date_str = buy_date.strftime('%Y-%m-%d')
    
    # 1. KIỂM TRA VĨ MÔ
    try:
        vn_price = vn_trend_idx.loc[buy_date_str, 'VNINDEX']
        vn_ma50 = vn_trend_idx.loc[buy_date_str, 'MA50']
        z_dxy = vn_trend_idx.loc[buy_date_str, 'Z_DXY']
        vn_mfi = vn_trend_idx.loc[buy_date_str, 'VN_MFI']
    except: continue
    
    if (pd.notna(z_dxy) and z_dxy > 1.0) or \
       (pd.notna(vn_price) and pd.notna(vn_ma50) and vn_price < vn_ma50 and vn_mfi < 40):
        continue 

    # 2. KIỂM TRA TRÙNG LỆNH
    already_holding = False
    for slot in slot_status:
        if slot['ticker'] == ticker and slot['free_date'] > buy_date:
            already_holding = True
            break
    if already_holding: continue 

    # 3. TÌM SLOT TRỐNG VÀ GIẢI NGÂN
    try:
        available_slot_idx = -1
        for i in range(MAX_SLOTS):
            if slot_status[i]['free_date'] <= buy_date:
                available_slot_idx = i
                break 
                
        if available_slot_idx == -1: continue 
            
        # 🟢 TÍNH TIỀN LÃI NGÂN HÀNG TRONG THỜI GIAN CHỜ MUA (Idle Cash)
        idle_days = (buy_date - slot_status[available_slot_idx]['last_active_date']).days
        if idle_days > 0:
            idle_interest = slot_status[available_slot_idx]['capital'] * DAILY_RATE * idle_days
            slot_status[available_slot_idx]['capital'] += idle_interest
            total_bank_interest += idle_interest

        buy_idx = stocks_close.index.get_loc(buy_date_str)
        buy_price = stocks_close[ticker].iloc[buy_idx]
        
        sell_idx = -1
        sell_price = 0
        sell_reason = ""
        
        # Vòng lặp gồng lãi Macro
        for day_offset in range(1, 250):
            check_idx = buy_idx + day_offset
            if check_idx >= len(stocks_close):
                sell_idx = len(stocks_close) - 1
                sell_price = stocks_close[ticker].iloc[sell_idx]
                sell_reason = "HẾT DỮ LIỆU"
                break
                
            current_date_str = stocks_close.index[check_idx].strftime('%Y-%m-%d')
            current_price = stocks_close[ticker].iloc[check_idx]
            
            try:
                vn_today = vn_trend_idx.loc[current_date_str, 'VNINDEX']
                ma50_today = vn_trend_idx.loc[current_date_str, 'MA50']
                mfi_today = vn_trend_idx.loc[current_date_str, 'VN_MFI']
                dxy_today = vn_trend_idx.loc[current_date_str, 'Z_DXY']
            except:
                vn_today, ma50_today, mfi_today, dxy_today = 10000, 0, 50, 0 
                
            if pd.notna(vn_today) and pd.notna(ma50_today) and vn_today < ma50_today:
                if (pd.notna(mfi_today) and mfi_today < 40) or (pd.notna(dxy_today) and dxy_today > 1.0):
                    sell_idx = check_idx
                    sell_price = current_price
                    sell_reason = "VNI gãy trend + Vĩ mô xấu"
                    break
                
        if sell_idx == -1:
            sell_idx = buy_idx + 249
            if sell_idx >= len(stocks_close): sell_idx = len(stocks_close) - 1
            sell_price = stocks_close[ticker].iloc[sell_idx]
            sell_reason = "Hết hạn gồng (1 năm)"
            
        sell_date = stocks_close.index[sell_idx]
        
        # CẬP NHẬT TIỀN TỪ CỔ PHIẾU
        profit_pct = (sell_price / buy_price) - 1 - FEE_TAX
        current_capital = slot_status[available_slot_idx]['capital'] 
        trade_profit_vnd = current_capital * profit_pct
        new_capital = current_capital + trade_profit_vnd 
        
        # TIỀN NHẢ VỀ LẠI THÀNH TIỀN MẶT CHỜ GỬI BANK
        slot_status[available_slot_idx] = {
            'free_date': pd.to_datetime(sell_date),
            'ticker': ticker,
            'capital': new_capital,
            'last_active_date': pd.to_datetime(sell_date) # Bắt đầu đếm rảnh rỗi từ ngày bán
        }
        
        trade_logs.append({
            'Buy_Date': buy_date_str,
            'Sell_Date': sell_date.strftime('%Y-%m-%d'),
            'Ticker': ticker,
            'Reason': sell_reason,
            'Stock_Profit_%': round(profit_pct * 100, 2),
            'Stock_Profit_VND': trade_profit_vnd,
            'Slot_Capital': round(new_capital, 0)
        })
    except:
        continue

# 🟢 CHỐT SỔ CUỐI CÙNG: CỘNG LÃI BANK CHO CÁC NGÀY RẢNH RỖI CÒN LẠI ĐẾN HIỆN TẠI
for slot in slot_status:
    remaining_idle_days = (sim_end_date - slot['last_active_date']).days
    if remaining_idle_days > 0:
        final_interest = slot['capital'] * DAILY_RATE * remaining_idle_days
        slot['capital'] += final_interest
        total_bank_interest += final_interest

if len(trade_logs) > 0:
    df_trades = pd.DataFrame(trade_logs)
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['Stock_Profit_VND'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    final_capital = sum(slot['capital'] for slot in slot_status)
    total_profit = final_capital - INITIAL_CAPITAL
    roi = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    print(f"🎯 Số lệnh ĐƯỢC MUA (Thực tế): {total_trades} lệnh")
    print(f"🏆 Tỷ lệ thắng lệnh cổ phiếu (Win-rate): {win_rate:.1f}%")
    print("-" * 50)
    print(f"💵 VỐN BAN ĐẦU       : {INITIAL_CAPITAL:,.0f} VNĐ")
    print(f"🏦 LÃI TỪ GỬI BANK 5%: +{total_bank_interest:,.0f} VNĐ (Tiền đẻ ra tiền khi rảnh)")
    print(f"📈 TỔNG LÃI/LỖ RÒNG  : {total_profit:,.0f} VNĐ (Bao gồm cả Bank + Cổ phiếu)")
    print(f"💎 TỔNG TÀI SẢN (NAV): {final_capital:,.0f} VNĐ")
    print(f"🚀 TỶ SUẤT LỢI NHUẬN (ROI TOÀN DIỆN): {roi:.2f}%")
    
    print("\n🔍 X-RAY 10 LỆNH GẦN NHẤT (DANH MỤC 10 SLOT):")
    print(df_trades[['Buy_Date', 'Sell_Date', 'Ticker', 'Reason', 'Stock_Profit_%', 'Slot_Capital']].tail(10).to_string(index=False))
else:
    print("⚠️ Không có lệnh nào khớp và chốt lời thành công.")
    # 1. Bỏ giới hạn hiển thị của Pandas để in toàn bộ các dòng ra màn hình
import pandas as pd
pd.set_option('display.max_rows', None) 
print(df_trades[['Buy_Date', 'Sell_Date', 'Ticker', 'Reason', 'Stock_Profit_%', 'Slot_Capital']])

# 2. Xuất toàn bộ sổ lệnh ra file CSV để mở bằng Excel
df_trades.to_csv('Lich_Su_Lenh_Toan_Tap_2024.csv', index=False, encoding='utf-8-sig')
print("✅ Đã xuất file Lich_Su_Lenh_Toan_Tap_2024.csv thành công!")
