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
# 5. MÔ PHỎNG THỰC CHIẾN 100 TRIỆU TỪ 2024
# ==========================================
print("\n" + "="*50)
print("💰 MÔ PHỎNG ĐÁNH THỰC CHIẾN VỚI 100 TRIỆU (TỪ 01/2024)")
print("="*50)

# 1. Lấy lại các lệnh báo MUA từ hệ thống Sniper
sim_data = test_data.copy()
sim_data['AI_Score'] = xgb_model.predict_proba(sim_data[ml_features])[:, 1]
buy_signals = sim_data[sim_data['AI_Score'] >= 0.75].copy()

# 2. Cài đặt luật chơi
INITIAL_CAPITAL = 100_000_000  # Vốn khởi điểm: 100 củ
SLOT_SIZE = 20_000_000         # Đánh 20 củ / 1 lệnh
HOLD_DAYS = 15                 # CHÚ Ý: Bác tự sửa số 15 này khớp với số T+ bác cài ở trên nhé!
FEE_TAX = 0.003                # Phí giao dịch + Thuế (0.3% / vòng)

trade_logs = []

for idx, row in buy_signals.iterrows():
    ticker = row['Ticker']
    buy_date = row['Date']
    
    try:
        buy_idx = stocks_close.index.get_loc(buy_date)
        
        # Chỉ tính những lệnh đã đủ thời gian Hold (đã chốt xong)
        if buy_idx + HOLD_DAYS < len(stocks_close):
            buy_price = stocks_close[ticker].iloc[buy_idx]
            sell_price = stocks_close[ticker].iloc[buy_idx + HOLD_DAYS]
            sell_date = stocks_close.index[buy_idx + HOLD_DAYS].strftime('%Y-%m-%d')
            
            # Tính % lợi nhuận (đã trừ phí thuế)
            profit_pct = (sell_price / buy_price) - 1 - FEE_TAX
            
            # Quy ra tiền thật
            profit_vnd = SLOT_SIZE * profit_pct
            
            trade_logs.append({
                'Buy_Date': buy_date,
                'Sell_Date': sell_date,
                'Ticker': ticker,
                'Buy_Pr': round(buy_price, 2),
                'Sell_Pr': round(sell_price, 2),
                'Profit_%': round(profit_pct * 100, 2),
                'Profit_VND': profit_vnd
            })
    except:
        continue

# ==========================================
# 5. MÔ PHỎNG P&L: BỘ LỌC ĐA CHỈ BÁO (SMART RADAR)
# ==========================================
print("\n" + "="*50)
print("🛡 MÔ PHỎNG THỰC CHIẾN TỪ 01/2024: RADAR ĐA CHỈ BÁO")
print("="*50)

if 'MA50' not in vnindex_df.columns:
    vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50).mean()

sim_data = test_data.copy()
sim_data['AI_Score'] = xgb_model.predict_proba(sim_data[ml_features])[:, 1]

vn_trend = vnindex_df[['Date', 'VNINDEX', 'MA50']].copy()
vn_trend['Date'] = pd.to_datetime(vn_trend['Date']).dt.strftime('%Y-%m-%d')
sim_data = sim_data.merge(vn_trend, on='Date', how='left')

buy_signals = sim_data[sim_data['AI_Score'] >= 0.75].copy()
buy_signals = buy_signals.sort_values('Date')

INITIAL_CAPITAL = 100_000_000
MAX_SLOTS = 5
SLOT_SIZE = INITIAL_CAPITAL / MAX_SLOTS 
HOLD_DAYS = 15                          
FEE_TAX = 0.003                         

capital_slots = [pd.Timestamp('2000-01-01')] * MAX_SLOTS
trade_logs = []
blocked_count = 0

for idx, row in buy_signals.iterrows():
    ticker = row['Ticker']
    buy_date = pd.to_datetime(row['Date'])
    
    # 🟢 LẤY ĐẦY ĐỦ CÁC CHỈ BÁO TỪ PHẦN 3
    vn_price = row.get('VNINDEX', 0)
    vn_ma50 = row.get('MA50', 0)
    z_dxy = row.get('Z_DXY', 0)
    vn_mfi = row.get('VN_MFI', 0)
    rsi = row.get('RSI', 0)
    macd = row.get('MACD_Hist', 0)
    vol_ratio = row.get('Vol_Ratio', 0)
    
    # 🔴 BỘ LỌC ĐA CHIỀU (MASTER SWITCH 2.0)
    # Từ chối mua nếu: DXY quá nóng (>1.0) HOẶC (VNINDEX gãy MA50 VÀ Dòng tiền MFI cạn kiệt < 40)
    if (pd.notna(z_dxy) and z_dxy > 1.0) or \
       (pd.notna(vn_price) and pd.notna(vn_ma50) and vn_price < vn_ma50 and vn_mfi < 40):
        blocked_count += 1
        continue 

    try:
        available_slot_idx = -1
        for i in range(MAX_SLOTS):
            if capital_slots[i] <= buy_date:
                available_slot_idx = i
                break 
                
        if available_slot_idx == -1:
            continue 
            
        buy_idx = stocks_close.index.get_loc(buy_date.strftime('%Y-%m-%d'))
        
        if buy_idx + HOLD_DAYS < len(stocks_close):
            buy_price = stocks_close[ticker].iloc[buy_idx]
            sell_price = stocks_close[ticker].iloc[buy_idx + HOLD_DAYS]
            sell_date = stocks_close.index[buy_idx + HOLD_DAYS]
            
            capital_slots[available_slot_idx] = pd.to_datetime(sell_date)
            
            profit_pct = (sell_price / buy_price) - 1 - FEE_TAX
            profit_vnd = SLOT_SIZE * profit_pct
            
            trade_logs.append({
                'Buy_Date': buy_date.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'VN_MFI': round(vn_mfi, 1),
                'RSI': round(rsi, 1),
                'MACD': round(macd, 3),
                'Vol_Ratio': round(vol_ratio, 2),
                'Profit_%': round(profit_pct * 100, 2),
                'Profit_VND': profit_vnd
            })
    except:
        continue

if len(trade_logs) > 0:
    df_trades = pd.DataFrame(trade_logs)
    
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['Profit_VND'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    total_profit = df_trades['Profit_VND'].sum()
    final_capital = INITIAL_CAPITAL + total_profit
    roi = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    print(f"🛑 Số lệnh bị lọc bỏ (Vĩ mô xấu + Cạn dòng tiền): {blocked_count} lệnh")
    print(f"🎯 Số lệnh ĐƯỢC MUA (Thỏa mãn Radar): {total_trades} lệnh")
    print(f"🏆 Tỷ lệ thắng (Win-rate): {win_rate:.1f}%")
    print("-" * 50)
    print(f"💵 VỐN BAN ĐẦU       : {INITIAL_CAPITAL:,.0f} VNĐ")
    print(f"📈 TỔNG LÃI/LỖ RÒNG  : {total_profit:,.0f} VNĐ")
    print(f"🚀 TỶ SUẤT LỢI NHUẬN (ROI): {roi:.2f}%")
    
    print("\n🔍 X-RAY 10 LỆNH GẦN NHẤT ĐƯỢC AI BÁO MUA:")
    print(df_trades[['Buy_Date', 'Ticker', 'VN_MFI', 'RSI', 'MACD', 'Vol_Ratio', 'Profit_%']].tail(10).to_string(index=False))
else:
    print("⚠️ Không có lệnh nào qua được bộ lọc Đa chỉ báo.")
