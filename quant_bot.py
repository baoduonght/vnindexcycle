
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tvDatafeed import TvDatafeed, Interval
from sklearn.ensemble import RandomForestClassifier
import warnings
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

# ==========================================
# 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG
# ==========================================
print("1. Đang tải Market Proxy (VNINDEX) và tính toán Dòng tiền MFI...")
vnindex_df = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=4000).reset_index()
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

# THÊM MỚI: Tính RSI sẵn cho toàn bộ cổ phiếu để trích xuất lúc báo lệnh
stocks_rsi = pd.DataFrame(index=stocks_close.index)
for c in stock_tickers:
    stocks_rsi[c] = calc_rsi(stocks_close[c])

import sys

# ==========================================
# 3. NHÚNG AI - HUẤN LUYỆN & CHẤM ĐIỂM (MÀNG LỌC KÉP VN-US)
# ==========================================
print("3. Tải dữ liệu & Đào tạo AI (Chỉ chạy khi 100% ĐỒNG BỘ Data tươi từ VN và US)...")

# 1. ĐỌC VÀ CHỐT LỊCH GIAO DỊCH MỸ (US)
try:
    macro_df = pd.read_csv('global_markets_database10years.csv')
    macro_df = macro_df[['Date', 'DXY', 'US10Y']]
    
    macro_df['DXY'] = pd.to_numeric(macro_df['DXY'], errors='coerce')
    macro_df['US10Y'] = pd.to_numeric(macro_df['US10Y'], errors='coerce')
    macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # 🟢 MÀNG LỌC 1: Lấy danh sách các ngày thị trường Mỹ THỰC SỰ GIAO DỊCH
    fresh_us_dates = macro_df.dropna(subset=['DXY', 'US10Y'])['Date'].tolist()
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

# 2. XỬ LÝ TOÁN HỌC (Z-SCORE)
mfi_data = vnindex_df[['Date', 'VN_MFI']].copy()
mfi_data['Date'] = pd.to_datetime(mfi_data['Date']).dt.strftime('%Y-%m-%d')

macro_df = macro_df.merge(mfi_data, on='Date', how='left')
macro_df['VN_MFI'] = macro_df['VN_MFI'].ffill()

macro_cols = ['DXY', 'US10Y']
for col in macro_cols:
    # Ffill tạm thời HẬU TRƯỜNG để hàm rolling có số liệu tính toán trung bình
    macro_df[col] = macro_df[col].ffill()
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()
z_cols = [f'Z_{c}' for c in macro_cols]

# 3. TẠO FEATURES CỔ PHIẾU (VN)
df_ml_list = []
clean_dates = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')

for c in stock_tickers:
    df_c = pd.DataFrame({'Date': clean_dates})
    df_c['Ticker'] = c
    df_c['Vol_Ratio'] = (stocks_volume[f'{c}_V'] / stocks_vol_ma20[f'{c}_V']).values
    df_c['Dist_MA50'] = (stocks_close[c] / stocks_ma50[c]).values
    df_c['RSI'] = calc_rsi(stocks_close[c]).values
    
    future_return = stocks_close[c].shift(-10) / stocks_close[c] - 1
    df_c['Target'] = (future_return > 0.02).astype(int).values
    df_ml_list.append(df_c)

concat_df = pd.concat(df_ml_list)
ml_master = concat_df.merge(macro_df, on='Date', how='left')
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'VN_MFI'] + z_cols

# ==========================================
# 🟢 KÍCH HOẠT MÀNG LỌC KÉP TRƯỚC KHI CHO AI HỌC
# ==========================================
# 1. Đá văng những ngày Mỹ nghỉ lễ (Thiếu Data US)
ml_master = ml_master[ml_master['Date'].isin(fresh_us_dates)]

# 2. Đá văng những ngày API Việt Nam bị lỗi, trả về NaN ở Volume/RSI (Thiếu Data VN)
# Chỉ giữ lại bảng dữ liệu "Siêu Sạch" (Hoàn hảo 100% chỉ báo của CẢ 2 THỊ TRƯỜNG)
ml_master_clean = ml_master.dropna(subset=ml_features).copy()

if ml_master_clean.empty:
    sys.exit("⚠️ LỖI ĐỒNG BỘ: Cả Mỹ và VN đều không đủ Data sạch. Bot buộc phải đứng ngoài!")

# 4. HUẤN LUYỆN AI (HỌC TỪ DATA THẬT 100%)
# Dùng bảng Siêu Sạch để học, chỉ cần lọc thêm Target bị khuyết (do T+10 của ngày hiện tại)
train_data = ml_master_clean.dropna(subset=['Target'])
train_data = train_data[train_data['Date'] < '2024-01-01']

X_train = train_data[ml_features]
y_train = train_data['Target']

if len(X_train) == 0:
    sys.exit("❌ LỖI TRÍ MẠNG: Tập X_train rỗng. Hãy kiểm tra lại API TradingView và CSV.")

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
print(f"✅ AI đã học xong từ {len(X_train)} dòng dữ liệu ĐỒNG BỘ CHUẨN MỰC của cả VN và US!")

# 5. XUẤT TÍN HIỆU
# Chỉ chấm điểm AI cho những ngày vượt qua được màng lọc kép
ml_master_clean['AI_Score'] = rf_model.predict_proba(ml_master_clean[ml_features])[:, 1]

# Ghép điểm AI trở lại bảng gốc. 
# Nếu mã cổ phiếu nào hôm nay bị khuyết VN Data hoặc US Data -> Nhận giá trị NaN -> KHÔNG MUA BÁN.
ml_master = ml_master.merge(ml_master_clean[['Date', 'Ticker', 'AI_Score']], on=['Date', 'Ticker'], how='left')
ai_dict = {c: ml_master[ml_master['Ticker'] == c].set_index('Date')['AI_Score'] for c in stock_tickers}
# ==========================================
# 4. HUẤN LUYỆN AI (HỌC TỪ DỮ LIỆU SẠCH 100%)
# ==========================================
# Lọc bỏ mọi dòng khuyết Feature hoặc Target để đảm bảo AI học chuẩn
train_data = ml_master.dropna(subset=ml_features + ['Target'])
train_data = train_data[train_data['Date'] < '2024-01-01']

X_train = train_data[ml_features]
y_train = train_data['Target']

if len(X_train) == 0:
    missing_report = ml_master[ml_features].isna().sum()
    print(f"\n🔍 BÁO CÁO CỘT LỖI GÂY TRỐNG DỮ LIỆU:\n{missing_report[missing_report > 0]}")
    sys.exit("❌ LỖI TRÍ MẠNG: X_train rỗng. Dữ liệu Vĩ mô hoặc Volume cổ phiếu đang bị khuyết hoàn toàn!")

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)

print("Đang nạp dữ liệu vào mạng nơ-ron Random Forest...")
rf_model.fit(X_train, y_train)
print(f"✅ AI đã huấn luyện xong trên {len(X_train)} dòng dữ liệu chuẩn!")

# ==========================================
# 5. DỰ ĐOÁN & XUẤT TÍN HIỆU (KHÔNG VAY MƯỢN DỮ LIỆU)
# ==========================================
# CHỈ LẤY những dòng hội tụ đủ 100% các biến số (cả VN và Vĩ mô)
ml_master_predict = ml_master.dropna(subset=ml_features).copy()

if ml_master_predict.empty:
    sys.exit("⚠️ Dữ liệu trống trơn. Không thể chạy dự đoán.")

# Chấm điểm AI dựa trên tập dữ liệu hoàn hảo
ml_master_predict['AI_Score'] = rf_model.predict_proba(ml_master_predict[ml_features])[:, 1]

# Ghép điểm AI_Score trả lại bảng ml_master gốc (các ngày khuyết dữ liệu sẽ tự động mang giá trị NaN)
ml_master = ml_master.merge(ml_master_predict[['Date', 'Ticker', 'AI_Score']], on=['Date', 'Ticker'], how='left')

# Trích xuất sang Bước 4. Nếu hôm nay thiếu Vĩ mô -> AI_Score = NaN -> Bot im lặng đứng ngoài thị trường
ai_dict = {c: ml_master[ml_master['Ticker'] == c].set_index('Date')['AI_Score'] for c in stock_tickers}
# ==========================================
# 4. CHẠY BACKTEST (TÍCH HỢP AI CHẤM ĐIỂM)
# ==========================================
print("4. Chạy Sổ Kế Toán & Trích xuất Lịch sử Mua/Bán của AI...")
def run_ai_quant_system_with_log():
    # THÊM 'VN_MFI' vào danh sách backtest để lấy dữ liệu dòng tiền
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
        market_mfi = backtest['VN_MFI'].iloc[i] # Lấy MFI của phiên hiện tại

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
                            if ai_score >= 0.55:
                                # Tính toán độ đột biến Vol và lấy RSI
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
                            
                            # GHI ĐẦY ĐỦ CÁC CHỈ BÁO VÀO SỔ LỆNH ĐỂ TELEGRAM ĐỌC
                            trade_log.append({
                                'Date': date, 'Action': 'BUY', 'Ticker': c, 'Price': buy_price, 
                                'AI_Score': round(score, 2), 'Vol_Ratio': round(v_ratio, 1),
                                'RSI': round(s_rsi, 1), 'VN_MFI': round(market_mfi, 1)
                            })
                            empty -= 1
                        if empty == 0 or cash <= 0: break

        nav_history[i] = cash + sum([held[c]['qty'] * stocks_close.at[date, c] for c in held.keys()])

    return nav_history, pd.DataFrame(trade_log)
# ==========================================
# 5. MODULE BÁO CÁO TÍN HIỆU QUA TELEGRAM
# ==========================================
print("5. Đang trích xuất tín hiệu mới nhất và đẩy lên Telegram...")

import os
import requests

# Lấy Token từ GitHub Secrets
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_msg(text_message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text_message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print("✅ Đã bắn tín hiệu Telegram thành công!")
        else:
            print(f"❌ Lỗi Telegram API: {res.text}")
    except Exception as e:
        print(f"❌ Lỗi kết nối mạng khi gửi Telegram: {e}")

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
                    vol_ratio = row.get('Vol_Ratio', 0)
                    rsi = row.get('RSI', 0)
                    mfi = row.get('VN_MFI', 0)
                    
                    msg += f"🟢 <b>MUA ĐỘT PHÁ: {ticker}</b> (Giá: {price:,.0f})\n"
                    msg += f"   + Nổ Volume: <b>{vol_ratio}x</b> trung bình\n"
                    msg += f"   + Sức mạnh RSI: <b>{rsi}</b>\n"
                    msg += f"   + Dòng tiền Vĩ mô (MFI): <b>{mfi}</b>\n"
                    msg += f"   + AI dự báo Win: <b>{ai_score * 100:.1f}%</b>\n\n"
                    buy_count += 1
                    
                elif 'SELL' in action:
                    profit = row.get('Profit(%)', 0)
                    icon = "🔥" if profit > 0 else "🩸"
                    msg += f"🔴 <b>BÁN TẤT TOÁN: {ticker}</b> (Lãi/Lỗ: {icon} {profit}%)\n\n"
                    sell_count += 1

    if buy_count == 0 and sell_count == 0:
        msg += "<i>⏳ Hệ thống nằm im quan sát, không có lệnh mới.</i>\n\n"

    # CHỈ HIỂN THỊ TOP 5 KHI CÓ LỆNH MUA TỪ HỆ THỐNG
    if buy_count > 0:
        msg += "➖➖➖➖➖➖➖➖➖➖\n"
        msg += "🌟 <b>TOP 5 CP VN30 ĐÁNG MUA NHẤT (Tham khảo thêm):</b>\n"
        msg += "<i>(Đang trong Uptrend & Điểm AI cao nhất)</i>\n\n"
        
        top_stocks = []
        for c in stock_tickers:
            curr_price = stocks_close.at[latest_date, c]
            ma50 = stocks_ma50.at[latest_date, c]
            ai_score = ai_dict[c].get(latest_date, 0)
            
            # Chỉ lọc các mã có giá nằm trên MA50
            if curr_price > ma50:
                top_stocks.append({'Ticker': c, 'AI_Score': ai_score, 'Price': curr_price})
                
        # Xếp hạng lấy 5 mã điểm cao nhất
        top_5 = sorted(top_stocks, key=lambda x: x['AI_Score'], reverse=True)[:5]
        
        if top_5:
            for i, item in enumerate(top_5, 1):
                msg += f"{i}. <b>{item['Ticker']}</b> - Giá: {item['Price']:,.0f} VND (AI: {item['AI_Score']*100:.1f}%)\n"
        else:
            msg += "⚠️ <i>Không có mã nào giữ được Trend tăng ngoài danh mục bot báo.</i>\n"
            
        msg += "\n"
        
    # CHỐT LẠI BẰNG FOOTER
    msg += "➖➖➖➖➖➖➖➖➖➖\n"
    msg += "<i>⚠️ Khuyến nghị từ AI Random Forest. Hãy chia đều tỷ trọng.</i>"
    
    # 🚀 CHỈ GỌI HÀM GỬI ĐÚNG 1 LẦN SAU KHI ĐÃ GÓI XONG TIN NHẮN
    # Nếu bạn đang dùng code xuất ảnh, hãy đổi thành: send_telegram_photo(chart_path, msg)
    send_telegram_msg(msg)

# Chạy báo cáo
process_and_send_signals(df_trade_log)

# Lưu sổ lệnh ra file CSV (Dành cho GitHub Actions)
if not df_trade_log.empty:
    df_trade_log.to_csv('trading_log.csv', index=False)

print("6. Vẽ biểu đồ...")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=backtest_result['Date'], y=backtest_result['VNINDEX'], name="VN-Index", mode='lines', line=dict(color='#B2B5BE', width=1)), secondary_y=False)
fig.add_trace(go.Scatter(x=backtest_result['Date'], y=backtest_result['NAV_AI_Driven'], name="NAV (AI Filter + Volatility Sizing)", mode='lines', line=dict(color='#00E676', width=2.5)), secondary_y=True)
fig.update_layout(title='Đấu trường VN-Index: Hệ Thống AI Nhận Diện Breakout Giả (Có MFI)', template='plotly_dark')
fig.show()
