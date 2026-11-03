import pandas as pd
import numpy as np
import yfinance as yf
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
df_clean = vnindex_df.dropna().copy()

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

stocks_rsi = pd.DataFrame(index=stocks_close.index)
for c in stock_tickers:
    stocks_rsi[c] = calc_rsi(stocks_close[c])

# ==========================================
# 3. TẢI NÃO BỘ AI & CHẤM ĐIỂM (CẬP NHẬT TỶ GIÁ & ĐỘNG LƯỢNG)
# ==========================================
print("3. Đang tải bộ nhớ AI tĩnh & Đồng bộ Vĩ mô CSV...")
try:
    macro_df = pd.read_csv('markets_database10years.csv')
    macro_df = macro_df[['Date', 'DXY', 'US10Y', 'USD_VND']]
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)
except Exception as e:
    sys.exit(f"🛑 LỖI ĐỌC FILE VĨ MÔ: {e}")

market_context = vnindex_df[['Date', 'VN_MFI', 'VNINDEX', 'MA200']].copy()
market_context['VN_Dist_MA200'] = market_context['VNINDEX'] / market_context['MA200']
market_context['Date'] = pd.to_datetime(market_context['Date']).dt.strftime('%Y-%m-%d')

macro_df = macro_df.merge(market_context, on='Date', how='left').ffill()
macro_cols = ['DXY', 'US10Y', 'USD_VND']
for col in macro_cols:
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()

df_ml_list = []
clean_dates = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')
for c in stock_tickers:
    df_c = pd.DataFrame({'Date': clean_dates, 'Ticker': c})
    df_c['Vol_Ratio'] = (stocks_volume[f'{c}_V'] / stocks_vol_ma20[f'{c}_V']).values
    df_c['Dist_MA50'] = (stocks_close[c] / stocks_ma50[c]).values
    df_c['RSI'] = calc_rsi(stocks_close[c]).values
    df_c['RSI_Momentum'] = pd.Series(df_c['RSI']).diff(3).values # Động lượng RSI
    df_c['MACD_Hist'] = calc_macd_hist(stocks_close[c]).values
    df_ml_list.append(df_c)

ml_master = pd.concat(df_ml_list).merge(macro_df, on='Date', how='left')
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200', 'Z_DXY', 'Z_US10Y', 'Z_USD_VND']

MODEL_FILE = "xgboost_ultimate.json"
xgb_model = xgb.XGBClassifier()

if os.path.exists(MODEL_FILE):
    xgb_model.load_model(MODEL_FILE)
else:
    sys.exit(f"⚠️ Không tìm thấy {MODEL_FILE}! Bác phải chạy file train_bot.py trước.")

ml_master['AI_Score'] = float('nan')
valid_rows = ml_master[ml_features].notna().all(axis=1)
if valid_rows.sum() > 0:
    ml_master.loc[valid_rows, 'AI_Score'] = xgb_model.predict_proba(ml_master.loc[valid_rows, ml_features])[:, 1]
ai_dict = {c: ml_master[ml_master['Ticker'] == c].set_index('Date')['AI_Score'] for c in stock_tickers}

# ==========================================
# 4. CHẠY BACKTEST KẾ TOÁN (10 SLOTS + LÃI BANK)
# ==========================================
print("4. Chạy Sổ Kế Toán (Diamond Hands)...")
def run_ai_quant_system_with_log():
    backtest = df_clean.copy()
    MAX_SLOTS = 10
    LIQUIDITY_FEE = 0.003
    DAILY_INTEREST = 0.05 / 252 
    cash = 100_000_000 
    held = {}
    trade_log = [] 

    for i in range(len(backtest)):
        date = backtest['Date'].iloc[i]
        idx = backtest['VNINDEX'].iloc[i]
        market_ma50, market_ma100, market_ma200 = backtest['MA50'].iloc[i], backtest['MA100'].iloc[i], backtest['MA200'].iloc[i]
        market_adx, market_atr_pct, market_mfi = backtest['ADX'].iloc[i], backtest['ATR_PCT'].iloc[i], backtest['VN_MFI'].iloc[i]

        if date not in stocks_close.index: continue
        cash *= (1 + DAILY_INTEREST)

        # 1. CÔNG TẮC VĨ MÔ: Bán tháo toàn bộ khi VNINDEX gãy MA100
        if idx < market_ma100:
            for c in list(held.keys()):
                sell_price = stocks_close.at[date, c]
                cash += held[c]['qty'] * sell_price * (1 - LIQUIDITY_FEE)
                trade_log.append({'Date': date, 'Action': 'SELL (Cắt Macro)', 'Ticker': c, 'Price': sell_price, 'Profit(%)': round((sell_price/held[c]['buy_price']-1)*100, 2)})
            held.clear()
        else:
            to_remove = []
            for c in held.keys():
                curr_price = stocks_close.at[date, c]
                buy_price = held[c]['buy_price']
                held[c]['highest'] = max(held[c]['highest'], curr_price)
                
                profit_margin = (curr_price - buy_price) / buy_price
                peak_margin = (held[c]['highest'] - buy_price) / buy_price

                # 2. XÓA BỎ CẮT LỖ VÀ CHỐT LỜI. CHỈ BÁN KHI MẤT ĐÀ TĂNG TRƯỞNG
                if peak_margin >= 0.12 and curr_price < held[c]['highest'] * 0.94:
                    to_remove.append(c) # Trailing Stop: Lãi >12% nhưng rớt 6% từ đỉnh thì chốt thu tiền về
                elif curr_price < stocks_ma20.at[date, c] and profit_margin > 0.05:
                    to_remove.append(c) # Chặn lãi: Thủng MA20 ngắn hạn khi đang có lãi nhẹ thì nhả hàng

            for c in to_remove:
                sell_price = stocks_close.at[date, c]
                cash += held[c]['qty'] * sell_price * (1 - LIQUIDITY_FEE)
                trade_log.append({'Date': date, 'Action': 'SELL', 'Ticker': c, 'Price': sell_price, 'Profit(%)': round((sell_price/held[c]['buy_price']-1)*100, 2)})
                del held[c]

            max_exposure = 0.3 if (idx < market_ma50 or market_atr_pct > 0.018) else 1.0
            current_total_nav = cash + sum([held[c]['qty'] * stocks_close.at[date, c] for c in held.keys()])
            allowed_capital = (current_total_nav * max_exposure) - sum([held[c]['qty'] * stocks_close.at[date, c] for c in held.keys()])

            empty = MAX_SLOTS - len(held)
            if empty > 0 and (idx > market_ma200) and (market_adx > 22) and (allowed_capital > 0):
                buy_candidates = []
                for c in stock_tickers:
                    if c not in held:
                        curr_price, curr_vol = stocks_close.at[date, c], stocks_volume.at[date, f'{c}_V']
                        prev_high_20, prev_vol_ma20 = stocks_high_20.at[date, c], stocks_vol_ma20.at[date, f'{c}_V']

                        if pd.isna(prev_high_20) or pd.isna(prev_vol_ma20): continue
                        if curr_price > prev_high_20 and curr_vol > (prev_vol_ma20 * 1.5):
                            ai_score = ai_dict[c].get(date, 0)
                            if pd.notna(ai_score) and ai_score >= 0.72: # Chuẩn Sniper tối ưu
                                buy_candidates.append((c, ai_score, curr_vol/prev_vol_ma20 if prev_vol_ma20>0 else 0, stocks_rsi.at[date, c]))

                if buy_candidates:
                    invest_amt = min(allowed_capital / empty, cash)
                    for c, score, v_ratio, s_rsi in buy_candidates:
                        if invest_amt > 0 and stocks_close.at[date, c] > 0 and cash >= invest_amt:
                            cash -= invest_amt
                            buy_price = stocks_close.at[date, c]
                            held[c] = {'qty': (invest_amt * (1 - LIQUIDITY_FEE)) / buy_price, 'buy_price': buy_price, 'highest': buy_price}
                            trade_log.append({'Date': date, 'Action': 'BUY', 'Ticker': c, 'Price': buy_price, 'AI_Score': round(score, 2), 'Vol_Ratio': round(v_ratio, 1), 'RSI': round(s_rsi, 1), 'VN_MFI': round(market_mfi, 1)})
                            empty -= 1
                        if empty == 0 or cash <= 0: break
    return pd.DataFrame(trade_log)

df_trade_log = run_ai_quant_system_with_log()
# ==========================================
# IN KẾT QUẢ BACKTEST RA MÀN HÌNH CONSOLE
# ==========================================
print("\n" + "="*50)
print("📊 TỔNG KẾT BACKTEST HIỆU QUẢ QUÁ KHỨ")
print("="*50)

if not df_trade_log.empty and 'Action' in df_trade_log.columns:
    sells = df_trade_log[df_trade_log['Action'].str.contains('SELL', na=False)]
    total_sells = len(sells)
    win_sells = len(sells[sells['Profit(%)'] > 0])

    print(f"Tổng số lệnh đã chốt sổ: {total_sells}")
    if total_sells > 0:
        print(f"🎯 Tỷ lệ thắng (Win Rate): {win_sells / total_sells * 100:.2f}%")
        print(f"💰 Lệnh lãi cao nhất: +{sells['Profit(%)'].max()}%")
        print(f"🩸 Lệnh lỗ nặng nhất: {sells['Profit(%)'].min()}%")
    else:
        print("⏳ Bot đang ôm hàng, chưa chốt sổ lệnh BÁN nào.")
else:
    print("🤖 Tiêu chuẩn quá khắt khe, chưa có lệnh MUA/BÁN nào được khớp.")
print("="*50 + "\n")
# ==========================================
# 5. TELEGRAM RADAR
# ==========================================
print("5. Báo cáo Telegram...")
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_msg(text_message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text_message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text_message, "parse_mode": "HTML"}, timeout=10)
    except: pass

latest_date = stocks_close.index[-1]
msg = f"🤖 <b>SIÊU RADAR VN30 - BÁO CÁO HÀNG NGÀY</b> 🤖\n📅 Ngày: <b>{latest_date.strftime('%d/%m/%Y')}</b>\n➖➖➖➖➖➖➖➖➖➖\n\n🎯 <b>LỆNH HÔM NAY:</b>\n"
buy_count, sell_count = 0, 0

if not df_trade_log.empty:
    latest_trades = df_trade_log[df_trade_log['Date'] == df_trade_log['Date'].max()]
    if not latest_trades.empty and df_trade_log['Date'].max() == latest_date:
        for _, row in latest_trades.iterrows():
            if 'BUY' in row['Action']:
                msg += f"🟢 <b>MUA: {row['Ticker']}</b> (Giá: {row['Price']:,.0f}) | AI: <b>{row['AI_Score'] * 100:.1f}%</b>\n\n"
                buy_count += 1
            elif 'SELL' in row['Action']:
                icon = "🔥" if row['Profit(%)'] > 0 else "🩸"
                msg += f"🔴 <b>BÁN: {row['Ticker']}</b> (Lãi/Lỗ: {icon} {row['Profit(%)']}%)\n\n"
                sell_count += 1

if buy_count == 0 and sell_count == 0:
    msg += "<i>⏳ Đứng ngoài quan sát (Đang nhận lãi Bank 5%).</i>\n\n"
msg += "➖➖➖➖➖➖➖➖➖➖"
send_telegram_msg(msg)
