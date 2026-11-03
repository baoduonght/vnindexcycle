import pandas as pd
import numpy as np
import requests
import warnings
import sys
import os
from tvDatafeed import TvDatafeed, Interval
import xgboost as xgb

warnings.filterwarnings('ignore')
tv = TvDatafeed()

# ==========================================
# HÀM TÍNH TOÁN CHỈ BÁO NỘI BỘ
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
# 1. TẢI VN-INDEX & BỘ LỌC THỊ TRƯỜNG CHUNG
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

stocks_close = stocks_close.ffill().dropna()
stocks_volume = stocks_volume.ffill().fillna(0)
stock_tickers = [c for c in valid_tickers if c in stocks_close.columns]

stocks_high_20 = stocks_close.rolling(20).max().shift(1)
stocks_ma20 = stocks_close.rolling(20).mean()
stocks_ma50 = stocks_close.rolling(50).mean()
stocks_vol_ma20 = stocks_volume.rolling(20).mean().shift(1)

stocks_rsi = pd.DataFrame(index=stocks_close.index)
for c in stock_tickers:
    stocks_rsi[c] = calc_rsi(stocks_close[c])

# ==========================================
# 3. TẢI BỘ NHỚ AI & CHẤM ĐIỂM
# ==========================================
print("3. Đang tải não bộ AI & Đồng bộ Vĩ mô CSV...")
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
for col in ['DXY', 'US10Y', 'USD_VND']:
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()
z_cols = [f'Z_{c}' for c in ['DXY', 'US10Y', 'USD_VND']]

df_ml_list = []
for c in stock_tickers:
    df_c = pd.DataFrame({'Date': stocks_close.index, 'Ticker': c})
    df_c['Vol_Ratio'] = (stocks_volume[f'{c}_V'] / stocks_vol_ma20[f'{c}_V'].replace(0, np.nan)).values
    df_c['Dist_MA50'] = (stocks_close[c] / stocks_ma50[c].replace(0, np.nan)).values
    df_c['RSI'] = stocks_rsi[c].values
    df_c['RSI_Momentum'] = pd.Series(df_c['RSI']).diff(3).values 
    df_c['MACD_Hist'] = calc_macd_hist(stocks_close[c]).values
    df_ml_list.append(df_c)

ml_master = pd.concat(df_ml_list, ignore_index=True).merge(macro_df, on='Date', how='left')
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200'] + z_cols

MODEL_FILE = "xgboost_ultimate.json"
xgb_model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    xgb_model.load_model(MODEL_FILE)
else:
    sys.exit(f"⚠️ Không tìm thấy {MODEL_FILE}! Bác phải chạy file train_bot.py trước.")

ml_master['AI_Score'] = np.nan
valid_rows = ml_master[ml_features].notna().all(axis=1)
if valid_rows.sum() > 0:
    ml_master.loc[valid_rows, 'AI_Score'] = xgb_model.predict_proba(ml_master.loc[valid_rows, ml_features])[:, 1]

# Giữ nguyên DatetimeIndex cho ai_dict để khớp với vòng lặp
ai_dict = {c: ml_master[ml_master['Ticker'] == c].set_index('Date')['AI_Score'] for c in stock_tickers}

# ==========================================
# 4. CHẠY BACKTEST KẾ TOÁN (TREND FOLLOWING & GỒNG LÃI)
# ==========================================
print("4. Chạy Sổ Kế Toán (Trend Following & Gồng Lãi)...")
def run_ai_quant_system_with_log():
    backtest = df_clean.copy()
    MAX_SLOTS = 10
    LIQUIDITY_FEE = 0.0035  # Phí giao dịch + Thuế bán 0.1%
    DAILY_INTEREST = 0.05 / 252 
    cash = 100_000_000 
    held = {}
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
            continue
            
        cash *= (1 + DAILY_INTEREST)

        # Cập nhật số ngày nắm giữ cho quy tắc T+2.5
        for c in held:
            held[c]['days_held'] += 1

        to_remove = []
        # 1. CÔNG TẮC THỊ TRƯỜNG CHUNG: Bán tháo toàn bộ khi VNINDEX gãy MA100
        if idx < market_ma100:
            for c in list(held.keys()):
                if held[c]['days_held'] >= 2:  # Đảm bảo hàng đã về tài khoản mới bán được
                    sell_price = stocks_close.at[date, c]
                    cash += held[c]['qty'] * sell_price * (1 - LIQUIDITY_FEE)
                    profit = (sell_price / held[c]['buy_price'] - 1) * 100
                    trade_log.append({'Date': date, 'Action': 'SELL (Thị trường gãy MA100)', 'Ticker': c, 'Price': sell_price, 'Profit(%)': round(profit, 2)})
                    del held[c]
        else:
            # 2. THỊ TRƯỜNG VẪN TỐT: GỒNG LÃI TỐI ĐA, CHỈ BÁN KHI MẤT ĐÀ / ĐẢO CHIỀU
            for c in held.keys():
                curr_price = stocks_close.at[date, c]
                buy_price = held[c]['buy_price']
                held[c]['highest'] = max(held[c]['highest'], curr_price)
                
                profit_margin = (curr_price - buy_price) / buy_price
                peak_margin = (held[c]['highest'] - buy_price) / buy_price

                if held[c]['days_held'] >= 2:
                    # Trailing Stop: Đang có lãi lớn (>= 12%) nhưng đảo chiều rơi 6% từ đỉnh -> Chốt lời
                    if peak_margin >= 0.12 and curr_price < held[c]['highest'] * 0.94:
                        to_remove.append((c, 'SELL (Chốt lời Trailing)'))
                    # Chặn lãi: Thủng MA20 ngắn hạn khi đang có lãi nhẹ (> 5%) -> Nhả hàng
                    elif curr_price < stocks_ma20.at[date, c] and profit_margin > 0.05:
                        to_remove.append((c, 'SELL (Thủng MA20)'))
                    # KHÔNG CẮT LỖ KHI ÂM: Ôm tiếp nếu VNINDEX chưa gãy MA100

            for c, reason in to_remove:
                sell_price = stocks_close.at[date, c]
                cash += held[c]['qty'] * sell_price * (1 - LIQUIDITY_FEE)
                profit = (sell_price / held[c]['buy_price'] - 1) * 100
                trade_log.append({'Date': date, 'Action': reason, 'Ticker': c, 'Price': sell_price, 'Profit(%)': round(profit, 2)})
                del held[c]

            # Quản trị tỷ trọng theo độ biến động thị trường
            max_exposure = 0.3 if (idx < market_ma50 or market_atr_pct > 0.018) else 1.0
            stock_val = sum([held[c]['qty'] * stocks_close.at[date, c] for c in held])
            current_total_nav = cash + stock_val
            allowed_capital = max(0, (current_total_nav * max_exposure) - stock_val)

            # Ưu tiên kiểm định VN-Index phải Uptrend trước (Trên MA200 và ADX > 20)
            empty = MAX_SLOTS - len(held)
            if empty > 0 and (idx > market_ma200) and (market_adx > 20) and (allowed_capital > 0):
                buy_candidates = []
                for c in stock_tickers:
                    if c not in held:
                        curr_price = stocks_close.at[date, c]
                        curr_vol = stocks_volume.at[date, f'{c}_V']
                        prev_high_20 = stocks_high_20.at[date, c]
                        prev_vol_ma20 = stocks_vol_ma20.at[date, f'{c}_V']

                        if pd.isna(prev_high_20) or pd.isna(prev_vol_ma20): 
                            continue
                            
                        # Breakout giá 20 phiên và nổ volume gấp rưỡi
                        if curr_price > prev_high_20 and curr_vol > (prev_vol_ma20 * 1.5):
                            ai_score = ai_dict[c].get(date, np.nan)
                            if pd.notna(ai_score) and ai_score >= 0.65:
                                buy_candidates.append((c, ai_score, curr_vol / prev_vol_ma20 if prev_vol_ma20 > 0 else 0, stocks_rsi.at[date, c]))

                # Sắp xếp ưu tiên cổ phiếu có điểm AI cao nhất
                buy_candidates.sort(key=lambda x: x[1], reverse=True)

                if buy_candidates:
                    slot_invest = min(allowed_capital / empty, cash)
                    for c, score, v_ratio, s_rsi in buy_candidates:
                        if slot_invest > 1_000_000 and cash >= slot_invest and empty > 0:
                            buy_price = stocks_close.at[date, c]
                            actual_invest = slot_invest * (1 - LIQUIDITY_FEE)
                            held[c] = {
                                'qty': actual_invest / buy_price,
                                'buy_price': buy_price,
                                'highest': buy_price,
                                'days_held': 0
                            }
                            cash -= slot_invest
                            empty -= 1
                            trade_log.append({
                                'Date': date, 'Action': 'BUY', 'Ticker': c, 
                                'Price': buy_price, 'AI_Score': round(score, 2), 
                                'Vol_Ratio': round(v_ratio, 1), 'RSI': round(s_rsi, 1), 
                                'VN_MFI': round(market_mfi, 1)
                            })
                            if empty == 0 or cash <= 0:
                                break

    return pd.DataFrame(trade_log), held, cash

df_trade_log, current_held, final_cash = run_ai_quant_system_with_log()

# ==========================================
# 5. XUẤT FILE TRADING LOG CHO POWER BI
# ==========================================
if not df_trade_log.empty:
    df_trade_log.to_csv('trading_log.csv', index=False, encoding='utf-8-sig')
    print(f"✅ Đã ghi nhận {len(df_trade_log)} lệnh vào file trading_log.csv (Sẵn sàng cho Power BI)!")
else:
    print("⚠️ Chưa có lệnh nào phát sinh.")

# ==========================================
# 6. IN KẾT QUẢ RA CONSOLE
# ==========================================
print("\n" + "="*50)
print("📊 TỔNG KẾT BACKTEST HIỆU QUẢ QUÁ KHỨ")
print("="*50)

if not df_trade_log.empty and 'Action' in df_trade_log.columns:
    sells = df_trade_log[df_trade_log['Action'].str.contains('SELL', na=False)]
    total_sells = len(sells)
    if total_sells > 0:
        win_sells = len(sells[sells['Profit(%)'] > 0])
        print(f"Tổng số lệnh đã chốt sổ: {total_sells}")
        print(f"🎯 Tỷ lệ thắng (Win Rate): {win_sells / total_sells * 100:.2f}%")
        print(f"💰 Lệnh lãi cao nhất: +{sells['Profit(%)'].max()}%")
        print(f"🩸 Lệnh lỗ sâu nhất: {sells['Profit(%)'].min()}%")
    else:
        print("⏳ Bot đang ôm hàng theo trend, chưa chốt sổ lệnh BÁN nào.")
else:
    print("🤖 Tiêu chuẩn quá khắt khe, chưa có lệnh MUA/BÁN nào được khớp.")
print("="*50 + "\n")

# ==========================================
# 7. TELEGRAM RADAR: BẢN TIN CHIẾN LƯỢC TUẦN
# ==========================================
print("7. Báo cáo Telegram Chiến Lược Tuần...")
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_msg(text_message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n--- NỘI DUNG TELEGRAM ---\n", text_message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text_message, "parse_mode": "HTML"}, timeout=10)
        if res.status_code == 200:
            print("✅ Đã gửi báo cáo Telegram thành công!")
        else:
            print(f"⚠️ Telegram trả lời: {res.text}")
    except Exception as e:
        print(f"Lỗi gửi Telegram: {e}")

latest_date = df_clean['Date'].iloc[-1]
latest_idx = df_clean['VNINDEX'].iloc[-1]
latest_ma50 = df_clean['MA50'].iloc[-1]
latest_ma100 = df_clean['MA100'].iloc[-1]
latest_ma200 = df_clean['MA200'].iloc[-1]
latest_adx = df_clean['ADX'].iloc[-1]

# Lấy dữ liệu vĩ mô chốt tuần
latest_macro = macro_df.iloc[-1] if not macro_df.empty else None
dxy_val = f"{latest_macro['DXY']:.1f}" if latest_macro is not None and pd.notna(latest_macro.get('DXY')) else "N/A"
us10y_val = f"{latest_macro['US10Y']:.2f}%" if latest_macro is not None and pd.notna(latest_macro.get('US10Y')) else "N/A"
usdvnd_val = f"{latest_macro['USD_VND']:,.0f}" if latest_macro is not None and pd.notna(latest_macro.get('USD_VND')) else "N/A"

# Soạn thảo Bản tin tuần
msg = f"📊 <b>BẢN TIN CHIẾN LƯỢC TUẦN - RADAR VN30</b> 📊\n"
msg += f"📅 Chốt tuần ngày: <b>{latest_date.strftime('%d/%m/%Y')}</b>\n"
msg += f"➖➖➖➖➖➖➖➖➖➖\n\n"

# 1. Bức tranh Vĩ mô
msg += f"🌍 <b>VĨ MÔ QUỐC TẾ CHỐT TUẦN:</b>\n"
msg += f"• DXY: <b>{dxy_val}</b> | US10Y: <b>{us10y_val}</b>\n"
msg += f"• Tỷ giá USD/VND: <b>{usdvnd_val}</b>\n\n"

# 2. Đánh giá Xu hướng VN-Index (Công tắc số 1)
is_uptrend = (latest_idx > latest_ma200) and (latest_adx > 20)
is_downtrend = (latest_idx < latest_ma100)

msg += f"📈 <b>XU HƯỚNG VN-INDEX:</b>\n"
msg += f"• Điểm số: <b>{latest_idx:.1f}</b>\n"
msg += f"• MA100: {latest_ma100:.1f} | MA200: {latest_ma200:.1f} | ADX: {latest_adx:.1f}\n"

if is_downtrend:
    msg += f"🛑 <b>Trạng thái: NGUY HIỂM (Dưới MA100)</b>\n"
    msg += f"<i>Khuyến nghị: Thị trường gãy kênh, ưu tiên cầm 100% Tiền mặt nhận lãi Bank 5%/năm.</i>\n\n"
elif is_uptrend:
    msg += f"🟢 <b>Trạng thái: UPTREND XÁC NHẬN (Trên MA200)</b>\n"
    msg += f"<i>Đèn xanh bật: Cho phép kích hoạt quét cổ phiếu VN30 để gom hàng cho tuần tới.</i>\n\n"
else:
    msg += f"🟡 <b>Trạng thái: TÍCH LŨY / CHỜ XÁC NHẬN</b>\n"
    msg += f"<i>Khuyến nghị: Thị trường phân hóa, chỉ giải ngân thăm dò với tỷ trọng nhỏ.</i>\n\n"

# 3. Quét cổ phiếu VN30 cho tuần tới
msg += f"🎯 <b>CỔ PHIẾU TIỀM NĂNG CHO TUẦN TỚI:</b>\n"
weekly_picks = []

if is_uptrend:
    for c in stock_tickers:
        curr_price = stocks_close.at[latest_date, c]
        curr_vol = stocks_volume.at[latest_date, f'{c}_V']
        prev_high_20 = stocks_high_20.at[latest_date, c]
        prev_vol_ma20 = stocks_vol_ma20.at[latest_date, f'{c}_V']
        
        if pd.notna(prev_high_20) and pd.notna(prev_vol_ma20):
            # Điều kiện Breakout đỉnh 20 ngày + Vol x1.5
            if curr_price > prev_high_20 and curr_vol > (prev_vol_ma20 * 1.5):
                score = ai_dict[c].get(latest_date, np.nan)
                if pd.notna(score) and score >= 0.65:
                    weekly_picks.append((c, curr_price, score, curr_vol / prev_vol_ma20))

    weekly_picks.sort(key=lambda x: x[2], reverse=True)

if weekly_picks:
    for ticker, price, score, v_r in weekly_picks:
        msg += f"⭐ <b>{ticker}</b> (Giá: {price:,.0f}) | AI: <b>{score*100:.1f}%</b> | Vol: <b>x{v_r:.1f}</b>\n"
else:
    msg += f"<i>Chưa có mã VN30 nào hội tụ đủ điểm nổ Vol và Alpha tuần này. Kiên nhẫn chờ cơ hội.</i>\n"

# 4. Danh mục đang nắm giữ
msg += f"\n💼 <b>DANH MỤC HIỆN TẠI:</b>\n"
if current_held:
    for ticker, info in current_held.items():
        curr_price = stocks_close.at[latest_date, ticker]
        pnl = (curr_price / info['buy_price'] - 1) * 100
        icon = "🟢" if pnl >= 0 else "🔴"
        msg += f"{icon} <b>{ticker}</b>: {pnl:+.1f}% (T+{info['days_held']})\n"
else:
    msg += f"<i>100% Tiền mặt (Đang nhận lãi Bank qua đêm).</i>\n"

msg += f"➖➖➖➖➖➖➖➖➖➖"
send_telegram_msg(msg)
