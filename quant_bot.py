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

stock_tickers = [c for c in valid_tickers if c in stocks_close.columns]

stocks_close_clean = stocks_close.ffill()
stocks_vol_clean = stocks_volume.fillna(0)

stocks_high_20 = stocks_close_clean.rolling(20).max().shift(1)
stocks_ma20 = stocks_close_clean.rolling(20).mean()
stocks_ma50 = stocks_close_clean.rolling(50).mean()
stocks_vol_ma20 = stocks_vol_clean.rolling(20).mean().shift(1)

stocks_rsi = pd.DataFrame(index=stocks_close.index)
for c in stock_tickers:
    stocks_rsi[c] = calc_rsi(stocks_close_clean[c])

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
    s_price = stocks_close[c].dropna()
    if len(s_price) < 100: 
        continue
    valid_dates = s_price.index
    s_vol = stocks_volume[f'{c}_V'].reindex(valid_dates).fillna(0)
    
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

ai_dict = {c: ml_master[ml_master['Ticker'] == c].set_index('Date')['AI_Score'] for c in stock_tickers}

# ==========================================
# 4. CHẠY BACKTEST KẾ TOÁN (CÓ TRƯỢT GIÁ THỰC TẾ & TÍNH MDD)
# ==========================================
print("4. Chạy Sổ Kế Toán (5 Slots + Trượt Giá 0.5% + Ghi Nhận Maximum Drawdown)...")
def run_ai_quant_system_with_log():
    backtest = df_clean[df_clean['Date'] >= '2014-01-01'].copy().reset_index(drop=True)
    MAX_SLOTS = 5
    
    # 🎯 MA SÁT THỊ TRƯỜNG THỰC TẾ:
    BROKER_FEE = 0.0015   # Phí giao dịch 0.15%
    TAX_SELL = 0.0010     # Thuế bán 0.1%
    SLIPPAGE = 0.0050     # Độ trượt giá 0.5% mỗi chiều (mua đuổi / bán tháo)
    
    BUY_FRICTION = BROKER_FEE + SLIPPAGE        # Tổng ma sát khi Mua: ~0.65%
    SELL_FRICTION = BROKER_FEE + TAX_SELL + SLIPPAGE # Tổng ma sát khi Bán: ~0.75%
    
    ANNUAL_BANK_RATE = 0.04 # Lãi suất quản lý tiền nhàn rỗi thận trọng: 4%/năm
    cash = 100_000_000 
    held = {}
    trade_log = [] 
    daily_nav_log = []
    total_bank_interest = 0.0
    prev_date = None

    for i in range(len(backtest)):
        date = backtest['Date'].iloc[i]
        idx = backtest['VNINDEX'].iloc[i]
        market_ma50 = backtest['MA50'].iloc[i]
        market_ma100 = backtest['MA100'].iloc[i]
        market_ma200 = backtest['MA200'].iloc[i]
        market_adx = backtest['ADX'].iloc[i]
        market_atr_pct = backtest['ATR_PCT'].iloc[i]
        market_mfi = backtest['VN_MFI'].iloc[i]

        if date not in stocks_close_clean.index: 
            continue
            
        # 🏦 Tính lãi ngân hàng cho tiền mặt theo ngày thực tế (365 ngày)
        if prev_date is not None:
            cal_days = (date - prev_date).days
            if cal_days > 0 and cash > 0:
                interest = cash * (((1 + ANNUAL_BANK_RATE) ** (cal_days / 365.25)) - 1)
                cash += interest
                total_bank_interest += interest
        prev_date = date

        # Cập nhật số ngày nắm giữ cho quy tắc T+2.5
        for c in held:
            held[c]['days_held'] += 1

        to_remove = []
        # 1. CÔNG TẮC THỊ TRƯỜNG CHUNG: Bán tháo toàn bộ khi VNINDEX gãy MA100
        if idx < market_ma100:
            for c in list(held.keys()):
                if held[c]['days_held'] >= 2:
                    sell_price = stocks_close_clean.at[date, c]
                    if pd.notna(sell_price) and sell_price > 0:
                        net_cash_received = held[c]['qty'] * sell_price * (1 - SELL_FRICTION)
                        cash += net_cash_received
                        profit = (sell_price * (1 - SELL_FRICTION) / held[c]['cost_basis'] - 1) * 100
                        trade_log.append({'Date': date, 'Action': 'SELL (Thị trường gãy MA100)', 'Ticker': c, 'Price': sell_price, 'Profit(%)': round(profit, 2)})
                        del held[c]
        else:
            # 2. THỊ TRƯỜNG VẪN TỐT: GỒNG LÃI TỐI ĐA, CHỈ BÁN KHI MẤT ĐÀ / ĐẢO CHIỀU
            for c in list(held.keys()):
                curr_price = stocks_close_clean.at[date, c]
                if pd.isna(curr_price) or curr_price <= 0:
                    continue
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

            for c, reason in to_remove:
                sell_price = stocks_close_clean.at[date, c]
                net_cash_received = held[c]['qty'] * sell_price * (1 - SELL_FRICTION)
                cash += net_cash_received
                profit = (sell_price * (1 - SELL_FRICTION) / held[c]['cost_basis'] - 1) * 100
                trade_log.append({'Date': date, 'Action': reason, 'Ticker': c, 'Price': sell_price, 'Profit(%)': round(profit, 2)})
                del held[c]

            # Quản trị tỷ trọng theo NAV hiện tại
            stock_val = sum([held[c]['qty'] * stocks_close_clean.at[date, c] for c in held if pd.notna(stocks_close_clean.at[date, c])])
            current_total_nav = cash + stock_val
            max_exposure = 0.3 if (idx < market_ma50 or market_atr_pct > 0.018) else 1.0
            allowed_capital = max(0, (current_total_nav * max_exposure) - stock_val)

            empty = MAX_SLOTS - len(held)
            if empty > 0 and (idx > market_ma200) and (market_adx > 20) and (allowed_capital > 0):
                buy_candidates = []
                for c in stock_tickers:
                    if c not in held:
                        if c not in stocks_close_clean.columns or f'{c}_V' not in stocks_vol_clean.columns:
                            continue
                        curr_price = stocks_close_clean.at[date, c]
                        curr_vol = stocks_vol_clean.at[date, f'{c}_V']
                        prev_high_20 = stocks_high_20.at[date, c]
                        prev_vol_ma20 = stocks_vol_ma20.at[date, f'{c}_V']

                        if pd.isna(curr_price) or pd.isna(prev_high_20) or pd.isna(prev_vol_ma20) or curr_price <= 0: 
                            continue
                            
                        # Breakout giá 20 phiên và nổ volume gấp rưỡi
                        if curr_price > prev_high_20 and curr_vol > (prev_vol_ma20 * 1.5):
                            ai_score = ai_dict[c].get(date, np.nan)
                            if pd.notna(ai_score) and ai_score >= 0.65:
                                s_rsi_val = stocks_rsi.at[date, c] if c in stocks_rsi.columns and date in stocks_rsi.index else 50
                                buy_candidates.append((c, ai_score, curr_vol / prev_vol_ma20 if prev_vol_ma20 > 0 else 0, s_rsi_val))

                buy_candidates.sort(key=lambda x: x[1], reverse=True)

                if buy_candidates:
                    target_slot_val = (current_total_nav * max_exposure) / MAX_SLOTS
                    for c, score, v_ratio, s_rsi in buy_candidates:
                        slot_invest = min(target_slot_val, cash, allowed_capital)
                        if slot_invest > 1_000_000 and empty > 0 and cash >= slot_invest:
                            buy_price = stocks_close_clean.at[date, c]
                            # Tính cả trượt giá và phí mua vào giá vốn
                            actual_invest = slot_invest * (1 - BUY_FRICTION)
                            held[c] = {
                                'qty': actual_invest / buy_price,
                                'buy_price': buy_price,
                                'cost_basis': buy_price * (1 + BUY_FRICTION), # Giá vốn thực tế sau ma sát
                                'highest': buy_price,
                                'days_held': 0
                            }
                            cash -= slot_invest
                            allowed_capital -= slot_invest
                            empty -= 1
                            trade_log.append({
                                'Date': date, 'Action': 'BUY', 'Ticker': c, 
                                'Price': buy_price, 'AI_Score': round(score, 2), 
                                'Vol_Ratio': round(v_ratio, 1), 'RSI': round(s_rsi, 1), 
                                'VN_MFI': round(market_mfi, 1)
                            })
                            if empty == 0 or cash <= 0 or allowed_capital <= 0:
                                break

        # Ghi nhận NAV hàng ngày để vẽ đường cong tài sản & tính Maximum Drawdown
        current_nav = cash + sum([held[c]['qty'] * stocks_close_clean.at[date, c] for c in held if pd.notna(stocks_close_clean.at[date, c])])
        daily_nav_log.append({'Date': date, 'NAV': current_nav})

    return pd.DataFrame(trade_log), pd.DataFrame(daily_nav_log), held, cash, total_bank_interest

df_trade_log, df_daily_nav, current_held, final_cash, total_bank_earned = run_ai_quant_system_with_log()

# ==========================================
# 5. XUẤT FILE TRADING LOG CHO POWER BI
# ==========================================
latest_date = df_clean['Date'].iloc[-1]

if not df_trade_log.empty:
    df_trade_log.to_csv('trading_log.csv', index=False, encoding='utf-8-sig')
    print(f"✅ Đã ghi nhận {len(df_trade_log)} lệnh vào file trading_log.csv (Sẵn sàng cho Power BI)!")
else:
    print("⚠️ Chưa có lệnh nào phát sinh.")

# ==========================================
# 6. IN KẾT QUẢ RA CONSOLE (TÍNH TOÁN MAXIMUM DRAWDOWN CHUẨN QUỸ)
# ==========================================
print("\n" + "="*55)
print("📊 BÁO CÁO ĐỊNH LƯỢNG RỦI RO & HIỆU QUẢ DANH MỤC 10 NĂM")
print("="*55)

INITIAL_CAPITAL = 100_000_000
stock_val_final = sum([current_held[c]['qty'] * stocks_close_clean.at[latest_date, c] for c in current_held if pd.notna(stocks_close_clean.at[latest_date, c])])
total_nav_final = final_cash + stock_val_final
total_return_pct = (total_nav_final / INITIAL_CAPITAL - 1) * 100

start_backtest_date = pd.to_datetime('2014-01-01')
total_years = (latest_date - start_backtest_date).days / 365.25
cagr = ((total_nav_final / INITIAL_CAPITAL) ** (1 / total_years) - 1) * 100 if total_years > 0 else 0

# 📉 TÍNH MAXIMUM DRAWDOWN (SỤT GIẢM TỐI ĐA TOÀN TÀI KHOẢN TỪ ĐỈNH)
if not df_daily_nav.empty:
    df_daily_nav['Peak_NAV'] = df_daily_nav['NAV'].cummax()
    df_daily_nav['Drawdown'] = (df_daily_nav['NAV'] - df_daily_nav['Peak_NAV']) / df_daily_nav['Peak_NAV']
    max_dd = df_daily_nav['Drawdown'].min() * 100
    
    # Tìm thời điểm đỉnh và đáy của đợt sụt giảm lớn nhất
    worst_idx = df_daily_nav['Drawdown'].idxmin()
    worst_row = df_daily_nav.loc[worst_idx]
    peak_row = df_daily_nav.loc[:worst_idx].loc[df_daily_nav.loc[:worst_idx, 'NAV'].idxmax()]
    mdd_peak_date = peak_row['Date'].strftime('%d/%m/%Y')
    mdd_trough_date = worst_row['Date'].strftime('%d/%m/%Y')
else:
    max_dd = 0.0
    mdd_peak_date, mdd_trough_date = "N/A", "N/A"

print(f"💵 Vốn khởi điểm (01/2014): {INITIAL_CAPITAL:,.0f} VNĐ")
print(f"🏦 TỔNG TÀI SẢN HIỆN TẠI (NAV): {total_nav_final:,.0f} VNĐ")
print(f"💰 Lãi gửi Bank thu về (4%/năm): +{total_bank_earned:,.0f} VNĐ")
print(f"🚀 TỔNG LỢI NHUẬN (Đã trừ trượt giá & phí): {total_return_pct:+,.2f}%")
print(f"📈 Lợi nhuận kép trung bình (CAGR): {cagr:.2f}% / năm")
print("-" * 55)
print(f"📉 SỤT GIẢM TỐI ĐA (Maximum Drawdown - MDD): {max_dd:.2f}%")
print(f"   (Thời kỳ chịu đựng lớn nhất: Từ đỉnh {mdd_peak_date} rơi về đáy {mdd_trough_date})")
print(f"📌 (So sánh: Năm 2022 VN-Index sụt -42.5%, ETF VN30 sụt -48.2%)")
print("-" * 55)

if not df_trade_log.empty and 'Action' in df_trade_log.columns:
    sells = df_trade_log[df_trade_log['Action'].str.contains('SELL', na=False)]
    total_sells = len(sells)
    if total_sells > 0:
        win_sells = len(sells[sells['Profit(%)'] > 0])
        print(f"Tổng số lệnh đã chốt sổ: {total_sells}")
        print(f"🎯 Tỷ lệ thắng (Win Rate): {win_sells / total_sells * 100:.2f}% (Thắng {win_sells}/{total_sells} trận)")
        print(f"💰 Lệnh lãi đơn lẻ cao nhất: +{sells['Profit(%)'].max()}%")
        print(f"🩸 Lệnh lỗ đơn lẻ sâu nhất: {sells['Profit(%)'].min()}%")
    else:
        print("⏳ Bot đang ôm hàng theo trend, chưa chốt sổ lệnh BÁN nào.")
print("="*55 + "\n")

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
            print(f"⚠️ Telegram phản hồi: {res.text}")
    except Exception as e:
        print(f"Lỗi gửi Telegram: {e}")

latest_idx = df_clean['VNINDEX'].iloc[-1]
latest_ma50 = df_clean['MA50'].iloc[-1]
latest_ma100 = df_clean['MA100'].iloc[-1]
latest_ma200 = df_clean['MA200'].iloc[-1]
latest_adx = df_clean['ADX'].iloc[-1]

latest_macro = macro_df.iloc[-1] if not macro_df.empty else None
dxy_val = f"{latest_macro['DXY']:.1f}" if latest_macro is not None and pd.notna(latest_macro.get('DXY')) else "N/A"
us10y_val = f"{latest_macro['US10Y']:.2f}%" if latest_macro is not None and pd.notna(latest_macro.get('US10Y')) else "N/A"
usdvnd_val = f"{latest_macro['USD_VND']:,.0f}" if latest_macro is not None and pd.notna(latest_macro.get('USD_VND')) else "N/A"

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
    msg += f"<i>Khuyến nghị: Thị trường gãy kênh, ưu tiên cầm 100% Tiền mặt nhận lãi Bank 4%/năm.</i>\n\n"
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
        if c in stocks_close_clean.columns and latest_date in stocks_close_clean.index:
            curr_price = stocks_close_clean.at[latest_date, c]
            curr_vol = stocks_vol_clean.at[latest_date, f'{c}_V']
            prev_high_20 = stocks_high_20.at[latest_date, c]
            prev_vol_ma20 = stocks_vol_ma20.at[latest_date, f'{c}_V']
            
            if pd.notna(curr_price) and pd.notna(prev_high_20) and pd.notna(prev_vol_ma20):
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

# 4. Danh mục & Hiệu quả tài sản
msg += f"\n💼 <b>TÀI CHÍNH & DANH MỤC HIỆN TẠI:</b>\n"
msg += f"• Tổng NAV: <b>{total_nav_final:,.0f} VNĐ</b> ({total_return_pct:+,.1f}%)\n"
msg += f"• Tiền lãi gửi Bank đã thu: <b>+{total_bank_earned:,.0f} VNĐ</b>\n"
msg += f"• Rủi ro sụt giảm tối đa (MDD): <b>{max_dd:.1f}%</b>\n"
if current_held:
    for ticker, info in current_held.items():
        curr_price = stocks_close_clean.at[latest_date, ticker]
        pnl = (curr_price / info['buy_price'] - 1) * 100
        icon = "🟢" if pnl >= 0 else "🔴"
        msg += f"{icon} <b>{ticker}</b>: {pnl:+.1f}% (T+{info['days_held']})\n"
else:
    msg += f"• Cổ phiếu: <i>100% Tiền mặt (Đang hưởng lãi Bank 4%/năm).</i>\n"

msg += f"➖➖➖➖➖➖➖➖➖➖"
send_telegram_msg(msg)
