import pandas as pd
import numpy as np
import requests
from tvDatafeed import TvDatafeed, Interval
import xgboost as xgb
import warnings
import sys
import os
import datetime

from vn30_universe import get_vn30_constituents, ALL_HISTORICAL_VN30_TICKERS

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
    """
    Tải dữ liệu từ TradingView. Nếu mã bị hủy niêm yết hoặc thiếu dữ liệu,
    tự động fallback sang VNDirect DChart API để có trọn vẹn 10 năm.
    """
    # 1. Thử sàn HOSE
    try:
        df = tv.get_hist(symbol=symbol, exchange='HOSE', interval=Interval.in_daily, n_bars=n_bars)
        if df is not None and len(df) > 50:
            df = df.reset_index()
            df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    # 2. Thử sàn UPCOM (cho các mã bị chuyển sàn như FLC, ITA...)
    try:
        df = tv.get_hist(symbol=symbol, exchange='UPCOM', interval=Interval.in_daily, n_bars=n_bars)
        if df is not None and len(df) > 50:
            df = df.reset_index()
            df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Date').reset_index(drop=True)
    except Exception:
        pass

    # 3. Fallback sang VNDirect DChart API
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
    sys.exit("❌ LỖI: Không thể tải dữ liệu VNINDEX từ cả TradingView lẫn VNDirect!")

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
print(f"2. Đang tải toàn bộ {len(ALL_HISTORICAL_VN30_TICKERS)} mã cổ phiếu VN30 lịch sử (Chống Survivorship Bias)...")

stocks_close = pd.DataFrame({'Date': df_clean['Date']}).set_index('Date')
stocks_volume = pd.DataFrame({'Date': df_clean['Date']}).set_index('Date')
stocks_high = pd.DataFrame({'Date': df_clean['Date']}).set_index('Date')
stocks_low = pd.DataFrame({'Date': df_clean['Date']}).set_index('Date')

loaded_tickers = []
delisted_or_empty = []

for ticker in ALL_HISTORICAL_VN30_TICKERS:
    df_t = fetch_ticker_data(ticker, n_bars=4000)
    if df_t is not None and not df_t.empty:
        df_t.set_index('Date', inplace=True)
        stocks_close[ticker] = df_t['Close']
        stocks_volume[f'{ticker}_V'] = df_t['Volume']
        stocks_high[f'{ticker}_H'] = df_t['High']
        stocks_low[f'{ticker}_L'] = df_t['Low']
        loaded_tickers.append(ticker)
    else:
        delisted_or_empty.append(ticker)

print(f"   ✅ Đã nạp thành công {len(loaded_tickers)} mã.")
if delisted_or_empty:
    print(f"   ⚠️ Các mã không có dữ liệu (đã hủy niêm yết triệt để): {', '.join(delisted_or_empty)}")

# Tính các đường trung bình và chỉ báo panel
stocks_high_20 = stocks_close.rolling(20).max().shift(1)
stocks_ma20 = stocks_close.rolling(20).mean()
stocks_ma50 = stocks_close.rolling(50).mean()
stocks_vol_ma20 = stocks_volume.rolling(20).mean().shift(1)

stocks_rsi = pd.DataFrame(index=stocks_close.index)
stocks_macd = pd.DataFrame(index=stocks_close.index)
for c in loaded_tickers:
    stocks_rsi[c] = calc_rsi(stocks_close[c])
    stocks_macd[c] = calc_macd_hist(stocks_close[c])

# ==========================================
# 3. TẢI BỘ NHỚ AI & ĐỒNG BỘ VĨ MÔ CSV
# ==========================================
print("3. Đang tải Não bộ AI tĩnh & Đồng bộ Vĩ mô CSV...")
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
clean_dates_str = pd.to_datetime(stocks_close.index).strftime('%Y-%m-%d')

for c in loaded_tickers:
    s_close = stocks_close[c]
    s_vol = stocks_volume[f'{c}_V']
    s_vol_ma = stocks_vol_ma20[f'{c}_V']
    s_ma50 = stocks_ma50[c]

    valid_mask = s_close.notna() & (s_close > 0)
    if valid_mask.sum() < 30:
        continue

    df_c = pd.DataFrame({
        'Date': clean_dates_str[valid_mask],
        'Ticker': c,
        'Vol_Ratio': (s_vol[valid_mask] / s_vol_ma[valid_mask].replace(0, np.nan)).values,
        'Dist_MA50': (s_close[valid_mask] / s_ma50[valid_mask]).values,
        'RSI': stocks_rsi[c][valid_mask].values,
        'RSI_Momentum': pd.Series(stocks_rsi[c][valid_mask]).diff(3).values,
        'MACD_Hist': stocks_macd[c][valid_mask].values
    })
    df_ml_list.append(df_c)

ml_master = pd.concat(df_ml_list, ignore_index=True).merge(macro_df, on='Date', how='left')
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'RSI_Momentum', 'VN_MFI', 'MACD_Hist', 'VN_Dist_MA200', 'Z_DXY', 'Z_US10Y', 'Z_USD_VND']

MODEL_FILE = "xgboost_ultimate.json"
xgb_model = xgb.XGBClassifier()

if os.path.exists(MODEL_FILE):
    xgb_model.load_model(MODEL_FILE)
    print(f"   🧠 Đã nạp thành công mô hình: {MODEL_FILE}")
else:
    sys.exit(f"⚠️ Không tìm thấy {MODEL_FILE}! Bác phải chạy train_bot.py trước.")

ml_master['AI_Score'] = np.nan
valid_rows = ml_master[ml_features].notna().all(axis=1)
if valid_rows.sum() > 0:
    ml_master.loc[valid_rows, 'AI_Score'] = xgb_model.predict_proba(ml_master.loc[valid_rows, ml_features])[:, 1]

# Tạo tra cứu AI Score dạng dict (Key: date_str, ticker) -> score
ai_dict = {}
for c in loaded_tickers:
    c_sub = ml_master[ml_master['Ticker'] == c]
    ai_dict[c] = dict(zip(c_sub['Date'], c_sub['AI_Score']))

# ==========================================
# 4. CHẠY SỔ KẾ TOÁN THỰC CHIẾN (POINT-IN-TIME + LIQUIDITY LOCK + 5 SLOTS)
# ==========================================
print("4. Chạy Sổ Kế Toán (Point-in-Time VN30 + 5 Slots + Compounding + T+2.5)...")

def run_ai_quant_system():
    # Lọc kiểm thử 10 năm từ 01/2014 đến nay
    backtest = df_clean[df_clean['Date'] >= '2014-01-01'].copy().reset_index(drop=True)
    
    MAX_SLOTS = 5                      # Tối đa 5 vị thế (20% NAV / slot)
    BUY_FRICTION = 0.0065              # 0.5% trượt giá + 0.15% phí mua
    SELL_FRICTION = 0.0075             # 0.5% trượt giá + 0.15% phí bán + 0.10% thuế
    ANNUAL_BANK_RATE = 0.04            # 4.0% / năm lãi gửi tiền mặt
    DAILY_BANK_RATE = ANNUAL_BANK_RATE / 252
    MIN_HOLD_DAYS = 3                  # Chu kỳ T+2.5 (mua T, bán sớm nhất T+3)
    
    cash = 100_000_000.0               # Vốn khởi điểm 100 triệu VNĐ
    total_bank_interest = 0.0
    held = {}                          # {ticker: {'qty': ..., 'buy_price': ..., 'highest': ..., 'buy_date': ..., 'hold_days': 0}}
    trade_log = []
    daily_nav_history = []

    for i in range(len(backtest)):
        date = backtest['Date'].iloc[i]
        date_str = date.strftime('%Y-%m-%d')
        idx = backtest['VNINDEX'].iloc[i]
        market_ma50 = backtest['MA50'].iloc[i]
        market_ma100 = backtest['MA100'].iloc[i]
        market_ma200 = backtest['MA200'].iloc[i]
        market_adx = backtest['ADX'].iloc[i]
        market_atr_pct = backtest['ATR_PCT'].iloc[i]
        market_mfi = backtest['VN_MFI'].iloc[i]

        if date not in stocks_close.index:
            continue

        # Lấy rổ VN30 chính xác tại ngày hôm đó (Chống Survivorship Bias)
        active_vn30 = get_vn30_constituents(date_str)

        # Tiền mặt nhàn rỗi nhận lãi ngân hàng
        if cash > 0:
            interest_today = cash * DAILY_BANK_RATE
            cash += interest_today
            total_bank_interest += interest_today

        # Cập nhật số ngày nắm giữ và giá đỉnh cao nhất của từng vị thế
        for c in held:
            held[c]['hold_days'] += 1
            curr_p = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
            if pd.notna(curr_p) and curr_p > 0:
                held[c]['highest'] = max(held[c]['highest'], curr_p)

        # ------------------------------------------------------------------
        # HÀM KIỂM TRA TẮC THANH KHOẢN ("TRẮNG BÊN MUA" / MÚA BÊN TRĂNG)
        # ------------------------------------------------------------------
        def is_liquidity_locked(ticker):
            """
            Nếu cổ phiếu rơi sàn (~-7%) và thanh khoản bị nghẽn kiệt (<15% MA20 volume),
            nhà đầu tư KHÔNG THỂ BÁN ĐƯỢC trên sàn HOSE thực tế (như OGC 2015, NVL 2022).
            Lệnh phải chờ đến khi thanh khoản được giải cứu mở sàn.
            """
            if i == 0 or ticker not in stocks_close.columns:
                return False
            prev_date = backtest['Date'].iloc[i-1]
            if prev_date not in stocks_close.index:
                return False
            c_p = stocks_close.at[date, ticker]
            p_p = stocks_close.at[prev_date, ticker]
            c_v = stocks_volume.at[date, f'{ticker}_V']
            v_ma = stocks_vol_ma20.at[date, f'{ticker}_V']
            if pd.isna(c_p) or pd.isna(p_p) or pd.isna(c_v) or pd.isna(v_ma):
                return False
            # Rơi sát sàn và thanh khoản tắt ngúm
            if p_p > 0 and (c_p <= p_p * 0.933) and (v_ma > 0 and c_v < v_ma * 0.15):
                return True
            return False

        # ------------------------------------------------------------------
        # BƯỚC 1: XỬ LÝ CÁC TÍN HIỆU BÁN (EXIT STRATEGY)
        # ------------------------------------------------------------------
        to_sell = [] # list of tuples: (ticker, reason)

        # CÔNG TẮC VĨ MÔ: Gãy MA100 -> Kích hoạt lệnh bán toàn bộ danh mục
        if idx < market_ma100:
            for c in list(held.keys()):
                if held[c]['hold_days'] >= MIN_HOLD_DAYS:
                    if not is_liquidity_locked(c):
                        to_sell.append((c, 'SELL (Cắt Macro gãy MA100)'))
        else:
            for c in list(held.keys()):
                # Kiểm tra chu kỳ T+2.5
                if held[c]['hold_days'] < MIN_HOLD_DAYS:
                    continue

                # Nếu bị kẹt thanh khoản (múa bên trăng), không thể bán hôm nay!
                if is_liquidity_locked(c):
                    continue

                curr_price = stocks_close.at[date, c] if c in stocks_close.columns else np.nan
                if pd.isna(curr_price) or curr_price <= 0:
                    continue

                buy_price = held[c]['buy_price']
                profit_margin = (curr_price - buy_price) / buy_price
                peak_margin = (held[c]['highest'] - buy_price) / buy_price

                # LÝ DO 1: Bị HOSE loại khỏi rổ VN30 tại kỳ đảo danh mục (như OGC 2015, ROS 2021, NVL 2023)
                if c not in active_vn30:
                    to_sell.append((c, 'SELL (Rời rổ VN30)'))
                # LÝ DO 2: Trailing Stop (Lãi đỉnh >= 12% nhưng tụt 6% từ đỉnh thì chốt khóa lãi)
                elif peak_margin >= 0.12 and curr_price < (held[c]['highest'] * 0.94):
                    to_sell.append((c, 'SELL (Trailing Stop chốt lời)'))
                # LÝ DO 3: Gãy MA20 ngắn hạn khi đang có lãi nhẹ
                elif c in stocks_ma20.columns and pd.notna(stocks_ma20.at[date, c]) and curr_price < stocks_ma20.at[date, c] and profit_margin > 0.05:
                    to_sell.append((c, 'SELL (Thủng MA20 bảo toàn lãi)'))

        # Khớp các lệnh bán hợp lệ
        for c, reason in to_sell:
            if c in held:
                sell_price = stocks_close.at[date, c]
                qty = held[c]['qty']
                gross_val = qty * sell_price
                net_val = gross_val * (1.0 - SELL_FRICTION)
                cash += net_val
                pnl_pct = ((sell_price * (1.0 - SELL_FRICTION)) / (held[c]['buy_price'] * (1.0 + BUY_FRICTION)) - 1.0) * 100.0
                trade_log.append({
                    'Date': date_str,
                    'Action': reason,
                    'Ticker': c,
                    'Price': round(sell_price, 2),
                    'Profit(%)': round(pnl_pct, 2),
                    'Hold_Days': held[c]['hold_days']
                })
                del held[c]

        # ------------------------------------------------------------------
        # BƯỚC 2: TÍNH TOÁN VỐN VÀ QUÉT TÍN HIỆU MUA (BUY STRATEGY)
        # ------------------------------------------------------------------
        # Tính tổng NAV hiện tại (Tiền mặt + Giá trị thị trường các cổ phiếu đang giữ)
        stock_val_now = sum([
            held[c]['qty'] * stocks_close.at[date, c]
            for c in held.keys()
            if c in stocks_close.columns and pd.notna(stocks_close.at[date, c])
        ])
        current_nav = cash + stock_val_now
        daily_nav_history.append({'Date': date, 'NAV': current_nav})

        empty_slots = MAX_SLOTS - len(held)
        max_exposure = 0.4 if (idx < market_ma50 or market_atr_pct > 0.018) else 1.0
        allowed_stock_cap = (current_nav * max_exposure) - stock_val_now

        # Điều kiện thị trường cho phép giải ngân: VNINDEX > MA200 và ADX > 22
        if empty_slots > 0 and (idx > market_ma200) and (market_adx > 22) and (allowed_stock_cap > 0):
            buy_candidates = []

            # QUAN TRỌNG NHẤT: CHỈ QUÉT CÁC CỔ PHIẾU NẰM TRONG RỔ VN30 TẠI NGÀY HÔM ĐÓ!
            for c in active_vn30:
                if c not in held and c in loaded_tickers and c in stocks_close.columns:
                    curr_price = stocks_close.at[date, c]
                    curr_vol = stocks_volume.at[date, f'{c}_V'] if f'{c}_V' in stocks_volume.columns else np.nan
                    prev_h20 = stocks_high_20.at[date, c] if c in stocks_high_20.columns else np.nan
                    prev_v_ma = stocks_vol_ma20.at[date, f'{c}_V'] if f'{c}_V' in stocks_vol_ma20.columns else np.nan

                    if pd.isna(curr_price) or pd.isna(curr_vol) or pd.isna(prev_h20) or pd.isna(prev_v_ma):
                        continue

                    # Điểm nổ Von (Volume Breakout) và vượt đỉnh 20 phiên
                    if curr_price > prev_h20 and prev_v_ma > 0 and curr_vol > (prev_v_ma * 1.5):
                        score = ai_dict.get(c, {}).get(date_str, np.nan)
                        if pd.notna(score) and score >= 0.65: # Chuẩn Alpha AI >= 65%
                            v_ratio = curr_vol / prev_v_ma
                            r_rsi = stocks_rsi.at[date, c] if c in stocks_rsi.columns else 50.0
                            buy_candidates.append({
                                'Ticker': c,
                                'Score': score,
                                'Price': curr_price,
                                'Vol_Ratio': v_ratio,
                                'RSI': r_rsi
                            })

            # Ưu tiên mã có điểm AI Score cao nhất
            if buy_candidates:
                buy_candidates = sorted(buy_candidates, key=lambda x: x['Score'], reverse=True)

                # LÃI KÉP TOÀN DIỆN (COMPOUNDING): Chia đều NAV hiện tại cho 5 slot
                target_slot_size = current_nav / MAX_SLOTS

                for cand in buy_candidates:
                    if empty_slots <= 0 or cash <= 1_000_000:
                        break
                    c = cand['Ticker']
                    buy_price = cand['Price']
                    alloc = min(target_slot_size, cash, allowed_stock_cap)

                    if alloc >= 2_000_000 and buy_price > 0:
                        actual_cost_per_share = buy_price * (1.0 + BUY_FRICTION)
                        shares_to_buy = np.floor(alloc / actual_cost_per_share)
                        if shares_to_buy > 0:
                            total_spent = shares_to_buy * actual_cost_per_share
                            cash -= total_spent
                            allowed_stock_cap -= total_spent
                            held[c] = {
                                'qty': shares_to_buy,
                                'buy_price': buy_price,
                                'highest': buy_price,
                                'buy_date': date_str,
                                'hold_days': 0
                            }
                            trade_log.append({
                                'Date': date_str,
                                'Action': 'BUY',
                                'Ticker': c,
                                'Price': round(buy_price, 2),
                                'AI_Score': round(cand['Score'], 2),
                                'Vol_Ratio': round(cand['Vol_Ratio'], 1),
                                'RSI': round(cand['RSI'], 1),
                                'VN_MFI': round(market_mfi, 1)
                            })
                            empty_slots -= 1

    # Chốt NAV phiên cuối cùng
    final_stock_val = sum([
        held[c]['qty'] * stocks_close.iloc[-1][c]
        for c in held.keys()
        if c in stocks_close.columns and pd.notna(stocks_close.iloc[-1][c])
    ])
    final_nav = cash + final_stock_val

    df_nav = pd.DataFrame(daily_nav_history)
    df_trades = pd.DataFrame(trade_log)
    return df_trades, df_nav, final_nav, total_bank_interest

# ==========================================
# CHẠY CHIẾN LƯỢC & XUẤT BÁO CÁO TOÀN DIỆN
# ==========================================
df_trade_log, df_nav_hist, final_nav, total_interest = run_ai_quant_system()

# Ghi nhận file trading_log.csv cho Power BI
df_trade_log.to_csv('trading_log.csv', index=False)
print("✅ Đã ghi nhận lịch sử giao dịch vào file trading_log.csv (Sẵn sàng cho Power BI)!")

# ==========================================
# TÍNH TOÁN METRICS ĐỊNH LƯỢNG RỦI RO & HIỆU SUẤT
# ==========================================
initial_capital = 100_000_000.0
total_return_pct = ((final_nav / initial_capital) - 1.0) * 100.0

# Tính CAGR (Lợi nhuận kép theo năm)
n_years = (df_nav_hist['Date'].iloc[-1] - df_nav_hist['Date'].iloc[0]).days / 365.25
cagr = ((final_nav / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0

# Tính Maximum Drawdown (MDD)
df_nav_hist['Peak'] = df_nav_hist['NAV'].cummax()
df_nav_hist['Drawdown'] = (df_nav_hist['NAV'] - df_nav_hist['Peak']) / df_nav_hist['Peak']
max_mdd = df_nav_hist['Drawdown'].min() * 100.0

# Tìm đáy sụt giảm sâu nhất
trough_idx = df_nav_hist['Drawdown'].idxmin()
trough_date = df_nav_hist['Date'].iloc[trough_idx]
peak_sub = df_nav_hist.iloc[:trough_idx+1]
peak_date = peak_sub.loc[peak_sub['NAV'] == peak_sub['Peak'].max(), 'Date'].iloc[-1]

calmar_ratio = abs(cagr / max_mdd) if max_mdd != 0 else 0.0

sells = df_trade_log[df_trade_log['Action'].str.contains('SELL', na=False)] if not df_trade_log.empty else pd.DataFrame()
total_sells = len(sells)
win_sells = len(sells[sells['Profit(%)'] > 0]) if total_sells > 0 else 0
win_rate = (win_sells / total_sells * 100.0) if total_sells > 0 else 0.0
best_trade = sells['Profit(%)'].max() if total_sells > 0 else 0.0
worst_trade = sells['Profit(%)'].min() if total_sells > 0 else 0.0

print("\n" + "="*58)
print("📊 BÁO CÁO ĐỊNH LƯỢNG THỰC CHIẾN 10 NĂM (SURVIVORSHIP-FREE)")
print("="*58)
print(f"💵 Vốn khởi điểm (01/2014):        {initial_capital:,.0f} VNĐ")
print(f"🏦 TỔNG TÀI SẢN HIỆN TẠI (NAV):   {final_nav:,.0f} VNĐ")
print(f"💰 LÃI BANK THU VỀ (4%/năm):      +{total_interest:,.0f} VNĐ")
print(f"🚀 TỔNG LỢI NHUẬN (Đã trừ phí):    +{total_return_pct:.2f}%")
print(f"📈 Lợi nhuận kép TB (CAGR):        {cagr:.2f}% / năm")
print(f"----------------------------------------------------------")
print(f"📉 SỤT GIẢM TỐI ĐA (Max Drawdown): {max_mdd:.2f}%")
print(f"   (Thời kỳ chịu đựng lớn nhất: Từ đỉnh {peak_date.strftime('%d/%m/%Y')} về đáy {trough_date.strftime('%d/%m/%Y')})")
print(f"⚡ Chỉ số Calmar (CAGR / MDD):     {calmar_ratio:.2f}")
print(f"----------------------------------------------------------")
print(f"Tổng số lệnh đã chốt sổ:          {total_sells}")
print(f"🎯 Tỷ lệ thắng (Win Rate):         {win_rate:.2f}% (Thắng {win_sells}/{total_sells} trận)")
print(f"💰 Lệnh lãi đơn lẻ cao nhất:       +{best_trade:.2f}%")
print(f"🩸 Lệnh lỗ đơn lẻ sâu nhất:        {worst_trade:.2f}%")
print("="*58 + "\n")

# ==========================================
# 5. TELEGRAM RADAR BÁO CÁO HÀNG NGÀY
# ==========================================
print("5. Báo cáo Telegram Radar...")
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_msg(text_message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text_message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text_message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"⚠️ Gửi Telegram thất bại: {e}")

latest_date = df_clean['Date'].iloc[-1]
latest_date_str = latest_date.strftime('%Y-%m-%d')
active_today = get_vn30_constituents(latest_date_str)

msg = f"🤖 <b>SIÊU RADAR VN30 - BÁO CÁO THỰC CHIẾN</b> 🤖\n"
msg += f"📅 Ngày: <b>{latest_date.strftime('%d/%m/%Y')}</b>\n"
msg += f"🏛 Rổ VN30 hiệu lực: <b>{len(active_today)} mã chuẩn HOSE</b>\n"
msg += f"➖➖➖➖➖➖➖➖➖➖\n\n🎯 <b>TÍN HIỆU HÔM NAY:</b>\n"

buy_count, sell_count = 0, 0
if not df_trade_log.empty:
    latest_trades = df_trade_log[df_trade_log['Date'] == latest_date_str]
    if not latest_trades.empty:
        for _, row in latest_trades.iterrows():
            if 'BUY' in row['Action']:
                msg += f"🟢 <b>MUA: {row['Ticker']}</b> (Giá: {row['Price']:,.0f}) | AI: <b>{row['AI_Score'] * 100:.1f}%</b>\n\n"
                buy_count += 1
            elif 'SELL' in row['Action']:
                icon = "🔥" if row['Profit(%)'] > 0 else "🩸"
                msg += f"🔴 <b>BÁN: {row['Ticker']}</b> ({row['Action']}) | Lãi/Lỗ: {icon} <b>{row['Profit(%)']}%</b>\n\n"
                sell_count += 1

if buy_count == 0 and sell_count == 0:
    msg += "<i>⏳ Đứng ngoài quan sát, thị trường chưa có điểm nổ Alpha đạt chuẩn (Tiền mặt nhận lãi Bank 4%/năm).</i>\n\n"

msg += f"➖➖➖➖➖➖➖➖➖➖\n"
msg += f"🏦 NAV Hiện tại: <b>{final_nav:,.0f} VNĐ</b> | Max Drawdown: <b>{max_mdd:.2f}%</b>"

send_telegram_msg(msg)
print("✅ Hoàn tất chu trình quant_bot.py!")
