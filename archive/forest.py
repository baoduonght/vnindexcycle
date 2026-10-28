!pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git -q
!pip install yfinance scikit-learn -q
import pandas as pd
import numpy as np
import yfinance as yf
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

# ==========================================
# 1. TẢI VN-INDEX & LẬP BỘ LỌC THỊ TRƯỜNG
# ==========================================
print("1. Đang tải Market Proxy (VNINDEX)...")
vnindex_df = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=4000).reset_index()
vnindex_df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'VNINDEX'}, inplace=True)
vnindex_df['Date'] = pd.to_datetime(vnindex_df['Date']).dt.tz_localize(None).dt.normalize()

vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50).mean()
vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100).mean()
vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200).mean()
vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
vnindex_df['ATR_PCT'] = calc_atr(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14) / vnindex_df['VNINDEX']
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

# ==========================================
# 3. NHÚNG AI - HUẤN LUYỆN RANDOM FOREST
# ==========================================
print("3. Thu thập dữ liệu Vĩ mô & Đào tạo AI nhận diện Bull Trap...")
macro_df = pd.DataFrame()
for ticker, name in {'DX-Y.NYB': 'DXY', '^TNX': 'US10Y'}.items():
    temp = yf.download(ticker, start="2012-01-01", progress=False)
    if isinstance(temp.columns, pd.MultiIndex): temp.columns = [c[0] for c in temp.columns]
    series = temp['Close'].rename(name)
    macro_df = pd.DataFrame(series) if macro_df.empty else macro_df.join(series, how='outer')

macro_df.index = macro_df.index.tz_localize(None).normalize()
macro_df.index.name = 'Date'
macro_df = macro_df.reset_index().ffill().bfill()

macro_cols = ['DXY', 'US10Y']
for col in macro_cols:
    macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()
z_cols = [f'Z_{c}' for c in macro_cols]

# Ghép nối ma trận Dữ liệu đa chiều (ML Dataset)
df_ml_list = []
for c in stock_tickers:
    df_c = pd.DataFrame({'Date': stocks_close.index})
    df_c['Ticker'] = c
    df_c['Vol_Ratio'] = (stocks_volume[f'{c}_V'] / stocks_vol_ma20[f'{c}_V']).values
    df_c['Dist_MA50'] = (stocks_close[c] / stocks_ma50[c]).values
    df_c['RSI'] = calc_rsi(stocks_close[c]).values

    # Target: Nắm giữ T+10 có lãi > 2% không?
    future_return = stocks_close[c].shift(-10) / stocks_close[c] - 1
    df_c['Target'] = (future_return > 0.02).astype(int).values
    df_ml_list.append(df_c)

ml_master = pd.concat(df_ml_list).merge(macro_df, on='Date', how='left').dropna()
ml_features = ['Vol_Ratio', 'Dist_MA50', 'RSI'] + z_cols

# Ép AI học trên tệp dữ liệu trước 2024 để tránh "nhìn lén tương lai"
X_train = ml_master[ml_master['Date'] < '2024-01-01'][ml_features]
y_train = ml_master[ml_master['Date'] < '2024-01-01']['Target']

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Điểm số AI (Probabilities) được lưu sẵn vào từ điển để tra cứu siêu tốc trong lúc Backtest
ml_master['AI_Score'] = rf_model.predict_proba(ml_master[ml_features])[:, 1]
ai_dict = {c: ml_master[ml_master['Ticker'] == c].set_index('Date')['AI_Score'] for c in stock_tickers}

# ==========================================
# 4. CHẠY BACKTEST (TÍCH HỢP AI CHẤM ĐIỂM)
# ==========================================
print("4. Chạy Sổ Kế Toán (Có sự giám sát của AI Model)...")
def run_ai_quant_system():
    backtest = df_clean[['Date', 'VNINDEX', 'MA50', 'MA100', 'MA200', 'ADX', 'ATR_PCT']].copy()
    MAX_SLOTS = 5
    LIQUIDITY_FEE = 0.003
    DAILY_INTEREST = 0.05 / 252
    cash = 1000000000
    held = {}
    nav_history = np.zeros(len(backtest))

    for i in range(len(backtest)):
        date = backtest['Date'].iloc[i]
        idx = backtest['VNINDEX'].iloc[i]
        market_ma50 = backtest['MA50'].iloc[i]
        market_ma100 = backtest['MA100'].iloc[i]
        market_ma200 = backtest['MA200'].iloc[i]
        market_adx = backtest['ADX'].iloc[i]
        market_atr_pct = backtest['ATR_PCT'].iloc[i]

        if date not in stocks_close.index:
            nav_history[i] = nav_history[i-1] if i > 0 else cash
            continue

        cash = cash * (1 + DAILY_INTEREST)

        if idx < market_ma100:
            for c in list(held.keys()):
                cash += held[c]['qty'] * stocks_close.at[date, c] * (1 - LIQUIDITY_FEE)
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
                cash += held[c]['qty'] * stocks_close.at[date, c] * (1 - LIQUIDITY_FEE)
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

                        # 1. Điều kiện Kỹ thuật: Nổ Volume vượt đỉnh
                        if curr_price > prev_high_20 and curr_vol > (prev_vol_ma20 * 1.5):
                            # 2. Điều kiện AI: Random Forest chấm điểm xác suất thành công > 55%
                            ai_score = ai_dict[c].get(date, 0)
                            if ai_score >= 0.55:
                                buy_candidates.append(c)

                if buy_candidates:
                    invest_amt = min(allowed_capital_to_invest / empty, cash)
                    for c in buy_candidates:
                        if invest_amt > 0 and stocks_close.at[date, c] > 0 and cash >= invest_amt:
                            cash -= invest_amt
                            qty = (invest_amt * (1 - LIQUIDITY_FEE)) / stocks_close.at[date, c]
                            held[c] = {'qty': qty, 'buy_price': stocks_close.at[date, c], 'highest': stocks_close.at[date, c]}
                            empty -= 1
                        if empty == 0 or cash <= 0: break

        nav_history[i] = cash + sum([held[c]['qty'] * stocks_close.at[date, c] for c in held.keys()])

    return nav_history

backtest_result = df_clean[['Date', 'VNINDEX']].copy()
backtest_result['NAV_AI_Driven'] = run_ai_quant_system()

print("5. Vẽ biểu đồ...")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=backtest_result['Date'], y=backtest_result['VNINDEX'], name="VN-Index", mode='lines', line=dict(color='#B2B5BE', width=1)), secondary_y=False)
fig.add_trace(go.Scatter(x=backtest_result['Date'], y=backtest_result['NAV_AI_Driven'], name="NAV (AI Filter + Volatility Sizing)", mode='lines', line=dict(color='#00E676', width=2.5)), secondary_y=True)
fig.update_layout(title='Đấu trường VN-Index: Hệ Thống AI Nhận Diện Breakout Giả', template='plotly_dark')
fig.show()
