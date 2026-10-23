"""
====================================================================
MACRO RADAR QUANT SYSTEM V1.0 (TREND + MACRO + RS + RISK MANAGEMENT)
Tác giả: quocbao21345 & AI Collaborator
Ngày hoàn thiện: 10/2026
====================================================================
"""
!pip install yfinance plotly imbalanced-learn scikit-learn requests pandas-datareader pandas_ta -q
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pandas_datareader.data as web
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("1. Đang tải dữ liệu VN-Index & Tính toán Chỉ báo (ADX, MA50, MA200)...")
url = "https://services.entrade.com.vn/chart-api/v2/ohlcs/index?resolution=1D&symbol=VNINDEX&from=1420070400&to=2000000000"
res = requests.get(url).json()
vnindex_df = pd.DataFrame({
    'Date': pd.to_datetime(res['t'], unit='s'),
    'Open': res['o'], 'High': res['h'], 'Low': res['l'], 'VNINDEX': res['c']
})
vnindex_df['Date'] = vnindex_df['Date'].dt.tz_localize(None).dt.normalize()

vnindex_df['ADX'] = ta.adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], length=14)['ADX_14']
vnindex_df['MA50'] = vnindex_df['VNINDEX'].rolling(50).mean()
vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200).mean()

print("2. Đang tải dữ liệu Vĩ mô Quốc tế & Bơm hút tiền FED...")
tickers_macro = {'DX-Y.NYB': 'DXY', '^TNX': 'US10Y', 'GC=F': 'GOLD', 'CL=F': 'OIL', 'VND=X': 'USD_VND'}
macro_df = pd.DataFrame()
for ticker, name in tickers_macro.items():
    temp = yf.download(ticker, start="2015-01-01", progress=False)
    if isinstance(temp.columns, pd.MultiIndex): temp.columns = temp.columns.droplevel(1)
    series = temp['Close'].rename(name)
    macro_df = pd.DataFrame(series) if macro_df.empty else macro_df.join(series, how='outer')

macro_df.index = macro_df.index.tz_localize(None).normalize()
macro_df.index.name = 'Date'
macro_df = macro_df.reset_index()

fed_df = web.DataReader('WALCL', 'fred', '2015-01-01', '2026-10-24').reset_index()
fed_df.columns = ['Date', 'FED_LIQUIDITY']
macro_df = pd.merge(macro_df, fed_df, on='Date', how='outer').sort_values('Date').ffill()

print("3. Đang tải Rổ Cổ phiếu VN30 & Tính sức mạnh giá (RS 90 ngày)...")
stock_tickers = ['FPT.VN', 'HPG.VN', 'SSI.VN', 'MWG.VN', 'PNJ.VN', 'TCB.VN', 'MBB.VN', 'VPB.VN', 'ACB.VN', 'VCB.VN']
stocks_df = yf.download(stock_tickers, start="2015-01-01", progress=False)['Close']
stocks_df.index = stocks_df.index.tz_localize(None).normalize()
stocks_df.index.name = 'Date'
stocks_df = stocks_df.ffill().bfill()

stock_returns = stocks_df.pct_change().fillna(0)
rs_90 = stocks_df.pct_change(periods=90).fillna(0) 

# ==========================================
# 4. CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN AI VĨ MÔ
# ==========================================
print("4. Đang huấn luyện Trí tuệ Nhân tạo (Random Forest)...")
df = pd.merge(vnindex_df, macro_df, on='Date', how='left')
df = df.dropna(subset=['VNINDEX']).reset_index(drop=True)
df = df.ffill().bfill() 

# Z-score hóa Vĩ mô (Trí nhớ 120 ngày)
macro_cols = ['DXY', 'US10Y', 'GOLD', 'OIL', 'USD_VND', 'FED_LIQUIDITY']
for col in macro_cols:
    df[f'Z_{col}'] = (df[col] - df[col].rolling(120).mean()) / df[col].rolling(120).std()
z_cols = [f'Z_{c}' for c in macro_cols]

# Nhãn học (Target): Uptrend = VNINDEX tăng > 2% sau 60 ngày
df['Future_Return'] = df['VNINDEX'].shift(-60) / df['VNINDEX'] - 1
df['Target'] = np.where(df['Future_Return'] > 0.02, 1, 0)
df_clean = df.dropna().copy()

split_date = '2025-06-01'
train_df = df_clean[df_clean['Date'] < split_date]
X_train, y_train = train_df[z_cols], train_df['Target']
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train_smote, y_train_smote)

# Cân bằng Timeline cho Cổ phiếu
stocks_df = stocks_df.reindex(df_clean['Date']).ffill()
stock_returns = stock_returns.reindex(df_clean['Date']).fillna(0)
rs_90 = rs_90.reindex(df_clean['Date']).fillna(0)

# ==========================================
# 5. VÒNG LẶP BACKTEST & QUẢN TRỊ RỦI RO DANH MỤC
# ==========================================
print("5. Đang chạy Backtest với Luật bảo vệ Kép (Hệ thống & Từng mã)...")
backtest_all = df_clean[['Date', 'VNINDEX', 'MA50', 'MA200', 'ADX']].copy()
backtest_all['Market_Return'] = backtest_all['VNINDEX'].pct_change().fillna(0)
backtest_all['Macro_Score'] = model.predict_proba(df_clean[z_cols])[:, 1]

MAX_SLOTS = 5            # Luật 20%: Tối đa 5 mã
SLOT_WEIGHT = 1.0 / MAX_SLOTS
INDIVIDUAL_STOP = 0.08   # Cắt lỗ độc lập 8% từ đỉnh

held_stocks = {}         
portfolio_returns = []
selected_stocks_log = []

for i in range(len(backtest_all)):
    date = backtest_all['Date'].iloc[i]
    price_idx = backtest_all['VNINDEX'].iloc[i]
    ma50 = backtest_all['MA50'].iloc[i]
    ma200 = backtest_all['MA200'].iloc[i]
    adx = backtest_all['ADX'].iloc[i]
    macro = backtest_all['Macro_Score'].iloc[i]
    
    # [A] KIỂM TRA RỦI RO HỆ THỐNG (BÁN SẠCH)
    if price_idx < ma200 or (price_idx < ma50 and macro < 0.4):
        held_stocks.clear() 
        
    else:
        # [B] KIỂM TRA RỦI RO TỪNG MÃ (CẮT LỖ ĐỘC LẬP CHỐNG DRAMA)
        to_remove = []
        for ticker in held_stocks.keys():
            current_price = stocks_df.loc[date, ticker]
            held_stocks[ticker] = max(held_stocks[ticker], current_price)
            if current_price < held_stocks[ticker] * (1 - INDIVIDUAL_STOP):
                to_remove.append(ticker)
                
        for ticker in to_remove:
            del held_stocks[ticker] 
            
        # [C] GIẢI NGÂN / TRÁM CHỖ (KHI VĨ MÔ & TREND ĐỀU ỦNG HỘ)
        if price_idx > ma50 and adx > 20 and macro >= 0.5:
            empty_slots = MAX_SLOTS - len(held_stocks)
            if empty_slots > 0:
                yesterday_rs = rs_90.iloc[i-1] if i > 0 else rs_90.iloc[0]
                top_candidates = yesterday_rs.sort_values(ascending=False).index.tolist()
                
                for candidate in top_candidates:
                    if candidate not in held_stocks:
                        held_stocks[candidate] = stocks_df.loc[date, candidate]
                        empty_slots -= 1
                    if empty_slots == 0:
                        break

    # [D] TÍNH TOÁN HIỆU SUẤT TRONG NGÀY
    if len(held_stocks) > 0:
        day_ret = sum([stock_returns.loc[date, t] for t in held_stocks.keys()]) * SLOT_WEIGHT
    else:
        day_ret = 0.0
        
    portfolio_returns.append(day_ret)
    selected_stocks_log.append(", ".join(held_stocks.keys()) if held_stocks else "CASH")

backtest_all['Portfolio_Return'] = portfolio_returns
backtest_all['Stock_Picked'] = selected_stocks_log

initial_capital = 100000000
backtest_all['Market_NAV'] = initial_capital * (1 + backtest_all['Market_Return']).cumprod()
backtest_all['Quant_Fund_NAV'] = initial_capital * (1 + backtest_all['Portfolio_Return']).cumprod()

print("\n=== HOÀN TẤT! HỆ THỐNG ĐÃ SẴN SÀNG ===")

# ==========================================
# 6. VẼ BIỂU ĐỒ 
# ==========================================
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=backtest_all['Date'], y=backtest_all['VNINDEX'], name="VN-Index (Thị trường chung)", mode='lines', line=dict(color='#B2B5BE', width=1.5)), secondary_y=False)
fig.add_trace(go.Scatter(x=backtest_all['Date'], y=backtest_all['Quant_Fund_NAV'], name="NAV Định Lượng (Master V1.0)", mode='lines', line=dict(color='#00E676', width=2.5)), secondary_y=True)

fig.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="red")
fig.add_annotation(x=split_date, y=0.95, yref="paper", text=" Bắt đầu Trade Thực Chiến", showarrow=False, font=dict(color="red", size=12), xanchor="left")

fig.update_layout(
    title='Macro Radar V1.0 (Trend + Macro AI + ADX Filter + RS Stock Picking)',
    template='plotly_dark', plot_bgcolor='#131722', paper_bgcolor='#131722',
    hovermode='x unified', margin=dict(l=20, r=20, t=50, b=20)
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#363c4e')
fig.update_yaxes(title_text="Điểm VN-Index", secondary_y=False, showgrid=True, gridwidth=1, gridcolor='#363c4e')
fig.update_yaxes(title_text="Giá trị tài sản (VND)", secondary_y=True, showgrid=False)
fig.show()
