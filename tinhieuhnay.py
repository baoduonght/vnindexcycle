import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from tvDatafeed import TvDatafeed, Interval
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Lấy "chìa khóa" từ GitHub Secrets
BOT_TOKEN = os.environ.get('BOT_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

# Các hàm tính toán kỹ thuật
def calc_adx(high, low, close, n=14):
    up, down = high - high.shift(1), low.shift(1) - low
    pdm = np.where((up > down) & (up > 0), up, 0)
    ndm = np.where((down > up) & (down > 0), down, 0)
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = pd.Series(tr).rolling(n).mean()
    pdi = 100 * pd.Series(pdm).rolling(n).mean() / atr
    ndi = 100 * pd.Series(ndm).rolling(n).mean() / atr
    dx = 100 * abs(pdi - ndi) / (pdi + ndi)
    return dx.rolling(n).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    down = (-1 * delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

def main():
    tv = TvDatafeed()
    
    # 1. TẢI DỮ LIỆU & TÍNH MA VĨ MÔ
    print("Đang tải dữ liệu VN-Index...")
    vnindex_df = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=1500).reset_index()
    vnindex_df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'VNINDEX'}, inplace=True)
    vnindex_df['Date'] = pd.to_datetime(vnindex_df['Date']).dt.tz_localize(None).dt.normalize()
    
    vnindex_df['MA100'] = vnindex_df['VNINDEX'].rolling(100).mean()
    vnindex_df['MA200'] = vnindex_df['VNINDEX'].rolling(200).mean()
    vnindex_df['MA20_Month'] = vnindex_df['VNINDEX'].rolling(400).mean()
    vnindex_df['ADX'] = calc_adx(vnindex_df['High'], vnindex_df['Low'], vnindex_df['VNINDEX'], 14)
    df_clean = vnindex_df.dropna().copy()
    
    today_date = df_clean['Date'].iloc[-1]
    today_idx = df_clean['VNINDEX'].iloc[-1]
    today_ma100 = df_clean['MA100'].iloc[-1]
    today_ma200 = df_clean['MA200'].iloc[-1]
    today_ma400 = df_clean['MA20_Month'].iloc[-1]
    today_adx = df_clean['ADX'].iloc[-1]
    
    # 2. KIỂM TRA CHỐT CHẶN TRƯỚC KHI CHẠY AI
    if today_idx < today_ma100:
        send_telegram_msg(f"🚨 **CẢNH BÁO BÁN THÁO (VN-INDEX)** 🚨\n\n📉 Chỉ số: {today_idx:.1f}\n⚠️ ĐÃ GÃY MA100 ({today_ma100:.1f})\n🛑 KHUYẾN NGHỊ: BÁN SẠCH DANH MỤC, ÔM TIỀN MẶT.")
        return

    # 3. TẢI VN30 & HUẤN LUYỆN AI NHANH TRÊN SERVER
    print("Đang tải VN30 và đào tạo AI (Random Forest)...")
    stock_tickers = ['FPT', 'HPG', 'SSI', 'MWG', 'PNJ', 'TCB', 'MBB', 'VPB', 'ACB', 'VCB']
    stocks_close, stocks_volume = pd.DataFrame(), pd.DataFrame()

    for ticker in stock_tickers:
        df_t = tv.get_hist(symbol=ticker, exchange='HOSE', interval=Interval.in_daily, n_bars=1500)
        if df_t is not None and not df_t.empty:
            df_t = df_t.reset_index()
            df_t['Date'] = pd.to_datetime(df_t['datetime']).dt.tz_localize(None).dt.normalize()
            df_t.set_index('Date', inplace=True)
            if stocks_close.empty:
                stocks_close, stocks_volume = df_t[['close']].rename(columns={'close': ticker}), df_t[['volume']].rename(columns={'volume': f'{ticker}_V'})
            else:
                stocks_close = stocks_close.join(df_t[['close']].rename(columns={'close': ticker}), how='outer')
                stocks_volume = stocks_volume.join(df_t[['volume']].rename(columns={'volume': f'{ticker}_V'}), how='outer')

    stocks_close = stocks_close.ffill().bfill()
    stocks_volume = stocks_volume.ffill().fillna(0)
    stocks_high_20 = stocks_close.rolling(20).max().shift(1)
    stocks_vol_ma20 = stocks_volume.rolling(20).mean().shift(1)
    stocks_ma50 = stocks_close.rolling(50).mean()

    # Nhúng Macro
    macro_df = pd.DataFrame()
    for t, n in {'DX-Y.NYB': 'DXY', '^TNX': 'US10Y'}.items():
        temp = yf.download(t, start="2018-01-01", progress=False)['Close'].rename(n)
        macro_df = pd.DataFrame(temp) if macro_df.empty else macro_df.join(temp, how='outer')
    macro_df.index = macro_df.index.tz_localize(None).normalize()
    macro_df = macro_df.ffill().bfill()
    for col in ['DXY', 'US10Y']: macro_df[f'Z_{col}'] = (macro_df[col] - macro_df[col].rolling(120).mean()) / macro_df[col].rolling(120).std()
    
    df_ml = []
    for c in stock_tickers:
        df_c = pd.DataFrame({'Date': stocks_close.index, 'Ticker': c})
        df_c['Vol_Ratio'] = (stocks_volume[f'{c}_V'] / stocks_vol_ma20[f'{c}_V']).values
        df_c['Dist_MA50'] = (stocks_close[c] / stocks_ma50[c]).values
        df_c['RSI'] = calc_rsi(stocks_close[c]).values
        df_c['Target'] = ((stocks_close[c].shift(-10) / stocks_close[c] - 1) > 0.02).astype(int).values
        df_ml.append(df_c)

    ml_master = pd.concat(df_ml).merge(macro_df.reset_index(), on='Date', how='left').dropna()
    features = ['Vol_Ratio', 'Dist_MA50', 'RSI', 'Z_DXY', 'Z_US10Y']
    
    X_train = ml_master[ml_master['Date'] < '2024-01-01'][features]
    y_train = ml_master[ml_master['Date'] < '2024-01-01']['Target']
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)

    ml_master['AI_Score'] = rf_model.predict_proba(ml_master[features])[:, 1]
    ai_dict = {c: ml_master[ml_master['Ticker'] == c].set_index('Date')['AI_Score'] for c in stock_tickers}

    # 4. TÌM ĐIỂM NỔ (BREAKOUT) HÔM NAY
    buy_candidates = []
    msg_details = ""
    
    if today_idx > today_ma400 and today_adx > 22:
        msg_details += f"✅ Thị trường thuận lợi (VN-Index > MA20 Tháng, ADX: {today_adx:.1f})\n\n"
        for c in stock_tickers:
            if today_date in stocks_close.index:
                price = stocks_close.at[today_date, c]
                vol = stocks_volume.at[today_date, f'{c}_V']
                prev_h = stocks_high_20.at[today_date, c]
                prev_v = stocks_vol_ma20.at[today_date, f'{c}_V']
                
                if pd.isna(prev_h) or pd.isna(prev_v): continue
                if price > prev_h and vol > (prev_v * 1.5):
                    ai_score = ai_dict[c].get(today_date, 0)
                    if ai_score >= 0.55:
                        buy_candidates.append({'Mã': c, 'Giá': price, 'Vol': vol/prev_v, 'AI': ai_score*100})
    else:
        msg_details += f"⚠️ Sideway hoặc Kém thuận lợi (ADX: {today_adx:.1f}). Bot không mở mua.\n"

    final_msg = f"📊 **BÁO CÁO QUANT VN30 - {today_date.strftime('%d/%m/%Y')}** 📊\n\n"
    final_msg += f"🌐 VN-Index: {today_idx:.1f}\n🎯 MA100: {today_ma100:.1f} | MA400: {today_ma400:.1f}\n\n{msg_details}"
    
    if buy_candidates:
        final_msg += f"🚀 **CỔ PHIẾU NỔ VOL (Đã qua lọc AI):**\n"
        for s in buy_candidates: final_msg += f"• **{s['Mã']}**: Giá {s['Giá']:.1f} | Vol: {s['Vol']:.1f}x | AI: {s['AI']:.1f}%\n"
        
    send_telegram_msg(final_msg)

if __name__ == "__main__":
    main()
