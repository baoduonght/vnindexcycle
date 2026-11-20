import pandas as pd
import numpy as np
try:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()
except Exception:
    tv = None
    Interval = None
import pandas_datareader.data as web
import yfinance as yf
import datetime
import requests

print("🚀 Khởi động Siêu Radar Vĩ mô: Bản Tối thượng (Mốc 2011-06 -> Nay | 100% Raw Data)...")

# Lấy dư từ 01/06/2011 để MA200 và Rolling Z-Score 120 phiên sẵn sàng ngay đầu năm 2012
START_FETCH_DATE = '2011-06-01'
start_date = pd.to_datetime(START_FETCH_DATE)
end_date = pd.to_datetime('today')
start_ts = int(start_date.timestamp())
end_ts = int(datetime.datetime.now().timestamp())

# MÀNG LỌC ĐẶC BIỆT: Triệt tiêu múi giờ (02:00:00) để chống lệch dòng
def clean_index(df):
    if df is not None and not df.empty:
        df = df.copy()
        df.index = pd.to_datetime(pd.to_datetime(df.index).strftime('%Y-%m-%d'))
        df = df[~df.index.duplicated(keep='last')]
    return df

# =====================================================================
# --- PHẦN 1: DỮ LIỆU CƠ SỞ & PHÁI SINH VIỆT NAM (HYBRID API) ---
# =====================================================================
print("1. Đang quét chứng khoán Việt Nam (TradingView + DChart Fallback)...")
local_df_list = []

def get_vietnam_data(symbol):
    if tv is not None and Interval is not None:
        try:
            exchange = 'HNX' if symbol == 'VN30F1M' else 'HOSE'
            raw = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=5000)
            if raw is not None and len(raw) > 100:
                return clean_index(raw)
        except Exception:
            pass

    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={start_ts}&to={end_ts}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if data.get('s') == 'ok' and len(data.get('t', [])) > 0:
                df = pd.DataFrame({
                    'Date': pd.to_datetime(data['t'], unit='s'),
                    'close': data['c'],
                    'volume': data.get('v', [np.nan] * len(data['t']))
                })
                df.set_index('Date', inplace=True)
                return clean_index(df)
    except Exception:
        pass
    return pd.DataFrame()

# 1. Kéo VNINDEX, VN30, Phái sinh
vni_raw = get_vietnam_data('VNINDEX')
if not vni_raw.empty:
    vni_df = vni_raw[['close', 'volume']].copy()
    vni_df.rename(columns={'close': 'VNINDEX', 'volume': 'VN_VOLUME'}, inplace=True)
    local_df_list.append(vni_df)

vn30_raw = get_vietnam_data('VN30')
if not vn30_raw.empty:
    vn30_df = vn30_raw[['close']].copy()
    vn30_df.rename(columns={'close': 'VN30'}, inplace=True)
    local_df_list.append(vn30_df)

ps_raw = get_vietnam_data('VN30F1M')
if not ps_raw.empty:
    ps_df = ps_raw[['close']].copy()
    ps_df.rename(columns={'close': 'VN30_F1M'}, inplace=True)
    local_df_list.append(ps_df)

if local_df_list:
    local_df = pd.concat(local_df_list, axis=1)
    if 'VN30_F1M' in local_df.columns and 'VN30' in local_df.columns:
        local_df['DERIVATIVE_SPREAD'] = local_df['VN30_F1M'] - local_df['VN30']
    local_df = local_df[~local_df.index.duplicated(keep='last')]
else:
    local_df = pd.DataFrame()

# =====================================================================
# --- PHẦN 2, 3, 4: KHỐI NGOẠI, VĨ MÔ TOÀN CẦU & FED ---
# =====================================================================
print("2. Đang kéo dữ liệu Dòng tiền, Vĩ mô Thế giới và Thanh khoản FED từ 2011-06...")

# Khối ngoại & Tự doanh
try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    for flow_type, url_endpoint in [('NET_FOREIGN_FLOW', 'foreign_trading'), ('NET_PROP_FLOW', 'proprietary_trading')]:
        url = f"https://finfo-api.vndirect.com.vn/v4/{url_endpoint}?q=code:VNINDEX~date:gte:{START_FETCH_DATE}&sort=date&size=5000"
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200 and 'data' in res.json() and res.json()['data']:
            df = pd.DataFrame(res.json()['data'])
            df[flow_type] = pd.to_numeric(df['buyVal'], errors='coerce') - pd.to_numeric(df['sellVal'], errors='coerce')
            df = df[['date', flow_type]].rename(columns={'date': 'Date'}).set_index('Date')
            df = clean_index(df)
            local_df = pd.concat([local_df, df[~df.index.duplicated(keep='last')]], axis=1)
except Exception:
    pass

# Vĩ mô toàn cầu (Yahoo Finance)
yf_symbols = {
    'DX-Y.NYB': 'DXY',
    '^TNX': 'US10Y',
    '^IRX': 'US2Y',
    'GC=F': 'GOLD',
    'CL=F': 'OIL',
    'JPY=X': 'JPY_USD',
    'VND=X': 'USD_VND',
    'CNY=X': 'USD_CNY',
    'VNQ': 'US_REAL_ESTATE'
}
try:
    raw_yf = yf.download(list(yf_symbols.keys()), start=START_FETCH_DATE, progress=False, auto_adjust=False)
    if raw_yf is not None and not raw_yf.empty:
        close_yf = raw_yf['Close'] if 'Close' in raw_yf else raw_yf
        global_df = close_yf.rename(columns=yf_symbols)
        global_df = clean_index(global_df)

        # CHUYỂN HỘ KHẨU TỶ GIÁ USD/VND VỀ VN & CHUẨN HÓA ĐƠN VỊ 2011-2013
        if 'USD_VND' in global_df.columns:
            usd_vnd_s = pd.to_numeric(global_df['USD_VND'], errors='coerce')
            usd_vnd_s = usd_vnd_s.apply(lambda x: x * 1000.0 if (pd.notna(x) and 10.0 <= x <= 40.0) else x)
            usd_vnd_s = usd_vnd_s.apply(lambda x: np.nan if (pd.notna(x) and (x < 15000.0 or x > 35000.0)) else x)
            local_df = pd.concat([local_df, usd_vnd_s.to_frame('USD_VND')], axis=1)
            global_df.drop(columns=['USD_VND'], inplace=True)
    else:
        global_df = pd.DataFrame()
except Exception:
    global_df = pd.DataFrame()

# VN10Y (Từ TradingView)
if tv is not None and Interval is not None:
    try:
        vn10y_raw = tv.get_hist(symbol='VN10Y', exchange='TVC', interval=Interval.in_daily, n_bars=5000)
        if vn10y_raw is not None and not vn10y_raw.empty:
            vn10y_df = vn10y_raw[['close']].rename(columns={'close': 'VN10Y'})
            vn10y_df = clean_index(vn10y_df)
            local_df = pd.concat([local_df, vn10y_df], axis=1)
    except Exception:
        pass

# FED Liquidity (100% Raw Data - Không dùng ffill/fillna)
try:
    fred_df = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start_date, end_date)
    fred_df['FED_LIQUIDITY'] = fred_df['WALCL'] - fred_df['WTREGEN'] - fred_df['RRPONTSYD']
    fred_df = fred_df[['FED_LIQUIDITY']]
    fred_df = clean_index(fred_df)
except Exception:
    fred_df = pd.DataFrame()

# =====================================================================
# --- PHẦN 5: HỢP NHẤT & GỌT RÁC ---
# =====================================================================
print("5. Đang xử lý hợp nhất và giữ trọn dữ liệu từ 2011-06-01...")
combo_df = pd.concat([local_df, global_df, fred_df], axis=1, join='outer')

# Giữ dữ liệu từ đúng 01/06/2011 trở đi (đủ vùng đệm cho toàn bộ năm 2012)
combo_df = combo_df[combo_df.index >= START_FETCH_DATE]

# Sắp xếp lại thời gian từ mới nhất xuống cũ nhất
combo_df = combo_df.apply(pd.to_numeric, errors='coerce').round(4).sort_index(ascending=False)
combo_df.index.name = 'Date'

# =====================================================================
# --- PHẦN 6: XUẤT XƯỞNG ---
# =====================================================================
print("6. Đang xuất CSV...")
def update_csv(filename, new_df):
    if not new_df.empty:
        new_df.to_csv(filename)
        print(f"✅ Đã đóng gói: {filename} ({len(new_df):,} dòng | {new_df.index.min().strftime('%Y-%m-%d')} -> {new_df.index.max().strftime('%Y-%m-%d')})")

# Xuất file tổng
update_csv('markets_database10years.csv', combo_df)
update_csv('markets_database_now.csv', combo_df.head(30))

# Xuất file VN (Vẫn giữ các ngày nghỉ lễ để đồng bộ mốc thời gian)
local_cols = local_df.columns.intersection(combo_df.columns)
if not local_cols.empty:
    local_df_clean = combo_df[local_cols]
    update_csv('vn_markets_database10years.csv', local_df_clean)

# Xuất file Vĩ mô Toàn cầu
global_cols = list(global_df.columns) + list(fred_df.columns)
global_cols = [c for c in global_cols if c in combo_df.columns]
if global_cols:
    global_macro_clean = combo_df[global_cols]
    update_csv('global_markets_database10years.csv', global_macro_clean)

print("🎉 HOÀN TẤT! Dữ liệu gốc 100% (không fill) đã phủ kín từ 06/2011 đến nay.")
