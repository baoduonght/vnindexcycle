import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import pandas_datareader.data as web
import yfinance as yf
import datetime
import requests

print("🚀 Khởi động Siêu Radar Vĩ mô: Bản Tối thượng (Giữ nguyên NaN thực tế)...")

tv = TvDatafeed()
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=10*365)
start_ts = int(start_date.timestamp())
end_ts = int(end_date.timestamp())

# =====================================================================
# --- PHẦN 1: DỮ LIỆU CƠ SỞ & PHÁI SINH VIỆT NAM (HYBRID API) ---
# =====================================================================
print("1. Đang quét chứng khoán Việt Nam (TradingView + DChart Fallback)...")
local_df_list = []

def get_vietnam_data(symbol):
    try:
        exchange = 'HNX' if symbol == 'VN30F1M' else 'HOSE'
        raw = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=3000)
        if raw is not None and len(raw) > 100:
            return raw
    except Exception:
        pass

    url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={start_ts}&to={end_ts}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if data.get('s') == 'ok':
                df = pd.DataFrame({
                    'Date': pd.to_datetime(data['t'], unit='s').dt.normalize(),
                    'close': data['c'],
                    'volume': data.get('v', 0)
                })
                df.set_index('Date', inplace=True)
                return df
    except Exception:
        pass
    return pd.DataFrame()

# 1. Kéo VNINDEX
vni_raw = get_vietnam_data('VNINDEX')
if not vni_raw.empty:
    vni_df = vni_raw[['close', 'volume']].copy()
    vni_df.rename(columns={'close': 'VNINDEX', 'volume': 'VN_VOLUME'}, inplace=True)
    local_df_list.append(vni_df)

# 2. Kéo VN30
vn30_raw = get_vietnam_data('VN30')
if not vn30_raw.empty:
    vn30_df = vn30_raw[['close']].copy()
    vn30_df.rename(columns={'close': 'VN30'}, inplace=True)
    local_df_list.append(vn30_df)

# 3. Kéo Phái sinh
ps_raw = get_vietnam_data('VN30F1M')
if not ps_raw.empty:
    ps_df = ps_raw[['close']].copy()
    ps_df.rename(columns={'close': 'VN30_F1M'}, inplace=True)
    local_df_list.append(ps_df)

# Tiến hành hợp nhất
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
print("2. Đang kéo dữ liệu Dòng tiền nội địa, Vĩ mô Thế giới và Thanh khoản FED...")

# Khối ngoại & Tự doanh
try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    for flow_type, url_endpoint in [('NET_FOREIGN_FLOW', 'foreign_trading'), ('NET_PROP_FLOW', 'proprietary_trading')]:
        url = f"https://finfo-api.vndirect.com.vn/v4/{url_endpoint}?q=code:VNINDEX~date:gte:2014-01-01&sort=date&size=3000"
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200 and 'data' in res.json() and res.json()['data']:
            df = pd.DataFrame(res.json()['data'])
            df[flow_type] = df['buyVal'] - df['sellVal']
            df = df[['date', flow_type]].rename(columns={'date': 'Date'}).set_index('Date')
            df.index = pd.to_datetime(df.index).normalize()
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
    global_df = yf.download(list(yf_symbols.keys()), start=start_date.strftime('%Y-%m-%d'), progress=False)['Close'].rename(columns=yf_symbols)
    global_df.index = pd.to_datetime(global_df.index).tz_localize(None).normalize()
    global_df = global_df[~global_df.index.duplicated(keep='last')]
except Exception:
    global_df = pd.DataFrame()

# VN10Y (Từ TradingView)
try:
    vn10y_raw = tv.get_hist(symbol='VN10Y', exchange='TVC', interval=Interval.in_daily, n_bars=3000)
    if vn10y_raw is not None and not vn10y_raw.empty:
        vn10y_df = vn10y_raw[['close']].rename(columns={'close': 'VN10Y'})
        vn10y_df.index = pd.to_datetime(vn10y_df.index).normalize()
        local_df = pd.concat([local_df, vn10y_df], axis=1)
except Exception:
    pass

# FED Liquidity
try:
    fred_df = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start_date, end_date)
    fred_df['FED_LIQUIDITY'] = fred_df['WALCL'] - fred_df['WTREGEN'] - fred_df['RRPONTSYD']
    fred_df = fred_df[['FED_LIQUIDITY']]
    fred_df.index = pd.to_datetime(fred_df.index).normalize()
    fred_df = fred_df[~fred_df.index.duplicated(keep='last')]
except Exception:
    fred_df = pd.DataFrame()


# =====================================================================
# --- PHẦN 5: HỢP NHẤT ---
# =====================================================================
print("5. Đang xử lý hợp nhất (Không tự ý điền dữ liệu ảo)...")
combo_df = pd.concat([local_df, global_df, fred_df], axis=1, join='outer')

# Chỉ làm tròn 4 chữ số thập phân để giảm dung lượng file
combo_df = combo_df.round(4) 

combo_df = combo_df.sort_index(ascending=False)
combo_df.index.name = 'Date'

# =====================================================================
# --- PHẦN 6: XUẤT XƯỞNG ---
# =====================================================================
print("6. Đang xuất CSV...")
def update_csv(filename, new_df):
    if not new_df.empty:
        new_df.to_csv(filename)
        print(f"✅ Đã đóng gói: {filename}")

update_csv('markets_database10years.csv', combo_df)
update_csv('markets_database_now.csv', combo_df.head(30))

if not local_df.empty:
    local_df_sorted = local_df.round(4).sort_index(ascending=False)
    update_csv('vn_markets_database10years.csv', local_df_sorted)

if not global_df.empty or not fred_df.empty:
    global_macro_df = pd.concat([global_df, fred_df], axis=1, join='outer')
    global_macro_df = global_macro_df.round(4).sort_index(ascending=False)
    global_macro_df.index.name = 'Date'
    update_csv('global_markets_database10years.csv', global_macro_df)

print("🎉 HOÀN TẤT! Pipeline sạch 100%, giữ nguyên các ô trống (NaN).")
