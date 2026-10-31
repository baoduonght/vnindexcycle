import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import pandas_datareader.data as web
import yfinance as yf
import datetime
import requests

print("🚀 Khởi động Siêu Radar Vĩ mô: Bản Tối thượng 10 Năm (Fix 100% Dữ Liệu VN)...")

tv = TvDatafeed()

# --- Tính toán mốc thời gian 10 năm ---
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
    # Ưu tiên 1: Thử kéo bằng TvDatafeed (Nguồn quen thuộc và ổn định)
    try:
        exchange = 'HNX' if symbol == 'VN30F1M' else 'HOSE'
        raw = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=3000)
        if raw is not None and len(raw) > 100:
            return raw
    except Exception as e:
        print(f"TvDatafeed lỗi {symbol}: {e}")

    # Ưu tiên 2: Gọi TRỰC TIẾP API DChart của VNDirect (Bỏ qua Proxy)
    print(f"⚠️ Chuyển sang API dự phòng DChart cho {symbol}...")
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
    except Exception as e:
        print(f"DChart lỗi {symbol}: {e}")
        
    return pd.DataFrame()

# 1. Kéo VNINDEX
vni_raw = get_vietnam_data('VNINDEX')
if not vni_raw.empty:
    vni_df = vni_raw[['close', 'volume']].copy()
    vni_df.rename(columns={'close': 'VNINDEX', 'volume': 'VN_VOLUME'}, inplace=True)
    vni_df.index = pd.to_datetime(vni_df.index).normalize()
    local_df_list.append(vni_df)

# 2. Kéo VN30
vn30_raw = get_vietnam_data('VN30')
if not vn30_raw.empty:
    vn30_df = vn30_raw[['close']].copy()
    vn30_df.rename(columns={'close': 'VN30'}, inplace=True)
    vn30_df.index = pd.to_datetime(vn30_df.index).normalize()
    local_df_list.append(vn30_df)

# 3. Kéo Phái sinh
ps_raw = get_vietnam_data('VN30F1M')
if not ps_raw.empty:
    ps_df = ps_raw[['close']].copy()
    ps_df.rename(columns={'close': 'VN30_F1M'}, inplace=True)
    ps_df.index = pd.to_datetime(ps_df.index).normalize()
    local_df_list.append(ps_df)

# Hợp nhất nội địa
if local_df_list:
    local_df = pd.concat(local_df_list, axis=1)
    if 'VN30_F1M' in local_df.columns and 'VN30' in local_df.columns:
        local_df['DERIVATIVE_SPREAD'] = local_df['VN30_F1M'] - local_df['VN30']
    local_df = local_df[~local_df.index.duplicated(keep='last')]
else:
    local_df = pd.DataFrame()

# =====================================================================
# --- PHẦN 2: API VNDIRECT (KHỐI NGOẠI & TỰ DOANH) ---
# =====================================================================
print("2. Đang kéo Dòng tiền (Đã loại bỏ Proxy gây nghẽn)...")
try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Khối ngoại 10 năm
    url_f = "https://finfo-api.vndirect.com.vn/v4/foreign_trading?q=code:VNINDEX~date:gte:2014-01-01&sort=date&size=3000"
    res_f = requests.get(url_f, headers=headers, timeout=15)
    if res_f.status_code == 200 and 'data' in res_f.json():
        f_df = pd.DataFrame(res_f.json()['data'])
        f_df['NET_FOREIGN_FLOW'] = f_df['buyVal'] - f_df['sellVal']
        f_df = f_df[['date', 'NET_FOREIGN_FLOW']].rename(columns={'date': 'Date'}).set_index('Date')
        f_df.index = pd.to_datetime(f_df.index).normalize()
        local_df = pd.concat([local_df, f_df[~f_df.index.duplicated(keep='last')]], axis=1)
        
    # Tự doanh 10 năm
    url_p = "https://finfo-api.vndirect.com.vn/v4/proprietary_trading?q=code:VNINDEX~date:gte:2014-01-01&sort=date&size=3000"
    res_p = requests.get(url_p, headers=headers, timeout=15)
    if res_p.status_code == 200 and 'data' in res_p.json() and res_p.json()['data']:
        p_df = pd.DataFrame(res_p.json()['data'])
        p_df['NET_PROP_FLOW'] = p_df['buyVal'] - p_df['sellVal']
        p_df = p_df[['date', 'NET_PROP_FLOW']].rename(columns={'date': 'Date'}).set_index('Date')
        p_df.index = pd.to_datetime(p_df.index).normalize()
        local_df = pd.concat([local_df, p_df[~p_df.index.duplicated(keep='last')]], axis=1)
except Exception as e:
    print(f"Lỗi phần Dòng tiền VNDirect: {e}")

# =====================================================================
# --- PHẦN 3: VỆ TINH VĨ MÔ TOÀN CẦU (YAHOO FINANCE) ---
# =====================================================================
print("3. Đang đột nhập hệ thống Yahoo Finance (Vàng, Dầu, Tỷ giá, Lợi suất)...")
yf_symbols = {
    'DX-Y.NYB': 'DXY',
    '^TNX': 'US10Y',
    'GC=F': 'GOLD',
    'CL=F': 'OIL',
    'JPY=X': 'JPY_USD',
    'VND=X': 'USD_VND',
    'CNH=X': 'USD_CNH',
    'VNQ': 'US_REAL_ESTATE'
}
try:
    yf_data = yf.download(list(yf_symbols.keys()), start=start_date.strftime('%Y-%m-%d'), progress=False)
    global_df = yf_data['Close'].rename(columns=yf_symbols)
    
    # Riêng VN10Y không có trên Yahoo, kéo tạm từ TV
    vn10y_raw = tv.get_hist(symbol='VN10Y', exchange='TVC', interval=Interval.in_daily, n_bars=3000)
    if vn10y_raw is not None and not vn10y_raw.empty:
        global_df['VN10Y'] = vn10y_raw['close']
        
    global_df.index = pd.to_datetime(global_df.index).tz_localize(None).normalize()
    global_df = global_df[~global_df.index.duplicated(keep='last')]
except Exception as e:
    print(f"Lỗi Yahoo Finance: {e}")
    global_df = pd.DataFrame()

# =====================================================================
# --- PHẦN 4: HỆ THỐNG THANH KHOẢN FED (FRED) ---
# =====================================================================
print("4. Đang đột nhập hệ thống thanh khoản Mỹ (FRED)...")
try:
    fred_df = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start_date, end_date)
    fred_df['FED_LIQUIDITY'] = fred_df['WALCL'] - fred_df['WTREGEN'] - fred_df['RRPONTSYD']
    fred_df = fred_df[['FED_LIQUIDITY']]
    fred_df.index = pd.to_datetime(fred_df.index).normalize()
    fred_df = fred_df[~fred_df.index.duplicated(keep='last')]
except Exception as e:
    print(f"Lỗi FRED: {e}")
    fred_df = pd.DataFrame()

# =====================================================================
# --- PHẦN 5: HỢP NHẤT TỔNG THỂ ---
# =====================================================================
print("5. Đang đồng bộ hóa toàn bộ mốc thời gian...")
combo_df = pd.concat([local_df, global_df, fred_df], axis=1, join='outer')
combo_df = combo_df.sort_index(ascending=False)
combo_df.index.name = 'Date'

# =====================================================================
# --- PHẦN 6: XUẤT XƯỞNG ---
# =====================================================================
print("6. Đang đóng gói dữ liệu và xuất CSV...")
def update_csv(filename, new_df):
    if not new_df.empty:
        new_df.to_csv(filename)
        print(f"✅ Đã đóng gói: {filename} (Tổng số dòng: {len(new_df)})")
    else:
        print(f"⚠️ Cảnh báo: Dữ liệu rỗng, bỏ qua {filename}")

update_csv('markets_database10years.csv', combo_df)
update_csv('markets_database_now.csv', combo_df.head(30))

if not local_df.empty:
    local_df_sorted = local_df.sort_index(ascending=False)
    update_csv('vn_markets_database10years.csv', local_df_sorted)

if not global_df.empty or not fred_df.empty:
    global_macro_df = pd.concat([global_df, fred_df], axis=1, join='outer').sort_index(ascending=False)
    global_macro_df.index.name = 'Date'
    update_csv('global_markets_database10years.csv', global_macro_df)

print("🎉 HOÀN TẤT! Data Pipeline 10 năm đã thông luồng và phân tách chuẩn xác.")
