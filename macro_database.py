import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import pandas_datareader.data as web
import datetime
import time
import requests

print("🚀 Khởi động Siêu Radar Vĩ mô: Bản Tối thượng (CORS Proxy Gateway)...")

tv = TvDatafeed()

# --- PHẦN 1: DỮ LIỆU CƠ SỞ & PHÁI SINH VIỆT NAM (TradingView) ---
print("1. Đang quét nội tại chứng khoán Việt Nam (VNINDEX, VN30, Phái sinh)...")
local_df_list = []

try:
    vn_raw = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=3000)
    if vn_raw is not None and not vn_raw.empty:
        vn_df = vn_raw[['close', 'volume']].copy()
        vn_df.rename(columns={'close': 'VNINDEX', 'volume': 'VN_VOLUME'}, inplace=True)
        local_df_list.append(vn_df)

    vn30_raw = tv.get_hist(symbol='VN30', exchange='HOSE', interval=Interval.in_daily, n_bars=3000)
    if vn30_raw is not None and not vn30_raw.empty:
        vn30_df = vn30_raw[['close']].copy()
        vn30_df.rename(columns={'close': 'VN30'}, inplace=True)
        local_df_list.append(vn30_df)

    ps_raw = tv.get_hist(symbol='VN30F1M', exchange='HNX', interval=Interval.in_daily, n_bars=3000)
    if ps_raw is not None and not ps_raw.empty:
        ps_df = ps_raw[['close']].copy()
        ps_df.rename(columns={'close': 'VN30_F1M'}, inplace=True)
        local_df_list.append(ps_df)

    if local_df_list:
        local_df = pd.concat(local_df_list, axis=1)
        if 'VN30_F1M' in local_df.columns and 'VN30' in local_df.columns:
            local_df['DERIVATIVE_SPREAD'] = local_df['VN30_F1M'] - local_df['VN30']
        local_df.index = local_df.index.normalize()
        local_df = local_df[~local_df.index.duplicated(keep='last')]
    else:
        local_df = pd.DataFrame()
except Exception as e:
    print(f"Lỗi hệ thống nội địa (TV): {e}")
    local_df = pd.DataFrame()

# --- PHẦN 2: TỰ GỌI API VNDIRECT - BẰNG TUYỆT CHIÊU PROXY GATEWAY ---
print("2. Đang dùng Trạm trung chuyển Proxy lách tường lửa VNDirect...")
try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Lấy Dòng tiền Khối ngoại qua trạm trung chuyển corsproxy
    url_foreign = "https://finfo-api.vndirect.com.vn/v4/foreign_trading?q=code:VNINDEX~date:gte:2021-01-01&sort=date&size=3000"
    bypass_url_f = f"https://corsproxy.io/?{url_foreign}"
    
    res_f = requests.get(bypass_url_f, headers=headers, timeout=15)
    
    if res_f.status_code == 200:
        f_data = res_f.json()['data']
        f_df = pd.DataFrame(f_data)
        f_df['NET_FOREIGN_FLOW'] = f_df['buyVal'] - f_df['sellVal']
        f_df = f_df[['date', 'NET_FOREIGN_FLOW']].copy()
        f_df.rename(columns={'date': 'Date'}, inplace=True)
        f_df.set_index('Date', inplace=True)
        f_df.index = pd.to_datetime(f_df.index).normalize()
        f_df = f_df[~f_df.index.duplicated(keep='last')]
        local_df = pd.concat([local_df, f_df], axis=1)
        print("  -> ✅ Lách tường lửa thành công: Đã gom Dòng tiền Khối ngoại!")
        
    # Lấy Dòng tiền Tự doanh qua trạm trung chuyển corsproxy
    url_prop = "https://finfo-api.vndirect.com.vn/v4/proprietary_trading?q=code:VNINDEX~date:gte:2021-01-01&sort=date&size=1500"
    bypass_url_p = f"https://corsproxy.io/?{url_prop}"
    
    res_p = requests.get(bypass_url_p, headers=headers, timeout=15)
    
    if res_p.status_code == 200 and 'data' in res_p.json() and len(res_p.json()['data']) > 0:
        p_data = res_p.json()['data']
        p_df = pd.DataFrame(p_data)
        p_df['NET_PROP_FLOW'] = p_df['buyVal'] - p_df['sellVal']
        p_df = p_df[['date', 'NET_PROP_FLOW']].copy()
        p_df.rename(columns={'date': 'Date'}, inplace=True)
        p_df.set_index('Date', inplace=True)
        p_df.index = pd.to_datetime(p_df.index).normalize()
        p_df = p_df[~p_df.index.duplicated(keep='last')]
        local_df = pd.concat([local_df, p_df], axis=1)
        print("  -> ✅ Lách tường lửa thành công: Đã gom Dòng tiền Tự doanh!")
        
except Exception as e:
    print(f"  -> ⚠️ Lỗi mạng hoặc trạm Proxy quá tải, tạm thời bỏ qua Nhóm 2: {e}")

# --- PHẦN 3: VỆ TINH VĨ MÔ TOÀN CẦU (TradingView) ---
tv_symbols = [
    ('VN10Y', 'TVC', 'VN10Y'),     
    ('DXY', 'TVC', 'DXY'),         
    ('US10Y', 'TVC', 'US10Y'),     
    ('US02Y', 'TVC', 'US2Y'),      
    ('USDJPY', 'FX_IDC', 'JPY_USD'), 
    ('USDVND', 'FX_IDC', 'USD_VND'), 
    ('USDCNH', 'FX_IDC', 'USD_CNH'), 
    ('XAUUSD', 'FX_IDC', 'GOLD'),    
    ('USOIL', 'TVC', 'OIL'),         
    ('VNQ', 'AMEX', 'US_REAL_ESTATE') 
]

print("3. Đang quét vệ tinh vĩ mô toàn cầu (Vàng, Dầu, Tỷ giá, BĐS)...")
df_list = []
for sym, exc, col_name in tv_symbols:
    try:
        data = tv.get_hist(symbol=sym, exchange=exc, interval=Interval.in_daily, n_bars=3000)
        if data is not None and not data.empty:
            temp_df = data[['close']].copy()
            temp_df.rename(columns={'close': col_name}, inplace=True)
            temp_df.index = temp_df.index.normalize()
            temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
            df_list.append(temp_df)
        time.sleep(0.5)
    except Exception as e:
        pass

if df_list:
    global_df = pd.concat(df_list, axis=1)
else:
    global_df = pd.DataFrame()

# --- PHẦN 4: HỆ THỐNG THANH KHOẢN FED (FRED) ---
print("4. Đang đột nhập hệ thống thanh khoản Mỹ (FRED)...")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=10*365)
try:
    fred_df = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start_date, end_date)
    fred_df['FED_LIQUIDITY'] = fred_df['WALCL'] - fred_df['WTREGEN'] - fred_df['RRPONTSYD']
    fred_df = fred_df[['FED_LIQUIDITY']]
    fred_df.index = fred_df.index.normalize()
    fred_df = fred_df[~fred_df.index.duplicated(keep='last')]
except Exception as e:
    print(f"Lỗi tải dữ liệu FRED: {e}")
    fred_df = pd.DataFrame()

# --- PHẦN 5: HỢP NHẤT VÀ DỌN DẸP LỖI ---
combo_df = pd.concat([local_df, global_df, fred_df], axis=1)
combo_df.sort_index(inplace=True)

# BỌC THÉP 2: Dọn rác an toàn - Chỉ check những cột ĐÃ CÓ trong file
check_cols = [c for c in ['VNINDEX', 'DXY', 'US10Y'] if c in combo_df.columns]
if check_cols:
    combo_df.dropna(subset=check_cols, how='all', inplace=True)
    
combo_df = combo_df.sort_index(ascending=False)
combo_df.index.name = 'Date'

# --- PHẦN 6: XUẤT XƯỞNG RA CÁC FILE CSV ---
def update_csv(filename, new_df):
    new_df.to_csv(filename)
    print(f"✅ Đã đóng gói: {filename}")

# Gom cột
us_cols = [col for _, _, col in tv_symbols if col not in ['VN10Y']]
if 'FED_LIQUIDITY' in combo_df.columns:
    us_cols.append('FED_LIQUIDITY')
us_cols_exist = [c for c in us_cols if c in combo_df.columns]

# Bổ sung 2 cột Tự doanh và Khối ngoại vào nhóm Việt Nam
vn_cols = ['VNINDEX', 'VN_VOLUME', 'VN30', 'VN30_F1M', 'DERIVATIVE_SPREAD', 'VN10Y', 'USD_VND', 'NET_FOREIGN_FLOW', 'NET_PROP_FLOW']
vn_cols_exist = [c for c in vn_cols if c in combo_df.columns]

update_csv('markets_database10years.csv', combo_df)
update_csv('global_markets_database10years.csv', combo_df[us_cols_exist])
update_csv('vn_markets_database10years.csv', combo_df[vn_cols_exist])
update_csv('markets_database_now.csv', combo_df.head(60))

print("🎉 HOÀN TẤT! Radar bọc thép lách tường lửa thành công.")
