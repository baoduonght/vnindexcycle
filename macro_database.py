import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import pandas_datareader.data as web
import yfinance as yf
import datetime
import time
import requests

print("🚀 Khởi động Siêu Radar Vĩ mô: Bản Tối thượng 10 Năm (YFinance + Proxy)...")

# Khởi tạo chế độ Guest bình thường để kéo Phái sinh (không nhập pass nữa)
tv = TvDatafeed()

# --- PHẦN 1: DỮ LIỆU CƠ SỞ & PHÁI SINH VIỆT NAM ---
print("1. Đang quét chứng khoán Việt Nam (VNINDEX, VN30, Phái sinh)...")
local_df_list = []

try:
    # Dùng yfinance lấy mốc 10 năm cho VNINDEX và VN30
    vn_tickers = yf.download(['^VNINDEX.VN', '^VN30.VN'], start="2016-01-01", progress=False)
    
    if not vn_tickers.empty:
        # Xử lý VNINDEX
        vn_df = pd.DataFrame({
            'VNINDEX': vn_tickers['Close']['^VNINDEX.VN'],
            'VN_VOLUME': vn_tickers['Volume']['^VNINDEX.VN']
        })
        local_df_list.append(vn_df)
        
        # Xử lý VN30
        vn30_df = pd.DataFrame({'VN30': vn_tickers['Close']['^VN30.VN']})
        local_df_list.append(vn30_df)

    # Phái sinh VN30F1M vẫn dùng TradingView (dữ liệu phái sinh VN chỉ có từ cuối 2017)
    ps_raw = tv.get_hist(symbol='VN30F1M', exchange='HNX', interval=Interval.in_daily, n_bars=3000)
    if ps_raw is not None and not ps_raw.empty:
        ps_df = ps_raw[['close']].copy()
        ps_df.rename(columns={'close': 'VN30_F1M'}, inplace=True)
        local_df_list.append(ps_df)

    if local_df_list:
        local_df = pd.concat(local_df_list, axis=1)
        if 'VN30_F1M' in local_df.columns and 'VN30' in local_df.columns:
            local_df['DERIVATIVE_SPREAD'] = local_df['VN30_F1M'] - local_df['VN30']
        local_df.index = pd.to_datetime(local_df.index).tz_localize(None).normalize()
        local_df = local_df[~local_df.index.duplicated(keep='last')]
    else:
        local_df = pd.DataFrame()
except Exception as e:
    print(f"Lỗi hệ thống nội địa: {e}")
    local_df = pd.DataFrame()

# --- PHẦN 2: API VNDIRECT (KHỐI NGOẠI & TỰ DOANH) ---
print("2. Đang kéo Dòng tiền từ Trạm trung chuyển Proxy...")
try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Khối ngoại 10 năm
    url_f = "https://finfo-api.vndirect.com.vn/v4/foreign_trading?q=code:VNINDEX~date:gte:2016-01-01&sort=date&size=3000"
    res_f = requests.get(f"https://corsproxy.io/?{url_f}", headers=headers, timeout=15)
    if res_f.status_code == 200:
        f_df = pd.DataFrame(res_f.json()['data'])
        f_df['NET_FOREIGN_FLOW'] = f_df['buyVal'] - f_df['sellVal']
        f_df = f_df[['date', 'NET_FOREIGN_FLOW']].rename(columns={'date': 'Date'}).set_index('Date')
        f_df.index = pd.to_datetime(f_df.index).normalize()
        local_df = pd.concat([local_df, f_df[~f_df.index.duplicated(keep='last')]], axis=1)
        
    # Tự doanh 10 năm
    url_p = "https://finfo-api.vndirect.com.vn/v4/proprietary_trading?q=code:VNINDEX~date:gte:2016-01-01&sort=date&size=3000"
    res_p = requests.get(f"https://corsproxy.io/?{url_p}", headers=headers, timeout=15)
    if res_p.status_code == 200 and 'data' in res_p.json() and res_p.json()['data']:
        p_df = pd.DataFrame(res_p.json()['data'])
        p_df['NET_PROP_FLOW'] = p_df['buyVal'] - p_df['sellVal']
        p_df = p_df[['date', 'NET_PROP_FLOW']].rename(columns={'date': 'Date'}).set_index('Date')
        p_df.index = pd.to_datetime(p_df.index).normalize()
        local_df = pd.concat([local_df, p_df[~p_df.index.duplicated(keep='last')]], axis=1)
        
except Exception as e:
    print(f"Lỗi phần VNDirect: {e}")

# --- PHẦN 3: VỆ TINH VĨ MÔ TOÀN CẦU (YAHOO FINANCE) ---
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
    yf_data = yf.download(list(yf_symbols.keys()), start="2016-01-01", progress=False)
    global_df = yf_data['Close'].rename(columns=yf_symbols)
    # Riêng VN10Y không có trên Yahoo, kéo tạm từ TV
    vn10y_raw = tv.get_hist(symbol='VN10Y', exchange='TVC', interval=Interval.in_daily, n_bars=3000)
    if vn10y_raw is not None:
        global_df['VN10Y'] = vn10y_raw['close']
        
    global_df.index = pd.to_datetime(global_df.index).tz_localize(None).normalize()
    global_df = global_df[~global_df.index.duplicated(keep='last')]
except Exception as e:
    print(f"Lỗi Yahoo Finance: {e}")
    global_df = pd.DataFrame()

# --- PHẦN 4: HỆ THỐNG THANH KHOẢN FED (FRED) ---
print("4. Đang đột nhập hệ thống thanh khoản Mỹ (FRED)...")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=10*365)
try:
    fred_df = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start_date, end_date)
    fred_df['FED_LIQUIDITY'] = fred_df['WALCL'] - fred_df['WTREGEN'] - fred_df['RRPONTSYD']
    fred_df = fred_df[['FED_LIQUIDITY']]
    fred_df.index = pd.to_datetime(fred_df.index).normalize()
    fred_df = fred_df[~fred_df.index.duplicated(keep='last')]
except Exception as e:
    print(f"Lỗi FRED: {e}")
    fred_df = pd.DataFrame()

# --- PHẦN 5: HỢP NHẤT ---
combo_df = pd.concat([local_df, global_df, fred_df], axis=1)
combo_df = combo_df.sort_index(ascending=False)
combo_df.index.name = 'Date'

# --- PHẦN 6: XUẤT XƯỞNG ---
def update_csv(filename, new_df):
    if not new_df.empty:
        new_df.to_csv(filename)
        print(f"✅ Đã đóng gói: {filename}")
    else:
        print(f"⚠️ Cảnh báo: Dữ liệu rỗng, bỏ qua {filename}")

# 1. File tổng hợp (Tất cả dữ liệu VN, Global, FED)
update_csv('markets_database10years.csv', combo_df)

# 2. File cập nhật nhanh (Chỉ cắt lấy dữ liệu 30 ngày gần nhất để load nhẹ hơn)
update_csv('markets_database_now.csv', combo_df.head(30))

# 3. File nội địa (Chỉ lưu các cột của thị trường Việt Nam)
if not local_df.empty:
    local_df_sorted = local_df.sort_index(ascending=False)
    update_csv('vn_markets_database10years.csv', local_df_sorted)

# 4. File vĩ mô toàn cầu (Ghép dữ liệu Global từ Yahoo Finance và thanh khoản FED)
if not global_df.empty or not fred_df.empty:
    global_macro_df = pd.concat([global_df, fred_df], axis=1).sort_index(ascending=False)
    global_macro_df.index.name = 'Date'
    update_csv('global_markets_database10years.csv', global_macro_df)

print("🎉 HOÀN TẤT! Data Pipeline 10 năm đã thông luồng và phân tách chuẩn xác.")
