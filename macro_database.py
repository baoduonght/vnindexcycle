import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import pandas_datareader.data as web
import datetime
import time

print("🚀 Khởi động Siêu Radar Vĩ mô: Bản Tối thượng (Raw Data 100%)...")

tv = TvDatafeed()

# --- PHẦN 1: DỮ LIỆU CƠ SỞ & PHÁI SINH VIỆT NAM (TradingView) ---
print("1. Đang quét nội tại chứng khoán Việt Nam...")
try:
    vn_raw = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=1500)
    vn_df = vn_raw[['close', 'volume']].copy()
    vn_df.rename(columns={'close': 'VNINDEX', 'volume': 'VN_VOLUME'}, inplace=True)

    vn30_raw = tv.get_hist(symbol='VN30', exchange='HOSE', interval=Interval.in_daily, n_bars=1500)
    vn30_df = vn30_raw[['close']].copy()
    vn30_df.rename(columns={'close': 'VN30'}, inplace=True)

    ps_raw = tv.get_hist(symbol='VN30F1M', exchange='HNX', interval=Interval.in_daily, n_bars=1500)
    ps_df = ps_raw[['close']].copy()
    ps_df.rename(columns={'close': 'VN30_F1M'}, inplace=True)

    local_df = pd.concat([vn_df, vn30_df, ps_df], axis=1)
    local_df['DERIVATIVE_SPREAD'] = local_df['VN30_F1M'] - local_df['VN30']
    
    local_df.index = local_df.index.normalize()
    local_df = local_df[~local_df.index.duplicated(keep='last')]
except Exception as e:
    print(f"Lỗi hệ thống nội địa (TV): {e}")
    local_df = pd.DataFrame()

# --- PHẦN 2: LÃI SUẤT LIÊN NGÂN HÀNG NỘI ĐỊA (vnstock) ---
print("2. Đang quét Lãi suất liên ngân hàng qua đêm...")
try:
    from vnstock import money_market_historical_data
    mm_df = money_market_historical_data(symbol='Lãi suất qua đêm', resolution='1D', type='interbank')
    if mm_df is not None and not mm_df.empty:
        mm_df.rename(columns={'close': 'VN_INTERBANK_ON'}, inplace=True)
        mm_df.set_index('date', inplace=True)
        mm_df.index = pd.to_datetime(mm_df.index).normalize()
        mm_df = mm_df[~mm_df.index.duplicated(keep='last')]
        local_df = pd.concat([local_df, mm_df[['VN_INTERBANK_ON']]], axis=1)
except Exception as e:
    print(f"Lỗi lấy dữ liệu Lãi suất liên ngân hàng: {e}")

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

print("3. Đang quét vệ tinh vĩ mô toàn cầu...")
df_list = []
for sym, exc, col_name in tv_symbols:
    try:
        data = tv.get_hist(symbol=sym, exchange=exc, interval=Interval.in_daily, n_bars=1500)
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
start_date = end_date - datetime.timedelta(days=5*365)
try:
    fred_df = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start_date, end_date)
    fred_df['FED_LIQUIDITY'] = fred_df['WALCL'] - fred_df['WTREGEN'] - fred_df['RRPONTSYD']
    fred_df = fred_df[['FED_LIQUIDITY']]
    fred_df.index = fred_df.index.normalize()
    fred_df = fred_df[~fred_df.index.duplicated(keep='last')]
except Exception as e:
    print(f"Lỗi tải dữ liệu FRED: {e}")
    fred_df = pd.DataFrame()

# --- PHẦN 5: HỢP NHẤT VÀ DỌN DẸP (NO FFILL) ---
combo_df = pd.concat([local_df, global_df, fred_df], axis=1)
combo_df.sort_index(inplace=True)

# KIỂM DUYỆT NGHIÊM NGẶT: Không dùng lệnh ffill ở đây để giữ nguyên gốc dữ liệu.

# Dọn rác: Xóa những dòng mà cả 3 thị trường chính đều nghỉ
combo_df.dropna(subset=['VNINDEX', 'DXY', 'US10Y'], how='all', inplace=True)
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

vn_cols = ['VNINDEX', 'VN_VOLUME', 'VN30', 'VN30_F1M', 'DERIVATIVE_SPREAD', 'VN10Y', 'VN_INTERBANK_ON', 'USD_VND']
vn_cols_exist = [c for c in vn_cols if c in combo_df.columns]

update_csv('macro_database5years.csv', combo_df)
update_csv('us_macro_database5years.csv', combo_df[us_cols_exist])
update_csv('vn_macro_database5years.csv', combo_df[vn_cols_exist])
update_csv('macro_database.csv', combo_df.head(60))

print("🎉 HOÀN TẤT! Dữ liệu 100% nguyên chất, nói không với độn số ảo.")
