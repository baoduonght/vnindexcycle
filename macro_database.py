import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import pandas_datareader.data as web
import datetime
import time

print("🚀 Khởi động Siêu Radar Vĩ mô (100% TradingView & FRED)...")

tv = TvDatafeed()

# 1. DANH SÁCH MÃ TRADINGVIEW: (Symbol, Exchange, Tên_Cột)
tv_symbols = [
    ('VNINDEX', 'HOSE', 'VNINDEX'),
    ('DXY', 'TVC', 'DXY'),
    ('US10Y', 'TVC', 'US10Y'),
    ('US02Y', 'TVC', 'US2Y'),
    ('USDJPY', 'FX_IDC', 'JPY_USD'),
    ('USDVND', 'FX_IDC', 'USD_VND'),
    ('USDCNH', 'FX_IDC', 'USD_CNH'),
    ('XAUUSD', 'FX_IDC', 'GOLD'),  # Mã vàng chuẩn xác nhất trên TV
    ('USOIL', 'TVC', 'OIL'),       # Mã dầu thô WTI
    ('VNQ', 'AMEX', 'US_REAL_ESTATE') # Quỹ ETF Bất động sản Mỹ
]

print("Đang tải dữ liệu từ TradingView...")
df_list = []

for sym, exc, col_name in tv_symbols:
    try:
        print(f"  -> Đang lấy {col_name} ({sym})...")
        data = tv.get_hist(symbol=sym, exchange=exc, interval=Interval.in_daily, n_bars=1500)
        if data is not None and not data.empty:
            temp_df = data[['close']].copy()
            temp_df.rename(columns={'close': col_name}, inplace=True)
            temp_df.index = temp_df.index.normalize()
            # Xóa ngày trùng lặp ngay từ đầu
            temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
            df_list.append(temp_df)
        # Tạm nghỉ 0.5s giữa các lần gọi để TradingView không chặn Bot
        time.sleep(0.5)
    except Exception as e:
        print(f"Lỗi khi lấy {col_name}: {e}")

# Gộp toàn bộ dữ liệu TradingView
if df_list:
    tv_df = pd.concat(df_list, axis=1)
else:
    tv_df = pd.DataFrame()

# 2. LẤY DỮ LIỆU THANH KHOẢN TỪ FED (FRED)
print("Đang đột nhập kho dữ liệu Cục Dự trữ Liên bang Mỹ (FRED)...")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5*365)
fred_tickers = ['WALCL', 'WTREGEN', 'RRPONTSYD']

try:
    fred_df = web.DataReader(fred_tickers, 'fred', start_date, end_date)
    fred_df['FED_LIQUIDITY'] = fred_df['WALCL'] - fred_df['WTREGEN'] - fred_df['RRPONTSYD']
    fred_df = fred_df[['FED_LIQUIDITY']]
    fred_df.index = fred_df.index.normalize()
    fred_df = fred_df[~fred_df.index.duplicated(keep='last')]
except Exception as e:
    print("Lỗi tải dữ liệu FRED:", e)
    fred_df = pd.DataFrame()

# 3. GỘP TOÀN BỘ VÀ XỬ LÝ SẠCH
combo_df = pd.concat([tv_df, fred_df], axis=1)
combo_df.sort_index(inplace=True)

# Lấp đầy riêng cho FED (vì FED cập nhật 1 lần/tuần)
if 'FED_LIQUIDITY' in combo_df.columns:
    combo_df['FED_LIQUIDITY'] = combo_df['FED_LIQUIDITY'].ffill()

# Dọn dẹp rác (nếu ngày đó VNINDEX, DXY, US10Y đều trống thì xóa dòng)
combo_df.dropna(subset=['VNINDEX', 'DXY', 'US10Y'], how='all', inplace=True)

# Đảo ngược để ngày mới nhất nằm trên cùng
combo_df = combo_df.sort_index(ascending=False)
combo_df.index.name = 'Date'

# 4. PHÂN BỔ VÀ LƯU CSV
def update_csv(filename, new_df):
    new_df.to_csv(filename)
    print(f"✅ Đã lưu thành công: {filename}")

# Lọc các cột cho file US
us_cols = [col for _, _, col in tv_symbols if col != 'VNINDEX']
if 'FED_LIQUIDITY' in combo_df.columns:
    us_cols.append('FED_LIQUIDITY')

us_cols_exist = [c for c in us_cols if c in combo_df.columns]
us_final = combo_df[us_cols_exist]

# Xuất xưởng
update_csv('macro_database5years.csv', combo_df)
update_csv('us_macro_database5years.csv', us_final)
if 'VNINDEX' in combo_df.columns:
    update_csv('vn_macro_database5years.csv', combo_df[['VNINDEX']])
update_csv('macro_database.csv', combo_df.head(60))

print("🎉 Nhiệm vụ hoàn tất! Hệ thống đã chạy độc lập với Yahoo Finance.")
