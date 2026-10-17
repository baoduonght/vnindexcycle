import pandas as pd
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import pandas_datareader.data as web
import datetime

print("🚀 Khởi động Siêu Radar Vĩ mô...")

# 1. LẤY DỮ LIỆU TỪ TRADINGVIEW (VN-INDEX & NGOẠI TỆ)
# Dùng TradingView lấy tỷ giá để đảm bảo không bao giờ bị khuyết dữ liệu như Yahoo
tv = TvDatafeed()

# Lấy VN-Index
vnindex = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=1500)
vn_df = vnindex[['close']].copy()
vn_df.rename(columns={'close': 'VNINDEX'}, inplace=True)

# Lấy Tỷ giá USD/JPY (Ký hiệu JPY_USD của bạn)
usdjpy = tv.get_hist(symbol='USDJPY', exchange='FX_IDC', interval=Interval.in_daily, n_bars=1500)
jpy_df = usdjpy[['close']].copy()
jpy_df.rename(columns={'close': 'JPY_USD'}, inplace=True)

# Gộp dữ liệu TradingView
tv_df = pd.concat([vn_df, jpy_df], axis=1)
tv_df.index = tv_df.index.normalize()

# 2. LẤY DỮ LIỆU MỸ & HÀNG HÓA TỪ YAHOO FINANCE
symbols = {
    'DX-Y.NYB': 'DXY',
    '^TNX': 'US10Y',
    '^TWO': 'US2Y',
    'VND=X': 'USD_VND',
    'CNH=F': 'USD_CNH', 
    'GC=F': 'GOLD',
    'CL=F': 'OIL',
    'VNQ': 'US_REAL_ESTATE'
}

print("Đang tải dữ liệu toàn cầu từ Yahoo Finance...")
us_data = yf.download(list(symbols.keys()), period="5y")
if 'Close' in us_data:
    us_df = us_data['Close'].copy()
else:
    us_df = us_data.copy()
    
us_df.rename(columns=symbols, inplace=True)
us_df.index = us_df.index.normalize()

# 3. LẤY DỮ LIỆU THANH KHOẢN TỪ FED (FRED)
print("Đang đột nhập kho dữ liệu Cục Dự trữ Liên bang Mỹ (FRED)...")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5*365)
fred_tickers = ['WALCL', 'WTREGEN', 'RRPONTSYD']

try:
    fred_df = web.DataReader(fred_tickers, 'fred', start_date, end_date)
    # Công thức thanh khoản cốt lõi
    fred_df['FED_LIQUIDITY'] = fred_df['WALCL'] - fred_df['WTREGEN'] - fred_df['RRPONTSYD']
    fred_df = fred_df[['FED_LIQUIDITY']]
except Exception as e:
    print("Lỗi tải dữ liệu FRED:", e)
    fred_df = pd.DataFrame()

# 4. GỘP TOÀN BỘ VÀ XỬ LÝ SẠCH
combo_df = pd.concat([tv_df, us_df, fred_df], axis=1)
combo_df.sort_index(inplace=True)

# Vì tài sản FED cập nhật 1 lần/tuần nên phải lấp đầy riêng cho nó, các chỉ số khác để nguyên số thực!
if 'FED_LIQUIDITY' in combo_df.columns:
    combo_df['FED_LIQUIDITY'] = combo_df['FED_LIQUIDITY'].ffill()

# Dọn dẹp các dòng rác hoàn toàn không có dữ liệu
combo_df.dropna(subset=['VNINDEX', 'DXY', 'US10Y'], how='all', inplace=True)

# Đảo ngược để ngày mới nhất nằm trên cùng
combo_df = combo_df.sort_index(ascending=False)
combo_df.index.name = 'Date'

# 5. PHÂN BỔ VÀ LƯU CSV
def update_csv(filename, new_df):
    new_df.to_csv(filename)
    print(f"✅ Đã lưu thành công: {filename}")

# Gom cột cho file US
us_cols = list(us_df.columns)
us_cols.append('JPY_USD')
if 'FED_LIQUIDITY' in combo_df.columns:
    us_cols.append('FED_LIQUIDITY')

us_cols_exist = [c for c in us_cols if c in combo_df.columns]
us_final = combo_df[us_cols_exist]

# Xuất xưởng
update_csv('macro_database5years.csv', combo_df)
update_csv('us_macro_database5years.csv', us_final)
update_csv('vn_macro_database5years.csv', combo_df[['VNINDEX']])
update_csv('macro_database.csv', combo_df.head(60))

print("🎉 Nhiệm vụ hoàn tất! Dữ liệu 100% là số thật, không vay mượn.")
