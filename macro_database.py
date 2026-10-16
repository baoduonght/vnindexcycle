import yfinance as yf
import pandas as pd
from tvDatafeed import TvDatafeed, Interval

print("🤖 Bot đang khởi động lấy dữ liệu ngày mới...")

# 1. Lấy dữ liệu VN-Index mới nhất
tv = TvDatafeed()
vn_data = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=5)
vn_latest = vn_data[['close']].rename(columns={'close': 'VNINDEX'})
vn_latest.index = vn_latest.index.strftime('%Y-%m-%d')
vn_latest.index.name = 'Date'
vn_latest = vn_latest.round(2).reset_index()

# 2. Lấy dữ liệu Mỹ mới nhất
tickers = ["DX-Y.NYB", "^TNX", "JPY=X"]
us_data = yf.download(tickers, period="5d", progress=False)['Close']
us_latest = us_data.rename(columns={'DX-Y.NYB': 'DXY', '^TNX': 'US10Y', 'JPY=X': 'JPY_USD'})
us_latest.index = us_latest.index.strftime('%Y-%m-%d')
us_latest.index.name = 'Date'
us_latest = us_latest.round(2).reset_index()
us_latest = us_latest.dropna(subset=['DXY', 'US10Y', 'JPY_USD'], how='all')

# 3. Tạo bảng tổng hợp mới nhất
combo_latest = pd.merge(vn_latest, us_latest, on='Date', how='outer')

# 4. Hàm cập nhật nối dữ liệu vào 3 file CSV
def update_csv(filename, new_data):
    try:
        # Đọc file cũ hiện có trên GitHub
        df_old = pd.read_csv(filename)
        # Nối dữ liệu mới vào
        df_new = pd.concat([df_old, new_data])
        # Xóa các dòng trùng lặp ngày, giữ lại dữ liệu mới nhất
        df_new = df_new.drop_duplicates(subset=['Date'], keep='last')
        
        # ĐẢO CHIỀU: Sắp xếp theo ngày giảm dần (ngày mới nhất lên đầu)
        df_new = df_new.sort_values('Date', ascending=False)
        
        # Lưu đè lại vào file
        df_new.to_csv(filename, index=False)
        print(f"✅ Cập nhật thành công {filename}")
    except Exception as e:
        print(f"❌ Lỗi với {filename}: {e}")

# 5. Chạy cập nhật cho cả 3 kho
update_csv('vn_macro_database5years.csv', vn_latest)
update_csv('us_macro_database5years.csv', us_latest)
update_csv('macro_database5years.csv', combo_latest)

print("🎉 Nhiệm vụ hoàn tất! Dữ liệu đã được nạp đầy và đảo chiều.")
