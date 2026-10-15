import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from tvDatafeed import TvDatafeed, Interval

print("Đang bật radar dò tìm cừu béo...")

# 1. Khởi tạo bộ khung dữ liệu
today = datetime.now().strftime("%Y-%m-%d")
data_today = {"Date": [today], "VNINDEX": [None], "DXY": [None], "US10Y": [None], "JPY_USD": [None]}

# =================================================================
# 2. LẤY VN-INDEX TỪ TRADINGVIEW (Mở rộng vùng quét 5 ngày)
# =================================================================
try:
    tv = TvDatafeed()
    # Yêu cầu 5 cây nến thay vì 1 để chống lỗi chưa mở cửa
    vn_data = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=5)

    if vn_data is not None and not vn_data.empty:
        # Lọc bỏ giá trị rỗng (NaN) và lấy con số hợp lệ cuối cùng
        data_today["VNINDEX"] = [round(float(vn_data['close'].dropna().iloc[-1]), 2)]
except Exception as e:
    print(f"❌ Lỗi lấy VN-Index: {e}")

# =================================================================
# 3. LẤY DỮ LIỆU VĨ MÔ QUỐC TẾ TỪ YAHOO (Mở rộng vùng quét 5 ngày)
# =================================================================
try:
    tickers_list = ["DX-Y.NYB", "^TNX", "JPY=X"]
    # Yêu cầu dữ liệu 5 ngày gần nhất
    df_raw = yf.download(tickers_list, period="5d", progress=False)

    # Lọc bỏ NaN và trích xuất giá đóng cửa hợp lệ cuối cùng
    data_today["DXY"]     = [round(float(df_raw['Close']['DX-Y.NYB'].dropna().iloc[-1]), 2)]
    data_today["US10Y"]   = [round(float(df_raw['Close']['^TNX'].dropna().iloc[-1]), 2)]
    data_today["JPY_USD"] = [round(float(df_raw['Close']['JPY=X'].dropna().iloc[-1]), 2)]
except Exception as e:
    print(f"❌ Lỗi lấy vĩ mô quốc tế: {e}")

# =================================================================
# 4. ĐÓNG GÓI VÀ LƯU VÀO DATABASE (FILE CSV)
# =================================================================
df_macro = pd.DataFrame(data_today)

print("\n--- TÍN HIỆU VĨ MÔ HÔM NAY ---")
print(df_macro.to_string(index=False))

# Lưu thẳng vào file CSV
csv_filename = "macro_database.csv"
file_exists = os.path.isfile(csv_filename)

df_macro.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
print(f"\n✅ Đã ghi chép xong dữ liệu vào '{csv_filename}'. Dao đã mài, sẵn sàng chờ thời điểm!")
