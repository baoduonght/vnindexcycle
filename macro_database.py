import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import pandas_datareader.data as web
import yfinance as yf
import datetime
import time
import requests

print("🚀 Khởi động Siêu Radar Vĩ mô: Bản Tối thượng 10 Năm...")

tv = TvDatafeed()

# --- PHẦN 1: DỮ LIỆU CƠ SỞ & PHÁI SINH VIỆT NAM (Sửa đổi: Dùng TradingView cho VN) ---
print("1. Đang quét chứng khoán Việt Nam (VNINDEX, VN30, Phái sinh)...")
local_df_list = []

try:
    # Dùng TradingView (HOSE) để lấy VNINDEX thay vì yfinance
    vn_raw = tv.get_hist(symbol='VNINDEX', exchange='HOSE', interval=Interval.in_daily, n_bars=3000)
    if vn_raw is not None and not vn_raw.empty:
        vn_df = vn_raw[['close', 'volume']].copy()
        vn_df.rename(columns={'close': 'VNINDEX', 'volume': 'VN_VOLUME'}, inplace=True)
        local_df_list.append(vn_df)

    # Dùng TradingView để lấy VN30
    vn30_raw = tv.get_hist(symbol='VN30', exchange='HOSE', interval=Interval.in_daily, n_bars=3000)
    if vn30_raw is not None and not vn30_raw.empty:
        vn30_df = vn30_raw[['close']].copy()
        vn30_df.rename(columns={'close': 'VN30'}, inplace=True)
        local_df_list.append(vn30_df)

    # Phái sinh VN30F1M 
    ps_raw = tv.get_hist(symbol='VN30F1M', exchange='HNX', interval=Interval.in_daily, n_bars=3000)
    if ps_raw is not None and not ps_raw.empty:
        ps_df = ps_raw[['close']].copy()
        ps_df.rename(columns={'close': 'VN30_F1M'}, inplace=True)
        local_df_list.append(ps_df)

    # Hợp nhất dữ liệu nội địa
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

# (Giữ nguyên PHẦN 2, 3, 4, 5 của bạn)

# --- PHẦN 5: HỢP NHẤT TỔNG THỂ ---
# Lưu ý quan trọng: Dùng how='outer' để không làm mất dữ liệu của thị trường nghỉ lễ lệch nhau
combo_df = pd.concat([local_df, global_df, fred_df], axis=1, join='outer')
combo_df = combo_df.sort_index(ascending=False)
combo_df.index.name = 'Date'

# --- PHẦN 6: XUẤT XƯỞNG ---
def update_csv(filename, new_df):
    if not new_df.empty:
        new_df.to_csv(filename)
        print(f"✅ Đã đóng gói: {filename}")
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
