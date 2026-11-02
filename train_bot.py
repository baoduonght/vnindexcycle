# (Giả định bác đã để các hàm tính toán và code tải dữ liệu vnindex_df, stocks_close ở phần trên)
import xgboost as xgb
import numpy as np

print("🔄 KHỞI ĐỘNG LÒ LUYỆN (THỨ 7): ĐÚC BỘ NÃO MỚI...")

# 1. Chuẩn bị Feature và Trọng số thời gian
train_data = ml_master_clean.dropna(subset=['Target']).copy()
train_data['Weight'] = np.where(train_data['Date'] >= '2024-01-01', 3.0, 1.0)

X_train = train_data[ml_features]
y_train = train_data['Target']
w_train = train_data['Weight']

scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1

# 2. Định hình và Huấn luyện AI
xgb_model = xgb.XGBClassifier(
    n_estimators=150,              
    max_depth=4,                   
    learning_rate=0.05,            
    subsample=0.8,                 
    colsample_bytree=0.8,          
    scale_pos_weight=scale_weight, 
    random_state=42,
    eval_metric='logloss'
)

# Chèn trọng số vào hàm fit
xgb_model.fit(X_train, y_train, sample_weight=w_train)

# 3. Xuất file não bộ
MODEL_FILE = "xgboost_ultimate.json"
xgb_model.save_model(MODEL_FILE)
print(f"✅ Đã luyện đan xong! Não bộ mới được lưu tại: {MODEL_FILE}")
