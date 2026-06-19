import pandas as pd                                      # Xử lý dữ liệu dạng bảng
from sklearn.model_selection import train_test_split     # Chia dữ liệu Train/Test
from sklearn.ensemble import RandomForestClassifier      # Thuật toán Random Forest
from sklearn.metrics import classification_report, accuracy_score # Đánh giá mô hình

# ==========================================
# 1. LOAD DỮ LIỆU & CHUẨN BỊ BIẾN MỤC TIÊU
# ==========================================

# Đọc dữ liệu từ file CSV
try:
    df = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')
except FileNotFoundError:
    df = pd.read_csv('superstore.csv', encoding='windows-1252')

# Tính giá trị lợi nhuận trung vị (Median Profit)
# Dùng làm ngưỡng phân loại lợi nhuận cao và thấp
median_profit = df['Profit'].median()

print(f"Mức lợi nhuận trung vị của siêu thị là: ${median_profit:.2f}")

# Tạo biến mục tiêu (Target)
# Profit > Median → 1 (Lợi nhuận cao)
# Profit <= Median → 0 (Lợi nhuận thấp)
df['Profit_Class'] = (df['Profit'] > median_profit).astype(int)

# ==========================================
# 2. LỰA CHỌN VÀ MÃ HÓA TÍNH NĂNG (FEATURES)
# ==========================================

# Chọn các biến đầu vào ảnh hưởng đến lợi nhuận
features = [
    'Sales',      # Doanh thu
    'Quantity',   # Số lượng bán
    'Discount',   # Mức giảm giá
    'Region',     # Khu vực
    'Category',   # Danh mục sản phẩm
    'Segment'     # Phân khúc khách hàng
]

# X là tập đặc trưng đầu vào
X = df[features]

# y là biến mục tiêu cần dự đoán
y = df['Profit_Class']

# Chuyển các cột dạng chữ thành dạng số bằng One-Hot Encoding
# Máy học chỉ làm việc với dữ liệu số
X = pd.get_dummies(
    X,
    columns=['Region', 'Category', 'Segment'],
    drop_first=True
)

# ==========================================
# 3. CHIA DỮ LIỆU TRAIN & TEST
# ==========================================

# Chia dữ liệu:
# 80% dùng huấn luyện mô hình
# 20% dùng kiểm tra mô hình
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH RANDOM FOREST
# ==========================================

print("\nĐang huấn luyện mô hình dự đoán...")

# Khởi tạo mô hình Random Forest gồm 100 cây quyết định
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Huấn luyện mô hình trên dữ liệu train
model.fit(X_train, y_train)

# ==========================================
# 5. ĐÁNH GIÁ ĐỘ CHÍNH XÁC CỦA MÔ HÌNH
# ==========================================

# Dự đoán kết quả trên tập test
y_pred = model.predict(X_test)

# Tính độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)

print("\n======= KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH =======")

print(f"Độ chính xác tổng thể (Accuracy): {accuracy * 100:.2f}%")

print("\nBáo cáo chi tiết (Classification Report):")

# Hiển thị Precision, Recall, F1-Score
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            'Lợi nhuận Thấp',
            'Lợi nhuận Cao'
        ]
    )
)

# ==========================================
# 6. ĐO LƯỜNG MỨC ĐỘ QUAN TRỌNG CỦA CÁC YẾU TỐ
# ==========================================

# Lấy mức độ ảnh hưởng của từng đặc trưng
importances = model.feature_importances_

# Tạo bảng xếp hạng mức độ quan trọng
feature_imp_df = pd.DataFrame({
    'Tính năng': X.columns,
    'Mức độ quan trọng (%)': importances * 100
}).sort_values(
    by='Mức độ quan trọng (%)',
    ascending=False
)

print("\n======= CÁC YẾU TỐ QUYẾT ĐỊNH ĐẾN LỢI NHUẬN =======")

# Hiển thị 5 yếu tố quan trọng nhất
print(feature_imp_df.head(5).to_string(index=False))