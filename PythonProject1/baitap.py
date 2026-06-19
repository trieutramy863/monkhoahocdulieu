import pandas as pd                  # Xử lý dữ liệu dạng bảng
import numpy as np                   # Hỗ trợ tính toán số học và ma trận
import matplotlib.pyplot as plt      # Vẽ biểu đồ
import seaborn as sns                # Trực quan hóa dữ liệu
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu
from sklearn.cluster import KMeans   # Thuật toán phân cụm K-Means

# ==========================================
# 1. LOAD DỮ LIỆU & ĐỊNH DẠNG
# ==========================================

# Đọc dữ liệu từ file CSV
try:
    df = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')
except FileNotFoundError:
    df = pd.read_csv('superstore.csv', encoding='windows-1252')

# Chuyển cột ngày đặt hàng sang định dạng DateTime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# ==========================================
# 2. TÍNH TOÁN BỘ CHỈ SỐ RFM
# ==========================================

# Lấy ngày sau đơn hàng cuối cùng 1 ngày để làm mốc tính Recency
snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)

# Gom nhóm dữ liệu theo từng khách hàng
rfm = df.groupby('Customer ID').agg({

    # Recency: số ngày kể từ lần mua hàng gần nhất
    'Order Date': lambda x: (snapshot_date - x.max()).days,

    # Frequency: số đơn hàng khác nhau của khách hàng
    'Order ID': 'nunique',

    # Monetary: tổng số tiền khách hàng đã chi tiêu
    'Sales': 'sum'

}).reset_index()

# Đổi tên cột cho dễ hiểu
rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

# ==========================================
# 3. TIỀN XỬ LÝ DỮ LIỆU (CHUẨN HÓA)
# ==========================================

# Biến đổi Log để giảm độ lệch dữ liệu
# Giúp giảm ảnh hưởng của các khách hàng chi tiêu quá lớn
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])

# Chuẩn hóa dữ liệu về cùng thang đo
# Trung bình = 0, độ lệch chuẩn = 1
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# ==========================================
# 4. CHẠY THUẬT TOÁN K-MEANS PHÂN NHÓM
# ==========================================

# Khởi tạo mô hình K-Means với 4 cụm khách hàng
kmeans = KMeans(
    n_clusters=4,     # Số nhóm cần phân cụm
    random_state=42,  # Đảm bảo kết quả lặp lại giống nhau
    n_init=10         # Chạy 10 lần để tìm nghiệm tốt nhất
)

# Gán nhãn Cluster cho từng khách hàng
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# ==========================================
# 5. ĐÁNH GIÁ VÀ ĐỌC KẾT QUẢ KINH DOANH
# ==========================================

# Tính giá trị trung bình của từng nhóm khách hàng
cluster_analysis = rfm.groupby('Cluster').agg({

    # Trung bình số ngày kể từ lần mua cuối
    'Recency': 'mean',

    # Trung bình số lần mua
    'Frequency': 'mean',

    # Trung bình tổng chi tiêu
    'Monetary': 'mean',

    # Đếm số lượng khách hàng trong nhóm
    'Customer ID': 'count'

}).rename(columns={
    'Customer ID': 'Số lượng khách'
}).reset_index()

# In kết quả phân tích
print(" \n======= ĐẶC TRƯNG HÀNH VI CỦA TỪNG NHÓM KHÁCH HÀNG =======")
print(cluster_analysis.to_string(index=False))

# ==========================================
# 6. TRỰC QUAN HÓA BẰNG ĐỒ THỊ
# ==========================================

# Thiết lập giao diện biểu đồ
sns.set_theme(style="whitegrid")

# Thiết lập kích thước biểu đồ
plt.figure(figsize=(10, 6))

# Biểu đồ phân tán thể hiện các nhóm khách hàng
sns.scatterplot(

    data=rfm,

    # Trục X: Tần suất mua hàng
    x='Frequency',

    # Trục Y: Tổng chi tiêu
    y='Monetary',

    # Màu sắc biểu thị từng cụm khách hàng
    hue='Cluster',

    # Kích thước điểm biểu thị Recency
    size='Recency',

    palette='Set1',
    sizes=(20, 200),
    alpha=0.7
)

# Tiêu đề biểu đồ
plt.title(
    'Phân Nhóm Khách Hàng Dựa Trên Hành Vi Mua Sắm (RFM)',
    fontsize=14,
    fontweight='bold'
)

# Tên trục X
plt.xlabel('Tần suất mua hàng (Số lần)')

# Tên trục Y
plt.ylabel('Tổng số tiền chi tiêu ($)')

# Dùng thang Log để tránh dữ liệu bị dồn vào một góc
plt.yscale('log')

# Hiển thị chú thích cụm khách hàng
plt.legend(title='Nhóm (Cluster)')

# Tự động căn chỉnh bố cục
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()