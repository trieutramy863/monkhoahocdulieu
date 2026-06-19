import pandas as pd          # Thư viện xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
import seaborn as sns        # Thư viện trực quan hóa dữ liệu nâng cao

# ==========================================
# 0. CẤU HÌNH BAN ĐẦU & LOAD DỮ LIỆU
# ==========================================

# Đọc dữ liệu từ file CSV
# encoding='windows-1252' giúp tránh lỗi font khi đọc dữ liệu
try:
    df = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dữ liệu! Bạn nhớ đổi tên file ở dòng dưới cho đúng nhé.")
    df = pd.read_csv('superstore.csv', encoding='windows-1252')

# Chuyển cột Order Date sang kiểu ngày tháng để phân tích theo thời gian
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Thiết lập giao diện biểu đồ mặc định
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ==========================================
# 1. BIẾN ĐỘNG DOANH THU & LỢI NHUẬN THEO THỜI GIAN
# ==========================================
def cau_1_xu_huong_thoi_gian():
    print("\n--- CÂU 1: ĐANG XỬ LÝ XU HƯỚNG THỜI GIAN ---")

    # Gom nhóm dữ liệu theo từng tháng
    # Tính tổng doanh thu (Sales) và lợi nhuận (Profit)
    trend_df = df.groupby(df['Order Date'].dt.to_period('M'))[['Sales', 'Profit']].sum().reset_index()

    # Chuyển định dạng tháng sang chuỗi để hiển thị trên trục X
    trend_df['Order Date'] = trend_df['Order Date'].astype(str)

    # Tạo biểu đồ với 2 trục Y
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Trục Y thứ nhất biểu diễn doanh thu
    color = 'tab:blue'
    ax1.set_xlabel('Thời gian (Tháng)')
    ax1.set_ylabel('Tổng Doanh Thu ($)', color=color)
    ax1.plot(trend_df['Order Date'], trend_df['Sales'],
             color=color, marker='o', linewidth=2, label='Doanh thu')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(trend_df['Order Date'], rotation=90)

    # Trục Y thứ hai biểu diễn lợi nhuận
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Tổng Lợi Nhuận ($)', color=color)
    ax2.plot(trend_df['Order Date'], trend_df['Profit'],
             color=color, marker='s', linestyle='--', label='Lợi nhuận')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Xu Hướng Biến Động Doanh Thu và Lợi Nhuận Theo Tháng',
              fontsize=16, fontweight='bold')

    fig.tight_layout()
    plt.show()


# ==========================================
# 2. KHU VỰC NÀO MANG LẠI KẾT QUẢ CAO NHẤT
# ==========================================
def cau_2_hieu_suat_khu_vuc():
    print("\n--- CÂU 2: ĐANG XỬ LÝ PHÂN TÍCH KHU VỰC ---")

    # Gom nhóm theo Region
    # Tính tổng doanh thu và lợi nhuận của từng khu vực
    region_df = df.groupby('Region')[['Sales', 'Profit']].sum().sort_values(by='Sales', ascending=False)

    # In kết quả ra màn hình
    print(region_df)

    # Vẽ biểu đồ cột so sánh các khu vực
    region_df.plot(kind='bar', width=0.6)

    plt.title('So Sánh Doanh Thu & Lợi Nhuận Giữa Các Khu Vực',
              fontsize=14, fontweight='bold')
    plt.ylabel('Giá trị ($)')
    plt.xlabel('Khu vực')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. ĐÓNG GÓP CỦA CÁC NHÓM SẢN PHẨM
# ==========================================
def cau_3_nhom_san_pham():
    print("\n--- CÂU 3: ĐANG XỬ LÝ NHÓM SẢN PHẨM ---")

    # Gom nhóm theo từng phân nhóm sản phẩm (Sub-Category)
    # Sắp xếp giảm dần theo lợi nhuận
    sub_cat = df.groupby('Sub-Category')[['Sales', 'Profit']].sum() \
                .sort_values(by='Profit', ascending=False).reset_index()

    # Biểu đồ cột ngang thể hiện lợi nhuận của từng nhóm sản phẩm
    plt.figure(figsize=(12, 7))
    sns.barplot(data=sub_cat,
                x='Profit',
                y='Sub-Category',
                palette='RdYlGn')

    plt.title('Xếp Hạng Lợi Nhuận Theo Từng Phân Nhóm Sản Phẩm',
              fontsize=14, fontweight='bold')
    plt.xlabel('Tổng Lợi Nhuận ($)')
    plt.ylabel('Phân Nhóm Sản Phẩm')
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. TOP SẢN PHẨM BÁN CHẠY VÀ LỢI NHUẬN CAO
# ==========================================
def cau_4_top_san_pham():
    print("\n--- CÂU 4: ĐANG TRÍCH XUẤT TOP SẢN PHẨM ---")

    # Gom nhóm theo tên sản phẩm
    # Tính tổng số lượng bán và tổng lợi nhuận
    prod_df = df.groupby('Product Name').agg({
        'Quantity': 'sum',
        'Profit': 'sum'
    }).reset_index()

    # Lấy 5 sản phẩm bán chạy nhất
    top_selling = prod_df.sort_values(
        by='Quantity', ascending=False).head(5)

    # Lấy 5 sản phẩm có lợi nhuận cao nhất
    top_profitable = prod_df.sort_values(
        by='Profit', ascending=False).head(5)

    # Hiển thị danh sách top bán chạy
    print("\n⭐ TOP 5 SẢN PHẨM BÁN ĐƯỢC NHIỀU SỐ LƯỢNG NHẤT:")
    for idx, row in top_selling.iterrows():
        print(f"- {row['Product Name']}: {row['Quantity']} sản phẩm")

    # Hiển thị danh sách top lợi nhuận
    print("\n💰 TOP 5 SẢN PHẨM MANG LẠI LỢI NHUẬN CAO NHẤT:")
    for idx, row in top_profitable.iterrows():
        print(f"- {row['Product Name']}: ${row['Profit']:.2f}")

    # Tạo 2 biểu đồ đặt cạnh nhau
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Biểu đồ Top 5 sản phẩm bán chạy
    sns.barplot(data=top_selling,
                x='Quantity',
                y='Product Name',
                ax=axes[0],
                palette='Blues_r')

    # Biểu đồ Top 5 sản phẩm lợi nhuận cao
    sns.barplot(data=top_profitable,
                x='Profit',
                y='Product Name',
                ax=axes[1],
                palette='Greens_r')

    plt.tight_layout()
    plt.show()


# ==========================================
# 5. ẢNH HƯỞNG CỦA GIẢM GIÁ (DISCOUNT) TỚI LỢI NHUẬN
# ==========================================
def cau_5_anh_huong_giam_gia():
    print("\n--- CÂU 5: ĐANG PHÂN TÍCH TÁC ĐỘNG CỦA GIẢM GIÁ ---")

    # Gom nhóm theo mức giảm giá
    # Tính lợi nhuận trung bình tương ứng
    discount_perf = df.groupby('Discount')['Profit'].mean().reset_index()

    plt.figure(figsize=(10, 5))

    # Vẽ biểu đồ đường thể hiện mối quan hệ giữa giảm giá và lợi nhuận
    sns.lineplot(data=discount_perf,
                 x='Discount',
                 y='Profit',
                 marker='o',
                 color='red',
                 linewidth=2.5)

    # Đường ngang tại mức lợi nhuận = 0 (điểm hòa vốn)
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)

    plt.title('Mối Quan Hệ Giữa Mức Giảm Giá (Discount) và Lợi Nhuận Trung Bình',
              fontsize=14, fontweight='bold')

    plt.xlabel('Mức Giảm Giá')
    plt.ylabel('Lợi Nhuận Trung Bình trên Đơn Hàng ($)')
    plt.tight_layout()
    plt.show()


# ==========================================
# HÀM CHẠY CHÍNH (MAIN)
# ==========================================
if __name__ == '__main__':

    # Gọi lần lượt 5 hàm phân tích dữ liệu
    cau_1_xu_huong_thoi_gian()
    cau_2_hieu_suat_khu_vuc()
    cau_3_nhom_san_pham()
    cau_4_top_san_pham()
    cau_5_anh_huong_giam_gia()