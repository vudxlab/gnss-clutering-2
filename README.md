# GNSS Clustering

Phân cụm chuỗi thời gian dịch chuyển GNSS một chiều (tọa độ thẳng đứng **h_Coord**) theo từng đoạn giờ, sử dụng hai phương pháp bổ sung nhau:

| | Phương pháp 1 | Phương pháp 2 |
|---|---|---|
| **Ý tưởng** | Cluster trên giá trị chuỗi thô | Cluster trên vector đặc trưng vật lý |
| **Trả lời câu hỏi** | *"h nằm ở mức nào?"* | *"h có hành vi như thế nào?"* |
| **k tối ưu (thực nghiệm)** | 4 | 2 |
| **Silhouette tốt nhất** | 0.555 (HAC) | 0.251 (GMM) |

> Kết quả chi tiết, nhận xét ưu/nhược điểm và đề xuất cải tiến: xem [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Cấu trúc project

```
GNSS_Clustering2/
│
├── gnss_clustering/              # Package chính
│   ├── __init__.py               # Export 45 public symbols
│   ├── config.py                 # Đường dẫn, hyperparameter
│   ├── data_loader.py            # Tải CSV, tạo ma trận ngày/giờ
│   ├── preprocessing.py          # Hampel, Moving Avg, Gaussian, Butterworth, Kalman
│   ├── feature_extraction.py     # StandardScaler → PCA → t-SNE  (P.pháp 1)
│   ├── clustering.py             # TimeSeriesKMeans, HAC, DBSCAN, GMM
│   ├── optimization.py           # Tìm k tối ưu (voting system)
│   ├── visualization.py          # 20 hàm vẽ → lưu result/
│   └── feature_engineering.py   # Trích xuất đặc trưng + pipeline P.pháp 2
│
├── main.py                       # Entry point – chạy toàn bộ pipeline
│
├── data/                         # (git-ignored) Dữ liệu & cache trung gian
│   ├── full_gnss_2e.csv
│   ├── gnss_daily_matrix.npy
│   ├── gnss_hourly_matrix.npy
│   └── gnss_hourly_info.csv
│
├── result/                       # (git-ignored) Hình ảnh đầu ra (*.png)
│
├── docs/
│   └── RESULTS.md                # Kết quả chi tiết & phân tích
│
└── Clustering_GNSS_3e.ipynb      # Notebook gốc
```

---

## Cài đặt

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm
# Tùy chọn – TimeSeriesKMeans với DTW (nếu không có sẽ fallback sang KMeans)
pip install tslearn
```

---

## Chạy nhanh

### Toàn bộ pipeline (Phương pháp 1)

```bash
python main.py
```

### Phương pháp 2 – Feature-Based Clustering

```python
import gnss_clustering as gc

# Tải dữ liệu
df = gc.load_data('data/full_gnss_2e.csv')
daily_matrix, dates = gc.create_daily_matrix(df)
hourly_matrix, info = gc.create_hourly_matrix(daily_matrix, dates)

# Tiền xử lý
hampel_data, _ = gc.hampel_filter(hourly_matrix)

# Chạy pipeline đặc trưng
results = gc.run_feature_based_pipeline(
    hourly_matrix, hampel_data, info, n_clusters=2
)
```

### Sử dụng từng module

```python
import gnss_clustering as gc

# Tiền xử lý
data_filtered, hampel_data, _ = gc.preprocess_pipeline(hourly_matrix)

# Phương pháp 1: t-SNE → cluster
data_tsne, data_scaled, _ = gc.extract_features(data_filtered)
results = gc.run_all(data_tsne, data_scaled=data_scaled, n_clusters=4)
gc.plot_clustering_results(results, data_tsne)

# Phương pháp 2: feature → cluster
feature_df, _  = gc.extract_feature_matrix(hourly_matrix, hampel_data)
X_scaled, _, _ = gc.preprocess_features(feature_df)
results2 = gc.run_feature_based_pipeline(hourly_matrix, hampel_data, info)
```

---

## Đầu ra

Tất cả hình ảnh được lưu tự động vào `result/` (DPI=150):

| File | Nội dung |
|------|----------|
| `01_daily_heatmap.png` | Heatmap ma trận ngày × giây |
| `02_daily_timeseries.png` | Chuỗi thời gian mỗi ngày |
| `03_hourly_heatmap.png` | Heatmap ma trận giờ hợp lệ |
| `04_hourly_filter_stats.png` | Histogram tỷ lệ thiếu, bar số giờ hợp lệ |
| `05_hourly_overview_4subplots.png` | 3D scatter, histogram, boxplot, line mẫu |
| `06_hourly_analysis_2x2.png` | Heatmap chi tiết, so sánh trước/sau lọc, violin, weekday |
| `07_sample_hours.png` | 6 giờ mẫu tốt nhất |
| `08/09_first_20_hours_*.png` | Heatmap + lưới 20 giờ đầu |
| `10_hampel_compare_batch.png` | So sánh trước/sau Hampel |
| `11–13_*_grid.png` | Lưới chuỗi raw / Hampel / Kalman |
| `14_optimal_k_analysis.png` | Voting tìm k tối ưu |
| `15_clustering_scatter.png` | Scatter t-SNE 4 thuật toán |
| `16_clustering_metrics.png` | Bar chart Silhouette/Calinski/Davies |
| `17_lineplot_*.png` | Line plot trung bình ± std mỗi cụm |
| `F00_optimal_k_features.png` | Optimal k cho P.pháp 2 |
| `F01_feature_boxplot.png` | Phân phối 21 đặc trưng |
| `F02_pca_loadings.png` | PCA loadings |
| `F03_scatter_*.png` | Scatter không gian đặc trưng |
| `F04_cluster_profiles_*.png` | Profile đặc trưng trung bình mỗi cụm |
| `F05_cluster_ts_*.png` | Chuỗi thời gian trung bình mỗi cụm |

---

## Cấu hình

Sửa `gnss_clustering/config.py` để thay đổi:

```python
DATA_PATH          = 'data/full_gnss_2e.csv'   # File dữ liệu đầu vào
MISSING_THRESHOLD  = 0      # % thiếu tối đa cho giờ hợp lệ
HAMPEL_WINDOW_SIZE = 50     # Cửa sổ Hampel filter
RESHAPE_WINDOW_SIZE = 10    # Giảm chiều: 3600 → 360
DEFAULT_N_CLUSTERS = 4      # Số cụm mặc định
K_RANGE            = (2,11) # Khoảng k cần thử
```
