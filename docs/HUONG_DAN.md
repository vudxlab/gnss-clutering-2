# Hướng dẫn sử dụng – GNSS Clustering

## Tổng quan quy trình 2 bước

```
┌─────────────────────────────────────────────────────────────────────┐
│  BƯỚC 1 – Tìm số cụm tối ưu (step1_find_k.py)                      │
│                                                                      │
│  Tải dữ liệu → Tiền xử lý → Chạy k-search (k=2..10)               │
│  → Biểu đồ Silhouette / Calinski / Davies → Bỏ phiếu               │
│  → In khuyến nghị k cho PP1 và PP2                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  Xem biểu đồ → Quyết định k1, k2
┌──────────────────────────▼──────────────────────────────────────────┐
│  BƯỚC 2 – Phân cụm chi tiết (step2_cluster.py --k1 4 --k2 2)       │
│                                                                      │
│  PP1: t-SNE → HAC / GMM / KMeans / DBSCAN                          │
│  PP2: Feature(21D) → PCA → HAC / GMM / DBSCAN                      │
│  → Tất cả biểu đồ kết quả → result/                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Yêu cầu

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm
pip install tslearn          # tùy chọn – TimeSeriesKMeans với DTW
```

Dữ liệu đầu vào: `data/full_gnss_2e.csv` (cột `Timestamp`, `h_Coord`).

---

## BƯỚC 1 – Tìm số cụm tối ưu

### Lệnh cơ bản

```bash
python step1_find_k.py
```

### Tất cả tùy chọn

```bash
python step1_find_k.py [OPTIONS]

OPTIONS:
  --k-min INT         Giá trị k nhỏ nhất cần thử (mặc định: config.K_RANGE[0] = 2)
  --k-max INT         Giá trị k lớn nhất +1 (mặc định: config.K_RANGE[1] = 11)
  --method1-only      Chỉ chạy Phương pháp 1 (bỏ qua feature-based k-search)
  --method2-only      Chỉ chạy Phương pháp 2 (bỏ qua t-SNE k-search)
  --no-display        Không mở cửa sổ (dùng Agg backend – cho server/headless)
  --no-cache          Bỏ qua cache, tải lại dữ liệu từ CSV
```

### Ví dụ

```bash
# Mặc định: k từ 2 đến 10, cả hai phương pháp
python step1_find_k.py

# Thu hẹp khoảng k để chạy nhanh hơn
python step1_find_k.py --k-min 2 --k-max 7

# Chỉ tìm k cho Phương pháp 2 (feature-based)
python step1_find_k.py --method2-only

# Môi trường server không có GUI
python step1_find_k.py --no-display

# Buộc tải lại từ CSV (bỏ qua cache)
python step1_find_k.py --no-cache
```

### Đầu ra Bước 1

| File | Nội dung |
|------|----------|
| `result/14_optimal_k_analysis.png` | PP1: Silhouette / Calinski / Davies theo k + voting bar |
| `result/F00_optimal_k_features.png` | PP2: Silhouette / Calinski / Davies theo k + voting bar |
| In ra terminal | Bảng metrics từng k, bỏ phiếu, khuyến nghị k cuối |

**Ví dụ in ra terminal:**

```
======================================================================
BƯỚC 1 – TÌM SỐ CỤM TỐI ƯU
  Kết quả lưu vào: result/
======================================================================

Khoảng k: 2 đến 10

[1/4] Tải / kiểm tra cache dữ liệu...
  Cache tồn tại – đang tải...
  hourly_matrix: (191, 3600), valid hours: 191

[2/4] Tiền xử lý (Hampel → reshape → Kalman)...

[3/4] PHƯƠNG PHÁP 1 – Tìm k trên không gian t-SNE
  ...
  [PP1] Số cụm đề xuất: k = 4

[4/4] PHƯƠNG PHÁP 2 – Tìm k trên không gian đặc trưng
   k  HAC_Sil   HAC_Cal   HAC_Dav  GMM_Sil   GMM_Cal   GMM_Dav
  -------------------------------------------------------------------
   2   0.2340    62.34    1.2100   0.2510    58.21    1.3200
   3   0.1980    55.12    1.4500   0.1870    49.80    1.5600
   ...
  [PP2] Số cụm đề xuất: k = 2

======================================================================
TÓM TẮT KẾT QUẢ – BƯỚC 1
======================================================================
Phương pháp                    k đề xuất   Số phiếu
-------------------------------------------------------
PP1 – Raw t-SNE (HAC/GMM/KMeans)         4          5
PP2 – Feature-Based (HAC/GMM)            2          4

Tiếp theo, chạy phân cụm chi tiết (Bước 2):
  python step2_cluster.py --k1 4 --k2 2
```

### Cách đọc biểu đồ để chọn k

Biểu đồ `14_optimal_k_analysis.png` và `F00_optimal_k_features.png` gồm 6 ô:

| Ô | Nội dung | Cách đọc |
|---|----------|----------|
| [0,0] | Silhouette vs k | **Chọn k tại đỉnh cao nhất** |
| [0,1] | Calinski-Harabasz vs k | **Chọn k tại đỉnh cao nhất** |
| [0,2] | Davies-Bouldin vs k | **Chọn k tại điểm thấp nhất** |
| [1,0] | Bar: k tốt nhất theo từng metric | Tham khảo độ nhất quán |
| [1,1] | Heatmap performance tại k đề xuất | Xanh = tốt, đỏ = kém |
| [1,2] | Bỏ phiếu Top-5 k | **k nhiều phiếu nhất = khuyến nghị** |

> **Lưu ý:** Không nhất thiết phải chọn k có số phiếu cao nhất. Nếu biểu đồ Silhouette có hai đỉnh gần bằng nhau, hãy ưu tiên k nhỏ hơn (Occam's razor – mô hình đơn giản hơn).

---

## BƯỚC 2 – Phân cụm chi tiết

### Lệnh cơ bản

```bash
python step2_cluster.py --k1 4 --k2 2
```

### Tất cả tùy chọn

```bash
python step2_cluster.py [OPTIONS]

OPTIONS:
  --k1 INT            Số cụm cho Phương pháp 1 (mặc định: config.DEFAULT_N_CLUSTERS)
  --k2 INT            Số cụm cho Phương pháp 2 (mặc định: config.DEFAULT_N_CLUSTERS)
  --method1-only      Chỉ chạy Phương pháp 1
  --method2-only      Chỉ chạy Phương pháp 2
  --no-display        Không mở cửa sổ (Agg backend)
  --no-cache          Bỏ qua cache, tải lại từ CSV
```

### Ví dụ

```bash
# Chạy cả hai phương pháp với k được chọn từ Bước 1
python step2_cluster.py --k1 4 --k2 2

# Chỉ phân cụm Phương pháp 1 với k=3
python step2_cluster.py --k1 3 --method1-only

# Chỉ phân cụm Phương pháp 2 với k=3
python step2_cluster.py --k2 3 --method2-only

# Server/headless
python step2_cluster.py --k1 4 --k2 2 --no-display
```

### Đầu ra Bước 2

#### Phương pháp 1 – Raw Time-Series Clustering

| File | Nội dung |
|------|----------|
| `result/10_hampel_compare_batch.png` | So sánh trước/sau Hampel (25 giờ mẫu) |
| `result/11_raw_hourly_grid.png` | Lưới 25 chuỗi giờ gốc |
| `result/12_hampel_grid.png` | Lưới 25 chuỗi sau Hampel |
| `result/13_kalman_grid.png` | Lưới 25 chuỗi sau Kalman |
| `result/15_clustering_scatter.png` | Scatter t-SNE 2D tô màu theo cụm (4 thuật toán) |
| `result/16_clustering_metrics.png` | Bar chart Silhouette / Calinski / Davies (4 TT) |
| `result/17_lineplot_hac.png` | Mean ± std chuỗi giờ theo từng cụm (HAC) |
| `result/17_lineplot_gmm.png` | Mean ± std chuỗi giờ theo từng cụm (GMM) |
| `result/17_lineplot_timeseries_kmeans.png` | Mean ± std theo từng cụm (KMeans) |

#### Phương pháp 2 – Feature-Based Clustering

| File | Nội dung |
|------|----------|
| `result/F01_feature_boxplot.png` | Phân phối 21 đặc trưng đã chuẩn hóa |
| `result/F02_pca_loadings.png` | PCA loadings: đóng góp của đặc trưng vào PC1/PC2 |
| `result/F03_scatter_hac.png` | Scatter PCA 2D tô màu theo cụm HAC |
| `result/F03_scatter_gmm.png` | Scatter PCA 2D tô màu theo cụm GMM |
| `result/F03_scatter_dbscan.png` | Scatter PCA 2D tô màu theo cụm DBSCAN |
| `result/F04_cluster_profiles_hac.png` | Bar chart: giá trị trung bình đặc trưng mỗi cụm (HAC) |
| `result/F04_cluster_profiles_gmm.png` | Bar chart: giá trị trung bình đặc trưng mỗi cụm (GMM) |
| `result/F05_cluster_ts_hac.png` | Chuỗi thời gian trung bình mỗi cụm (HAC) |
| `result/F05_cluster_ts_gmm.png` | Chuỗi thời gian trung bình mỗi cụm (GMM) |

### Đọc kết quả Bước 2

**Biểu đồ `15_clustering_scatter.png`**: 4 subplot tương ứng 4 thuật toán. Mỗi điểm = 1 giờ quan sát, màu = cụm. Cụm tốt khi các điểm cùng màu tập trung chặt và tách biệt khỏi màu khác.

**Biểu đồ `16_clustering_metrics.png`**: So sánh 3 chỉ số chất lượng:
- Silhouette: **cao hơn = tốt hơn** (max = 1)
- Calinski-Harabasz: **cao hơn = tốt hơn**
- Davies-Bouldin: **thấp hơn = tốt hơn** (min = 0)

**Biểu đồ `17_lineplot_*.png`**: Đường trung bình ± std của từng cụm. Giúp xác định đặc trưng hành vi của mỗi cụm (ví dụ: Cụm 0 = nhiễu cao, Cụm 1 = ổn định).

**Biểu đồ `F04_cluster_profiles_*.png`**: Bar chart trung bình đặc trưng đã chuẩn hóa. Giúp giải thích vật lý từng cụm:
- `energy_high` cao → giờ có nhiều dao động cao tần
- `autocorr_lag1` cao → chuỗi mượt, tương quan cao
- `trend_slope` lớn → có xu hướng tăng/giảm trong giờ đó
- `hurst_exponent` > 0.5 → tín hiệu có xu hướng bền vững (persistent)

---

## Chạy nhanh (toàn bộ pipeline)

Nếu muốn chạy tất cả trong một lệnh (tương đương `main.py` cũ):

```bash
python main.py
```

> `main.py` tự động tìm k tối ưu (Bước 1) và phân cụm ngay (Bước 2) với k đó, không dừng để hỏi. Dùng `step1_find_k.py` + `step2_cluster.py` khi muốn xem xét kỹ và tự quyết định k.

---

## Tùy chỉnh cấu hình

Sửa `gnss_clustering/config.py`:

```python
DATA_PATH           = 'data/full_gnss_2e.csv'  # File CSV đầu vào
MISSING_THRESHOLD   = 0       # % thiếu tối đa (0 = chỉ giữ giờ đầy đủ)
HAMPEL_WINDOW_SIZE  = 50      # Cửa sổ Hampel filter
HAMPEL_N_SIGMAS     = 1       # Ngưỡng sigma Hampel
RESHAPE_WINDOW_SIZE = 10      # Giảm chiều: 3600 → 360
PCA_N_COMPONENTS    = 50      # Số thành phần PCA (PP1)
TSNE_METRIC         = 'l1'    # Khoảng cách t-SNE
DEFAULT_N_CLUSTERS  = 2       # k mặc định nếu không truyền --k1/--k2
K_RANGE             = (2, 11) # Khoảng k cần thử (end không bao gồm)
SEED                = 23      # Random seed
```

---

## Sơ đồ kiến trúc module

```
gnss_clustering/
├── config.py              Đường dẫn, hyperparameter
├── data_loader.py         load_data, create_daily_matrix, create_hourly_matrix
│                          load_cached_matrices
├── preprocessing.py       hampel_filter, kalman_filter_2d, preprocess_pipeline
├── feature_extraction.py  scale_data, apply_pca, apply_tsne, extract_features
│                          (Phương pháp 1 – dữ liệu thô → t-SNE)
├── clustering.py          run_timeseries_kmeans, run_hierarchical_clustering
│                          run_dbscan_clustering, run_gmm_clustering, run_all
├── optimization.py        find_optimal_clusters   ← B1 cho PP1
│                          create_optimization_plots
├── visualization.py       20 hàm vẽ → lưu result/
└── feature_engineering.py extract_feature_matrix, preprocess_features
                           find_optimal_clusters_features  ← B1 cho PP2
                           run_feature_based_pipeline      ← B2 cho PP2
                           plot_feature_importance/scatter/profiles/timeseries
```

---

## Giải thích 21 đặc trưng (Phương pháp 2)

| Nhóm | Đặc trưng | Ý nghĩa |
|------|-----------|---------|
| **Thống kê** | `mean` | Mức trung bình dịch chuyển trong giờ |
| | `std` | Độ biến động tổng thể |
| | `skewness` | Độ lệch phân phối |
| | `kurtosis` | Độ nhọn (nhiều giá trị ngoại lai?) |
| | `iqr` | Khoảng tứ vị phân (robust spread) |
| **Xu hướng** | `trend_slope` | Tốc độ tăng/giảm tuyến tính (mm/s) |
| | `trend_intercept` | Giá trị khởi đầu của trend |
| | `trend_r2` | Mức độ phù hợp của trend tuyến tính |
| | `trend_resid_std` | Độ phân tán quanh trend |
| **Tần số** | `dominant_freq` | Tần số dao động chủ đạo |
| | `spectral_entropy` | Mức độ phân tán năng lượng tần số |
| | `energy_low` | Tỉ lệ năng lượng tần số thấp (< 10% Nyquist) |
| | `energy_mid` | Tỉ lệ năng lượng tần số trung |
| | `energy_high` | Tỉ lệ năng lượng tần số cao (> 30% Nyquist) |
| **Cấu trúc TG** | `autocorr_lag1` | Tương quan tự hồi quy lag-1 |
| | `autocorr_lag5` | Tương quan tự hồi quy lag-5 |
| | `autocorr_lag30` | Tương quan tự hồi quy lag-30 |
| | `hurst_exponent` | H<0.5: trung bình hồi; H≈0.5: ngẫu nhiên; H>0.5: xu hướng bền |
| **Chất lượng** | `valid_ratio` | Tỉ lệ điểm không thiếu |
| | `outlier_ratio` | Tỉ lệ ngoại lai theo Hampel |
| | `signal_range` | Biên độ tín hiệu (max−min) |
| ~~`snr`~~ | *(loại bỏ)* | Bị loại do phương sai gần 0 |

---

## Câu hỏi thường gặp

**Q: Silhouette của PP2 (0.23–0.25) thấp hơn PP1 (0.55). PP2 tệ hơn không?**

A: Không. Silhouette đo khoảng cách hình học trong không gian hiện tại. PP1 dùng t-SNE 2D (đã tối ưu hóa để tách cụm), nên Silhouette tự nhiên cao hơn. PP2 làm việc trong không gian 21 chiều thực sự – các cụm gần nhau hơn nhưng có ý nghĩa vật lý rõ ràng hơn. Dùng biểu đồ `F04_cluster_profiles_*.png` để đánh giá chất lượng PP2.

**Q: Cache là gì và khi nào cần `--no-cache`?**

A: Sau lần chạy đầu tiên, dữ liệu trung gian được lưu vào `data/` (`.npy`, `.csv`). Các lần sau sẽ tải cache thay vì tính lại từ CSV (nhanh hơn nhiều). Dùng `--no-cache` khi thay đổi file CSV đầu vào hoặc thay đổi `MISSING_THRESHOLD`.

**Q: `tslearn` không cài được. Có ảnh hưởng không?**

A: Không. Khi không có `tslearn`, `run_timeseries_kmeans` tự động fallback sang `sklearn.KMeans`. Kết quả vẫn đầy đủ, chỉ mất phần DTW distance.

**Q: Chạy trên server không có màn hình?**

A: Thêm `--no-display` vào lệnh. Backend sẽ chuyển sang `Agg` (chỉ lưu file, không hiển thị).
