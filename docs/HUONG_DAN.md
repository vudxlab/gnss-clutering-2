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
│  [3] PP1: t-SNE → HAC / GMM / KMeans / DBSCAN                      │
│  [4] PP2: Feature(21D) → PCA → HAC / GMM / DBSCAN                  │
│  [5] PP2v2: Feature(40D) → Weighting → HAC / GMM / HDBSCAN / Ens.  │
│  [6] M3A: Conv1D Autoencoder → Latent 32D → HAC / GMM / HDBSCAN    │
│  [7] M3B: Moment Foundation Model → Embed 1024D → HAC / GMM / HDB  │
│  [8] Stability Analysis (Bootstrap ARI + Temporal coherence)        │
│  → Tất cả biểu đồ kết quả → result/                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Yêu cầu

```bash
# Cài đặt trong conda environment
conda activate torch-cuda12.8

# Packages cơ bản
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm

# Packages cho PP2v2
pip install pywt hdbscan umap-learn

# Packages cho M3A (Conv1D Autoencoder)
pip install torch

# Packages cho M3B (Moment Foundation Model)
pip install momentfm

# Tùy chọn – TimeSeriesKMeans với DTW
pip install tslearn
```

Dữ liệu đầu vào: `data/full_gnss_2e.csv` (cột `Timestamp`, `h_Coord`).

---

## BƯỚC 1 – Tìm số cụm tối ưu

### Lệnh cơ bản

```bash
conda run -n torch-cuda12.8 python step1_find_k.py
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
conda run -n torch-cuda12.8 python step1_find_k.py

# Thu hẹp khoảng k để chạy nhanh hơn
conda run -n torch-cuda12.8 python step1_find_k.py --k-min 2 --k-max 7

# Chỉ tìm k cho Phương pháp 2 (feature-based)
conda run -n torch-cuda12.8 python step1_find_k.py --method2-only

# Môi trường server không có GUI
conda run -n torch-cuda12.8 python step1_find_k.py --no-display
```

### Đầu ra Bước 1

| File | Nội dung |
|------|----------|
| `result/14_optimal_k_analysis.png` | PP1: Silhouette / Calinski / Davies theo k + voting bar |
| `result/F00_optimal_k_features.png` | PP2: Silhouette / Calinski / Davies theo k + voting bar |
| In ra terminal | Bảng metrics từng k, bỏ phiếu, khuyến nghị k cuối |

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
conda run -n torch-cuda12.8 python step2_cluster.py --k1 4 --k2 2
```

### Tất cả tùy chọn

```bash
python step2_cluster.py [OPTIONS]

OPTIONS:
  --k1 INT            Số cụm cho Phương pháp 1 (mặc định: config.DEFAULT_N_CLUSTERS = 2)
  --k2 INT            Số cụm cho PP2, PP2v2, M3A, M3B (mặc định: config.DEFAULT_N_CLUSTERS = 2)
  --method1-only      Chỉ chạy PP1 (bỏ qua PP2, PP2v2, M3A, M3B)
  --method2-only      Chỉ chạy PP2 + PP2v2 + M3A + M3B (bỏ qua PP1)
  --no-display        Không mở cửa sổ (Agg backend)
  --no-cache          Bỏ qua cache, tải lại từ CSV
```

### Ví dụ

```bash
# Chạy tất cả 5 phương pháp
conda run -n torch-cuda12.8 python step2_cluster.py --k1 4 --k2 2

# Chỉ PP1
conda run -n torch-cuda12.8 python step2_cluster.py --k1 4 --method1-only

# PP2 + PP2v2 + M3A + M3B (bỏ PP1 cho nhanh)
conda run -n torch-cuda12.8 python step2_cluster.py --k2 2 --method2-only

# Server/headless
conda run -n torch-cuda12.8 python step2_cluster.py --k1 4 --k2 2 --no-display
```

### Pipeline Bước 2

```
[1/4] Tải / cache dữ liệu
[2/4] Tiền xử lý (Hampel → reshape → Kalman)
[3/4] PP1 – Raw t-SNE (HAC / GMM / KMeans / DBSCAN)           ← dùng k1
[4/6] PP2 – Feature-Based (22 đặc trưng → HAC / GMM / DBSCAN) ← dùng k2
[5/6] PP2v2 – Feature-Based V2 (40 đặc trưng → weighted → HAC / GMM / HDBSCAN / Ensemble)
[6/7] M3A – Conv1D Autoencoder (latent 32D → HAC / GMM / HDBSCAN)
[7/8] M3B – Moment Foundation Model (embed 1024D → PCA 50D → HAC / GMM / HDBSCAN)
[8/8] Stability Analysis (Bootstrap ARI + Temporal coherence cho tất cả PP)
```

> `--k1` chỉ ảnh hưởng PP1. `--k2` ảnh hưởng PP2, PP2v2, M3A, M3B.

### Đầu ra Bước 2

#### Tiền xử lý (chung)

| File | Nội dung |
|------|----------|
| `result/10_hampel_compare_batch.png` | So sánh trước/sau Hampel (25 giờ mẫu) |
| `result/11_raw_hourly_grid.png` | Lưới 25 chuỗi giờ gốc |
| `result/12_hampel_grid.png` | Lưới 25 chuỗi sau Hampel |
| `result/13_kalman_grid.png` | Lưới 25 chuỗi sau Kalman |

#### PP1 – Raw Time-Series Clustering

| File | Nội dung |
|------|----------|
| `result/15_clustering_scatter.png` | Scatter t-SNE 2D tô màu theo cụm (4 thuật toán) |
| `result/16_clustering_metrics.png` | Bar chart Silhouette / Calinski / Davies (4 TT) |
| `result/17_lineplot_hac.png` | Mean ± std chuỗi giờ theo từng cụm (HAC) |
| `result/17_lineplot_gmm.png` | Mean ± std chuỗi giờ theo từng cụm (GMM) |
| `result/17_lineplot_timeseries_kmeans.png` | Mean ± std theo từng cụm (KMeans) |

#### PP2 – Feature-Based Clustering

| File | Nội dung |
|------|----------|
| `result/F01_feature_boxplot.png` | Phân phối 21 đặc trưng đã chuẩn hóa |
| `result/F02_pca_loadings.png` | PCA loadings: đóng góp của đặc trưng vào PC1/PC2 |
| `result/F03_scatter_{hac,gmm,dbscan}.png` | Scatter PCA 2D tô màu theo cụm |
| `result/F04_cluster_profiles_{hac,gmm,dbscan}.png` | Bar chart giá trị trung bình đặc trưng mỗi cụm |
| `result/F05_cluster_ts_{hac,gmm,dbscan}.png` | Chuỗi thời gian trung bình mỗi cụm |

#### PP2v2 – Feature-Based V2

| File | Nội dung |
|------|----------|
| `result/F2_01_feature_weights.png` | Trọng số đặc trưng (Silhouette-guided importance) |
| `result/F2_03_scatter_{hac,gmm,hdbscan,ensemble}.png` | Scatter UMAP 2D tô màu theo cụm |
| `result/F03_scatter_{hac,gmm,hdbscan,ensemble}_v2.png` | Scatter PCA 2D tô màu theo cụm |
| `result/F04_cluster_profiles_{hac,gmm,...}_v2.png` | Bar chart giá trị trung bình đặc trưng mỗi cụm |
| `result/F05_cluster_ts_{hac,gmm,...}_v2.png` | Chuỗi thời gian trung bình mỗi cụm |
| `result/F2_06_co_association.png` | Ma trận co-association (Ensemble clustering) |

#### M3A – Conv1D Autoencoder

| File | Nội dung |
|------|----------|
| `result/M3_01_training_loss.png` | Loss qua các epoch (log scale) |
| `result/M3_02_reconstruction.png` | So sánh tín hiệu gốc vs tái tạo (6 mẫu) |
| `result/M3_03_latent_scatter_{hac,gmm,hdbscan}.png` | Scatter PCA 2D latent space tô màu theo cụm |
| `result/M3_05_cluster_ts_ae_{hac,gmm,hdbscan}.png` | Chuỗi thời gian trung bình mỗi cụm |

#### M3B – Moment Foundation Model

| File | Nội dung |
|------|----------|
| `result/M3_03_latent_scatter_moment_{hac,gmm,hdbscan}.png` | Scatter PCA 2D embedding space tô màu theo cụm |
| `result/M3_05_cluster_ts_moment_{hac,gmm,hdbscan}.png` | Chuỗi thời gian trung bình mỗi cụm |

#### Stability Analysis

| File | Nội dung |
|------|----------|
| `result/S01_bootstrap_stability.png` | Bootstrap ARI distribution (mỗi PP riêng) |
| `result/S02_temporal_coherence.png` | Runs test kết quả (mỗi PP riêng) |

---

## Đọc kết quả Bước 2

### Metrics phân cụm

| Metric | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| Silhouette | Độ tách biệt giữa các cụm | **Cao hơn** (max = 1) |
| Calinski-Harabasz | Tỉ lệ phương sai giữa/trong cụm | **Cao hơn** |
| Davies-Bouldin | Tỉ lệ khoảng cách nội/ngoại cụm | **Thấp hơn** (min = 0) |

### Stability Analysis

| Metric | Ý nghĩa | Đánh giá |
|--------|---------|----------|
| Bootstrap ARI | Adjusted Rand Index qua 100 lần bootstrap 80% | > 0.8: ổn định; < 0.5: không đáng tin |
| Temporal coherence (p-value) | Kiểm tra Wald-Wolfowitz runs test | p < 0.05: có cấu trúc thời gian → cụm có ý nghĩa vật lý |

### Cách đọc biểu đồ

**`F04_cluster_profiles_*.png`**: Bar chart trung bình đặc trưng đã chuẩn hóa. Giúp giải thích vật lý từng cụm:
- `energy_high` cao → giờ có nhiều dao động cao tần
- `autocorr_lag1` cao → chuỗi mượt, tương quan cao
- `trend_slope` lớn → có xu hướng tăng/giảm trong giờ đó
- `hurst` > 0.5 → tín hiệu có xu hướng bền vững (persistent)
- `sample_entropy` cao → tín hiệu phức tạp, khó dự đoán

**`F2_01_feature_weights.png`**: Trọng số Silhouette-guided cho PP2v2. Đặc trưng có importance cao đóng góp nhiều nhất vào phân tách cụm.

**`F2_06_co_association.png`**: Ma trận co-association (Ensemble). Ô càng đỏ = hai mẫu hay được xếp cùng cụm. Khối vuông rõ ràng = cụm ổn định.

**`M3_01_training_loss.png`**: Loss giảm dần → autoencoder học tốt. Loss thấp nhưng Silhouette thấp → biểu diễn latent chưa tốt cho clustering.

**`M3_02_reconstruction.png`**: So sánh tín hiệu gốc (xanh) vs tái tạo (đỏ). Nếu gần nhau → autoencoder nắm bắt được cấu trúc tín hiệu.

---

## Chạy nhanh (toàn bộ pipeline)

Nếu muốn chạy tất cả trong một lệnh (tương đương `main.py` cũ):

```bash
conda run -n torch-cuda12.8 python main.py
```

> `main.py` tự động tìm k tối ưu (Bước 1) và phân cụm ngay (Bước 2) với k đó, không dừng để hỏi. Dùng `step1_find_k.py` + `step2_cluster.py` khi muốn xem xét kỹ và tự quyết định k.

---

## Tùy chỉnh cấu hình

Sửa `gnss_clustering/config.py`:

```python
# Dữ liệu
DATA_PATH             = 'data/full_gnss_2e.csv'  # File CSV đầu vào
MISSING_THRESHOLD     = 0       # % thiếu tối đa (0 = chỉ giữ giờ đầy đủ)
SEED                  = 23      # Random seed

# Tiền xử lý
HAMPEL_WINDOW_SIZE    = 50      # Cửa sổ Hampel filter
HAMPEL_N_SIGMAS       = 1       # Ngưỡng sigma Hampel
RESHAPE_WINDOW_SIZE   = 10      # Giảm chiều: 3600 → 360

# PP1 – t-SNE
PCA_N_COMPONENTS      = 50      # Số thành phần PCA
TSNE_METRIC           = 'l1'    # Khoảng cách t-SNE

# Phân cụm chung
DEFAULT_N_CLUSTERS    = 2       # k mặc định nếu không truyền --k1/--k2
K_RANGE               = (2, 11) # Khoảng k cần thử (end không bao gồm)

# PP2v2 – Feature-Based V2
WAVELET_NAME          = 'db4'   # Wavelet cho DWT
WAVELET_MAX_LEVEL     = 4       # Số mức phân tích wavelet
HDBSCAN_MIN_CLUSTER_SIZE = 5    # HDBSCAN min cluster size
HDBSCAN_MIN_SAMPLES   = 3       # HDBSCAN min samples
UMAP_N_NEIGHBORS      = 15      # UMAP n_neighbors
UMAP_MIN_DIST         = 0.1     # UMAP min_dist

# Stability Analysis
STABILITY_N_ITERATIONS = 100    # Số lần bootstrap
STABILITY_SAMPLE_RATIO = 0.8    # Tỷ lệ mẫu mỗi lần (80%)
```

---

## Sơ đồ kiến trúc module

```
gnss_clustering/
├── config.py                Đường dẫn, hyperparameter
├── data_loader.py           load_data, create_daily_matrix, create_hourly_matrix
│                            load_cached_matrices
├── preprocessing.py         hampel_filter, kalman_filter_2d, preprocess_pipeline
├── feature_extraction.py    PP1: scale_data → apply_pca → apply_tsne → extract_features
├── clustering.py            run_timeseries_kmeans, run_hierarchical_clustering
│                            run_dbscan_clustering, run_gmm_clustering, run_all
├── feature_engineering.py   PP2: extract_feature_matrix (22 đặc trưng cơ bản)
│                                 preprocess_features, run_feature_based_pipeline
│                            PP2v2: extract_feature_matrix(extended=True) (40 đặc trưng)
│                                   preprocess_features_v2 (PowerTransformer + RobustScaler)
│                                   _silhouette_feature_weighting, _ensemble_clustering
│                                   reduce_features_umap, run_feature_based_pipeline_v2
│                            Tìm k: find_optimal_clusters_features
├── deep_clustering.py       M3A: Conv1DAutoencoder, train_autoencoder
│                                 run_autoencoder_pipeline
│                            M3B: extract_moment_embeddings, run_moment_pipeline
│                            Chung: cluster_latent_space (HAC / GMM / HDBSCAN)
│                            Viz: plot_training_loss, plot_reconstruction,
│                                 plot_latent_scatter, plot_cluster_timeseries_deep
├── optimization.py          PP1: find_optimal_clusters, create_optimization_plots
├── stability.py             bootstrap_stability, temporal_coherence
│                            run_stability_analysis, plot_stability_results
├── visualization.py         20+ hàm vẽ → lưu result/
└── __init__.py              Exports tất cả module
```

---

## Giải thích đặc trưng

### 22 đặc trưng cơ bản (PP2)

| Nhóm | Đặc trưng | Ý nghĩa |
|------|-----------|---------|
| **Thống kê** (5) | `mean` | Mức trung bình dịch chuyển trong giờ |
| | `std` | Độ biến động tổng thể |
| | `skewness` | Độ lệch phân phối |
| | `kurtosis` | Độ nhọn (nhiều giá trị ngoại lai?) |
| | `iqr` | Khoảng tứ vị phân (robust spread) |
| **Xu hướng** (4) | `trend_slope` | Tốc độ tăng/giảm tuyến tính (mm/s) |
| | `trend_intercept` | Giá trị khởi đầu của trend |
| | `trend_r2` | Mức độ phù hợp của trend tuyến tính |
| | `trend_resid_std` | Độ phân tán quanh trend |
| **Tần số** (5) | `dominant_freq` | Tần số dao động chủ đạo |
| | `spectral_entropy` | Mức độ phân tán năng lượng tần số |
| | `energy_low` | Tỉ lệ năng lượng tần số thấp (< 10% Nyquist) |
| | `energy_mid` | Tỉ lệ năng lượng tần số trung |
| | `energy_high` | Tỉ lệ năng lượng tần số cao (> 30% Nyquist) |
| **Cấu trúc TG** (4) | `autocorr_lag1/5/30` | Tương quan tự hồi quy tại các lag |
| | `hurst` | H<0.5: mean-reverting; H≈0.5: ngẫu nhiên; H>0.5: persistent |
| **Chất lượng** (4) | `valid_ratio` | Tỉ lệ điểm không thiếu |
| | `outlier_ratio` | Tỉ lệ ngoại lai theo Hampel |
| | `signal_range` | Biên độ tín hiệu (max−min) |
| | `snr` | Tỉ số tín hiệu / nhiễu |

### 18 đặc trưng mở rộng (PP2v2, thêm khi `extended=True`)

| Nhóm | Đặc trưng | Ý nghĩa |
|------|-----------|---------|
| **Wavelet** (10) | `wavelet_energy_d1..d4` | Tỉ lệ năng lượng mỗi mức detail (DWT db4) |
| | `wavelet_entropy_d1..d4` | Entropy tại mỗi mức detail |
| | `wavelet_energy_approx` | Tỉ lệ năng lượng approximation |
| | `wavelet_detail_var_ratio` | Phương sai detail / phương sai tổng |
| **Complexity** (5) | `sample_entropy` | Độ phức tạp mẫu (template matching) |
| | `permutation_entropy` | Entropy hoán vị (tính ngẫu nhiên) |
| | `zero_crossing_rate` | Tỉ lệ đổi dấu quanh trung bình |
| | `mean_abs_diff` | Độ gồ ghề (roughness) bậc 1 |
| | `coeff_variation` | Hệ số biến thiên (std/mean) |
| **Stationarity** (3) | `mean_shift` | Biến đổi trung bình giữa 4 đoạn con |
| | `var_shift` | Biến đổi phương sai giữa 4 đoạn con |
| | `trend_strength` | 1 - var(residuals)/var(x) |

---

## Câu hỏi thường gặp

**Q: Silhouette của PP2 (0.25) thấp hơn PP1 (0.55). PP2 tệ hơn không?**

A: Không. PP1 dùng t-SNE 2D (đã tối ưu hóa để tách cụm), nên Silhouette tự nhiên cao. PP2/PP2v2 làm việc trong không gian đặc trưng thực – ranh giới cụm mờ hơn nhưng có ý nghĩa vật lý rõ ràng. PP2v2 Ensemble đạt Silhouette 0.329 (cải thiện +31% so với PP2 gốc).

**Q: PP2v2 khác PP2 gốc như thế nào?**

A: PP2v2 cải tiến ở 4 khía cạnh: (1) thêm 18 đặc trưng wavelet/complexity/stationarity, (2) chuẩn hóa bằng PowerTransformer + RobustScaler thay vì StandardScaler, (3) trọng số đặc trưng dựa trên Silhouette importance, (4) thêm HDBSCAN và Ensemble clustering.

**Q: M3B (Moment) cho kết quả kém. Tại sao vẫn chạy?**

A: Moment là pre-trained model domain chung, kết quả kém cho GNSS (Sil=0.066) là thông tin hữu ích – cho thấy zero-shot embedding không phù hợp. Nếu không cần, dùng `--method1-only` để bỏ qua, hoặc sửa code trong `step2_cluster.py` để bỏ bước [7/8].

**Q: Cache là gì và khi nào cần `--no-cache`?**

A: Sau lần chạy đầu tiên, dữ liệu trung gian được lưu vào `data/` (`.npy`, `.csv`). Các lần sau sẽ tải cache thay vì tính lại từ CSV. Dùng `--no-cache` khi thay đổi file CSV đầu vào hoặc thay đổi `MISSING_THRESHOLD`.

**Q: `tslearn` không cài được. Có ảnh hưởng không?**

A: Không. Khi không có `tslearn`, `run_timeseries_kmeans` tự động fallback sang `sklearn.KMeans`. Kết quả vẫn đầy đủ, chỉ mất phần DTW distance.

**Q: Chạy trên server không có màn hình?**

A: Thêm `--no-display` vào lệnh. Backend sẽ chuyển sang `Agg` (chỉ lưu file, không hiển thị).

**Q: Chạy mất bao lâu?**

A: Phụ thuộc vào phương pháp. PP1 (t-SNE) và M3B (load Moment model) chậm nhất. Nếu chỉ cần PP2/PP2v2, dùng `--method2-only` để bỏ PP1. Toàn bộ pipeline trên CPU mất khoảng vài phút.

**Q: Stability Analysis cho biết gì?**

A: Hai thông tin: (1) Bootstrap ARI cho biết kết quả phân cụm có ổn định khi lấy mẫu lại không (ARI > 0.8 = ổn định). (2) Temporal coherence cho biết các giờ cùng cụm có xu hướng liên tiếp theo thời gian không (p < 0.05 = có cấu trúc thời gian → cụm có ý nghĩa vật lý).
