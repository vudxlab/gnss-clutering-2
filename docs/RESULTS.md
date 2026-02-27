# Kết quả phân cụm & Phân tích so sánh

**Dữ liệu:** `full_gnss_2e.csv` · trạm GNSS 2E · 29/05/2015 – 11/06/2015

---

## 1. Mô tả dữ liệu

| Thông số | Giá trị |
|----------|---------|
| Tổng số mẫu | 897,808 |
| Số ngày | 13 (2015-05-29 → 2015-06-11) |
| Tần số lấy mẫu | 1 Hz (1 mẫu/giây) |
| Kênh phân tích | `h_Coord` – tọa độ thẳng đứng (m) |
| Số giờ tiềm năng | 312 (13 ngày × 24 giờ) |
| **Giờ hợp lệ** (0% thiếu) | **191 / 312 (61.2%)** |
| Tỷ lệ dữ liệu có trong toàn bộ | 79.93% |

### Phân bố số giờ hợp lệ theo ngày

| Ngày | Số điểm | Giờ hợp lệ |
|------|---------|-----------|
| 2015-05-29 | 25,186 | Một phần ngày (bắt đầu 02:41) |
| 2015-05-30 | 77,852 | Cao |
| 2015-05-31 | 44,474 | Trung bình |
| 2015-06-02 | 76,874 | Cao (không có 01/06) |
| 2015-06-03 → 06-10 | ~85,000–86,000/ngày | Gần đầy đủ |
| 2015-06-11 | 16,926 | Một phần ngày (kết thúc 04:42) |

---

## 2. Pipeline tiền xử lý

```
hourly_matrix (191, 3600)          ← 191 giờ hợp lệ × 3600 giây/giờ
      │
      ▼  Hampel filter (window=50, n_sigma=1)
hampel_data (191, 3600)
      │
      ▼  Reshape: trung bình cửa sổ 10 giây
reshape_data (191, 360)            ← giảm từ 3600 → 360 chiều
      │
      ▼  Kalman filter
data_filtered (191, 360)
```

---

## 3. Phương pháp 1 – Cluster trên chuỗi thô (Raw Time-Series)

### Pipeline

```
data_filtered (191, 360)
      │
      ▼  StandardScaler (chuẩn hóa theo cột)
data_scaled (191, 360)
      │
      ▼  PCA (50 thành phần, giải thích 100% phương sai)
data_pca (191, 50)
      │
      ▼  t-SNE (metric=L1, perplexity=30, 2 lần fit_transform)
data_tsne (191, 2)
      │
      ▼  HAC / GMM / DBSCAN / TimeSeriesKMeans
   Nhãn phân cụm
```

### Kết quả (k = 4)

| Thuật toán | k | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ |
|------------|---|:---:|:---:|:---:|
| **HAC** (Ward) | 4 | **0.5549** | **392.94** | **0.5995** |
| **GMM** (full) | 4 | 0.5541 | 390.95 | 0.6018 |
| DBSCAN | 5 | 0.3878 | 152.76 | 0.7278 |
| KMeans *(fallback)* | 4 | 0.2560 | 74.18 | 1.3574 |

> **HAC** và **GMM** cho kết quả tốt nhất, gần như tương đương nhau.

### Tìm k tối ưu (voting system)

| k | HAC Sil | GMM Sil | Votes |
|---|---------|---------|-------|
| **4** | **0.555** | **0.554** | **2** (Sil HAC + Sil GMM) |
| 5 | 0.504 | 0.532 | 3 (DBSCAN tự nhiên) |
| 9 | 0.495 | — | 1 (Calinski HAC) |

**→ Đề xuất: k = 4** (HAC và GMM đồng thuận ở Silhouette cao nhất).

---

## 4. Phương pháp 2 – Feature-Based Clustering

### Vector đặc trưng (22 chiều, giữ lại 21 sau lọc phương sai)

Từ mỗi đoạn 1 giờ, trích xuất 22 đặc trưng vật lý:

| Nhóm | Đặc trưng | Diễn giải |
|------|-----------|-----------|
| **Thống kê** (5) | `mean`, `std`, `skewness`, `kurtosis`, `iqr` | Phân phối giá trị h_Coord |
| **Xu hướng** (4) | `trend_slope`, `trend_intercept`, `trend_r2`, `trend_resid_std` | Dịch chuyển có xu hướng tuyến tính không? |
| **Phổ tần số** (5) | `dominant_freq`, `spectral_entropy`, `energy_low`, `energy_mid`, `energy_high` | Thành phần tần số nào chiếm ưu thế? |
| **Cấu trúc thời gian** (4) | `autocorr_lag1/5/30`, `hurst` | Tính nhớ, persistent vs. mean-reverting |
| **Chất lượng tín hiệu** (4) | `valid_ratio`, `outlier_ratio`, `signal_range`, `snr` | Mức độ nhiễu, biên độ dao động |

> *`valid_ratio` bị loại vì phương sai ≈ 0 (tất cả giờ đều có 100% dữ liệu).*

### Pipeline

```
hourly_matrix (191, 3600)  +  hampel_data (191, 3600)
      │
      ▼  extract_feature_matrix(fs=1.0)
feature_df (191, 22)
      │
      ▼  StandardScaler  →  loại cột phương sai ≈ 0
X_scaled (191, 21)
      │
      ▼  PCA 2D  →  giải thích 59.8% phương sai
X_pca (191, 2)
      │
      ▼  HAC / GMM / DBSCAN
   Nhãn phân cụm
```

### Kết quả (k = 2)

| Thuật toán | k | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ |
|------------|---|:---:|:---:|:---:|
| GMM | 2 | 0.2512 | 59.26 | 1.5881 |
| HAC | 2 | 0.2344 | 63.11 | 1.5980 |
| DBSCAN | 1 | -1.000 | -1.00 | -1.000 |

### Stability Analysis (k = 2)

| Method | ARI mean ± std | Temporal p-value | Ổn định? |
|--------|:-:|:-:|:-:|
| PP2_HAC | 1.000 ± 0.000 | 0.0000 | **Có** |
| PP2_GMM | 0.917 ± 0.220 | 0.1021 | Có |

---

## 5. Phương pháp 2v2 – Feature-Based V2 (cải tiến)

### Cải tiến so với PP2

| Khía cạnh | PP2 (gốc) | PP2v2 (cải tiến) |
|-----------|-----------|-------------------|
| Số đặc trưng | 22 (cơ bản) | 40 (22 cơ bản + 18 mở rộng) |
| Đặc trưng mới | — | Wavelet (DWT db4), Complexity (sample/permutation entropy), Stationarity |
| Chuẩn hóa | StandardScaler | PowerTransformer (Yeo-Johnson) + RobustScaler |
| Feature weighting | Không | Silhouette-guided permutation importance |
| Thuật toán | HAC, GMM, DBSCAN | HAC, GMM, HDBSCAN, **Ensemble** (co-association matrix) |
| Giảm chiều visualize | PCA 2D | PCA 2D + UMAP 2D |

### 18 đặc trưng mở rộng

| Nhóm | Đặc trưng | Diễn giải |
|------|-----------|-----------|
| **Wavelet** (10) | `wavelet_energy_d1..d4`, `wavelet_entropy_d1..d4`, `wavelet_energy_approx`, `wavelet_detail_var_ratio` | Phân tích đa phân giải DWT (db4, 4 mức) |
| **Complexity** (5) | `sample_entropy`, `permutation_entropy`, `zero_crossing_rate`, `mean_abs_diff`, `coeff_variation` | Độ phức tạp, tính ngẫu nhiên |
| **Stationarity** (3) | `mean_shift`, `var_shift`, `trend_strength` | Tính dừng: so sánh thống kê giữa 4 đoạn con |

### Pipeline

```
hourly_matrix (191, 3600)  +  hampel_data (191, 3600)
      │
      ▼  extract_feature_matrix(extended=True)
feature_df (191, 40)
      │
      ▼  PowerTransformer (Yeo-Johnson) + RobustScaler → loại cột phương sai ≈ 0
X_scaled (191, 34)
      │
      ▼  Initial HAC (k=2) → Silhouette-guided feature weighting
X_weighted (191, 34)           ← trọng số [0.5, 1.5]
      │
      ▼  HAC / GMM / HDBSCAN / Ensemble (10 runs x 2 methods)
   Nhãn phân cụm
      │
      ▼  UMAP 2D + PCA 2D (visualize)
```

### Top 5 đặc trưng quan trọng nhất (Silhouette-guided weighting)

| Đặc trưng | Importance | Weight |
|-----------|:---------:|:------:|
| sample_entropy | 0.0135 | 1.500 |
| energy_mid | 0.0128 | 1.444 |
| autocorr_lag1 | 0.0125 | 1.427 |
| autocorr_lag5 | 0.0125 | 1.427 |
| zero_crossing_rate | 0.0117 | 1.366 |

### Kết quả (k = 2)

| Thuật toán | k | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ |
|------------|---|:---:|:---:|:---:|
| **Ensemble** | 2 | **0.3289** | **141.75** | **1.1074** |
| GMM | 2 | 0.3247 | 140.16 | 1.1190 |
| HAC | 2 | 0.3228 | 137.27 | 1.1146 |
| HDBSCAN | 2 | 0.2419 | 21.84 | 0.9097 |

### So sánh PP2 → PP2v2

| Metric | PP2 (GMM) | PP2v2 (Ensemble) | Cải thiện |
|--------|:---------:|:----------------:|:---------:|
| Silhouette | 0.251 | 0.329 | **+31%** |
| Calinski-Harabasz | 59.3 | 141.8 | **+139%** |
| Davies-Bouldin | 1.588 | 1.107 | **-30%** (tốt hơn) |

### Stability Analysis (k = 2)

| Method | ARI mean ± std | Temporal p-value | Ổn định? |
|--------|:-:|:-:|:-:|
| PP2v2_HAC | 1.000 ± 0.000 | 0.0000 | **Có** |
| PP2v2_Ensemble | 1.000 ± 0.000 | 0.0000 | **Có** |
| PP2v2_GMM | 0.948 ± 0.097 | 0.0001 | **Có** |

---

## 6. Method 3A – Conv1D Autoencoder

### Kiến trúc

```
Input (batch, 1, 360)
      │
      ▼  Encoder: Conv1D 4 tầng
         Conv1d(1→16, k=7, s=2)  → (16, 180) → BN → ReLU
         Conv1d(16→32, k=5, s=2) → (32, 90)  → BN → ReLU
         Conv1d(32→64, k=5, s=2) → (64, 45)  → BN → ReLU
         Conv1d(64→128, k=3, s=2)→ (128, 23) → BN → ReLU
      │
      ▼  Flatten → FC(2944→256) → ReLU → FC(256→32)
Latent (batch, 32)
      │
      ▼  Decoder: FC → ConvTranspose1d 4 tầng (đối xứng)
Output (batch, 1, 360)
```

### Pipeline

```
hampel_data (191, 3600)
      │
      ▼  Interpolate NaN → reshape 3600 → 360 (avg window 10)
data_reshaped (191, 360)
      │
      ▼  StandardScaler → Train Conv1D Autoencoder (MSE loss, Adam, 100 epochs)
      │  Device: CPU, lr=1e-3, batch_size=32
      │
      ▼  Encode → latent vectors
latent_vectors (191, 32)
      │
      ▼  StandardScaler → HAC / GMM / HDBSCAN
   Nhãn phân cụm
```

### Training

| Epoch | MSE Loss |
|-------|:--------:|
| 1 | 1.6218 |
| 20 | 0.1746 |
| 40 | 0.0762 |
| 60 | 0.0620 |
| 80 | 0.0343 |
| 100 | 0.0310 |

### Kết quả (k = 2)

| Thuật toán | k | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ |
|------------|---|:---:|:---:|:---:|
| HDBSCAN | 2 | 0.2892 | 13.34 | 1.2969 |
| GMM | 2 | 0.2262 | 56.65 | 1.6753 |
| HAC | 2 | 0.2110 | 53.15 | 1.6887 |

> HDBSCAN cho Silhouette cao nhất nhưng noise=162/191 (84.8%) – chỉ giữ 29 mẫu trong 2 cụm.

### Stability Analysis (k = 2)

| Method | ARI mean ± std | Temporal p-value | Ổn định? |
|--------|:-:|:-:|:-:|
| M3A_HAC | 1.000 ± 0.000 | 0.0000 | **Có** |
| M3A_GMM | 0.955 ± 0.169 | 0.0000 | **Có** |

---

## 7. Method 3B – Moment Foundation Model

### Pipeline

```
hampel_data (191, 3600)
      │
      ▼  Interpolate NaN → reshape 3600 → 360 (avg window 10)
data_reshaped (191, 360)
      │
      ▼  StandardScaler → Pad 360 → 512 (Moment yêu cầu)
      │
      ▼  AutonLab/MOMENT-1-large (zero-shot, task="embedding")
      │  model.embed(x_enc=..., input_mask=...)
embeddings (191, 1024)
      │
      ▼  StandardScaler → PCA (1024D → 50D, 83.7% variance)
Z (191, 50)
      │
      ▼  HAC / GMM / HDBSCAN
   Nhãn phân cụm
```

> **Lưu ý:** PCA reduction bắt buộc vì n_dim (1024) >> n_samples (191).
> GMM fallback từ `full` → `diag` covariance nếu ma trận ill-defined.

### Kết quả (k = 2)

| Thuật toán | k | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ |
|------------|---|:---:|:---:|:---:|
| HDBSCAN | 2 | 0.2572 | 8.16 | 1.2046 |
| GMM | 2 | 0.0767 | 18.55 | 3.1059 |
| HAC | 2 | 0.0661 | 15.18 | 3.4718 |

> Silhouette rất thấp – pre-trained model (domain chung) không phù hợp với tín hiệu GNSS.
> HDBSCAN: noise=167/191 (87.4%) – hầu hết mẫu bị coi là noise.

### Stability Analysis (k = 2)

| Method | ARI mean ± std | Temporal p-value | Ổn định? |
|--------|:-:|:-:|:-:|
| M3B_HAC | 1.000 ± 0.000 | 0.3639 | Có (nhưng không có cấu trúc thời gian) |
| M3B_GMM | 0.651 ± 0.240 | 0.0186 | Tương đối |

---

## 8. Bảng tổng hợp tất cả phương pháp (k = 2)

### Metrics phân cụm

| Phương pháp | Thuật toán | Silhouette ↑ | Calinski ↑ | Davies ↓ | Noise |
|-------------|-----------|:---:|:---:|:---:|:---:|
| **PP2v2** | **Ensemble** | **0.329** | **141.8** | **1.107** | 0 |
| PP2v2 | GMM | 0.325 | 140.2 | 1.119 | 0 |
| PP2v2 | HAC | 0.323 | 137.3 | 1.115 | 0 |
| M3A (AE) | HDBSCAN | 0.289 | 13.3 | 1.297 | 162 |
| M3B (Moment) | HDBSCAN | 0.257 | 8.2 | 1.205 | 167 |
| PP2 | GMM | 0.251 | 59.3 | 1.588 | 0 |
| PP2v2 | HDBSCAN | 0.242 | 21.8 | 0.910 | 13 |
| PP2 | HAC | 0.234 | 63.1 | 1.598 | 0 |
| M3A (AE) | GMM | 0.226 | 56.7 | 1.675 | 0 |
| M3A (AE) | HAC | 0.211 | 53.2 | 1.689 | 0 |
| M3B (Moment) | GMM | 0.077 | 18.5 | 3.106 | 0 |
| M3B (Moment) | HAC | 0.066 | 15.2 | 3.472 | 0 |

### Stability Analysis

| Method | ARI mean ± std | Temporal p-value | Ổn định? | Cấu trúc TG? |
|--------|:-:|:-:|:-:|:-:|
| PP2_HAC | 1.000 ± 0.000 | 0.0000 | **Có** | **Có** |
| PP2v2_HAC | 1.000 ± 0.000 | 0.0000 | **Có** | **Có** |
| PP2v2_Ensemble | 1.000 ± 0.000 | 0.0000 | **Có** | **Có** |
| M3A_HAC | 1.000 ± 0.000 | 0.0000 | **Có** | **Có** |
| M3A_GMM | 0.955 ± 0.169 | 0.0000 | **Có** | **Có** |
| PP2v2_GMM | 0.948 ± 0.097 | 0.0001 | **Có** | **Có** |
| PP2_GMM | 0.917 ± 0.220 | 0.1021 | Có | Không |
| M3B_HAC | 1.000 ± 0.000 | 0.3639 | Có | Không |
| M3B_GMM | 0.651 ± 0.240 | 0.0186 | Tương đối | **Có** |

---

## 9. So sánh các phương pháp

### Bảng so sánh tổng hợp

| Tiêu chí | PP1 | PP2 | PP2v2 | M3A (AE) | M3B (Moment) |
|----------|:---:|:---:|:---:|:---:|:---:|
| **Silhouette tốt nhất** | **0.555** (k=4) | 0.251 | **0.329** | 0.289 | 0.257 |
| **k tối ưu** | 4 | 2 | 2 | 2 | 2 |
| **Không gian cluster** | 2D t-SNE | 21D đặc trưng | 34D có trọng số | 32D latent | 50D PCA(1024D) |
| **Khả năng giải thích** | Thấp | Cao | **Rất cao** | Thấp | Thấp |
| **Bất biến pha** | Không | **Có** | **Có** | Không | Không |
| **Stability (ARI)** | — | 1.000 | **1.000** | 1.000 | 0.651–1.000 |
| **Temporal coherence** | — | p=0.0000 | **p=0.0000** | p=0.0000 | p=0.3639 |
| **Tốc độ** | Chậm (t-SNE) | Nhanh | Trung bình | Chậm (train AE) | Chậm (load model) |
| **Cần GPU** | Không | Không | Không | Tùy chọn | Tùy chọn |

### Phân tích chi tiết

#### PP2v2 – Phương pháp tốt nhất tổng thể

- **Silhouette cao nhất** trong nhóm k=2: 0.329 (Ensemble), vượt PP2 gốc +31%
- **Calinski-Harabasz cao nhất**: 141.8 (gấp 2.4 lần PP2 gốc)
- **Stability tuyệt đối**: ARI = 1.000, temporal coherence p = 0.0000
- **Giải thích được**: biết đặc trưng nào quan trọng nhất (sample_entropy, energy_mid, autocorr)
- **Ensemble clustering**: kết hợp 10 lần HAC + 10 lần GMM với nhiễu nhỏ → nhãn ổn định

#### PP1 – Silhouette cao nhất (k=4) nhưng hạn chế

- Silhouette 0.555 cao do t-SNE tách các nhóm theo giá trị tuyệt đối h_Coord
- Không bất biến pha, không tái lập, không giải thích được
- Phù hợp khi mục tiêu là **phân tầng theo mức dịch chuyển**

#### M3A (Conv1D AE) – Tiềm năng nhưng chưa vượt PP2v2

- Latent 32D học được biểu diễn nén, loss giảm từ 1.62 → 0.03
- Silhouette trung bình (0.211–0.289), stability cao (ARI = 1.000)
- Temporal coherence rất mạnh (p = 0.0000) → cụm có ý nghĩa vật lý
- Có thể cải thiện bằng: tăng epochs, thêm regularization, fine-tune latent_dim

#### M3B (Moment) – Không phù hợp với GNSS

- Pre-trained trên domain chung (NLP/general time series), không hiểu đặc thù GNSS
- Silhouette rất thấp (0.066–0.077 cho HAC/GMM)
- HDBSCAN loại 87% mẫu thành noise
- Temporal coherence yếu (p = 0.364 cho HAC) → nhãn gần như ngẫu nhiên theo thời gian
- **Kết luận: zero-shot embedding không hiệu quả cho domain GNSS**

---

## 10. Kết luận & Đề xuất

### Phương pháp đề xuất sử dụng

**PP2v2 (Feature-Based V2) với Ensemble clustering, k = 2** là lựa chọn tốt nhất:
- Metrics tốt nhất trong nhóm k=2
- Ổn định tuyệt đối qua bootstrap (ARI = 1.000)
- Có cấu trúc thời gian rõ ràng (p = 0.0000)
- Giải thích được qua trọng số đặc trưng và cluster profiles

### Diễn giải 2 cụm (PP2v2 Ensemble, k=2)

| Cụm | Đặc trưng nổi bật | Diễn giải vật lý |
|-----|-------------------|-----------------|
| **Cluster 0** | `energy_high` ↑, `sample_entropy` ↑, `autocorr_lag1` ↓ | **Dao động tần số cao, phức tạp** – tín hiệu nhiễu, không ổn định |
| **Cluster 1** | `energy_low` ↑, `autocorr_lag1` ↑, `sample_entropy` ↓ | **Dao động tần số thấp, đơn giản** – tín hiệu trơn, xu hướng dài hạn ổn định |

### Hướng cải tiến tiếp theo

| Hướng | Mô tả | Ưu tiên |
|-------|-------|:-------:|
| Fine-tune M3A | Tăng epochs, thử VAE, contrastive learning | Trung bình |
| Multi-axis features | Tích hợp X_Coord, Y_Coord vào vector đặc trưng | Cao |
| Domain-specific fine-tune | Fine-tune Moment trên dữ liệu GNSS (supervised/self-supervised) | Thấp |
| Sliding window features | Chia 1 giờ thành 6 cửa sổ 10 phút, tính đặc trưng biến đổi nội giờ | Trung bình |
| Cross-station analysis | So sánh cụm giữa nhiều trạm GNSS | Cao |
