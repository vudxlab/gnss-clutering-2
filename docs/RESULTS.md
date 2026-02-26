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
> DBSCAN phát hiện thêm 1 cụm (5 cụm, 2 noise points).
> KMeans thấp do không dùng được DTW (thiếu `tslearn`).

### Phân bố số giờ theo cụm (HAC, k=4)

| Cụm | Số giờ | Tỷ lệ | Đặc điểm quan sát |
|-----|--------|-------|-------------------|
| Cluster 0 | 44 | 23.0% | Nhóm nhỏ, giá trị h đặc trưng riêng |
| Cluster 1 | 58 | 30.4% | Nhóm lớn nhất |
| Cluster 2 | 33 | 17.3% | Nhóm nhỏ nhất |
| Cluster 3 | 56 | 29.3% | Nhóm lớn thứ hai |

### Tìm k tối ưu (voting system)

| k | HAC Sil | GMM Sil | Votes |
|---|---------|---------|-------|
| **4** | **0.555** | **0.554** | **2** (Sil HAC + Sil GMM) |
| 5 | 0.504 | 0.532 | 3 (DBSCAN tự nhiên) |
| 9 | 0.495 | — | 1 (Calinski HAC) |

**→ Đề xuất: k = 4** (HAC và GMM đồng thuận ở Silhouette cao nhất).

---

## 4. Phương pháp 2 – Feature-Based Clustering

### Vector đặc trưng (21 chiều)

Từ mỗi đoạn 1 giờ, trích xuất 21 đặc trưng vật lý:

| Nhóm | Đặc trưng | Diễn giải |
|------|-----------|-----------|
| **Thống kê** (5) | `mean`, `std`, `skewness`, `kurtosis`, `iqr` | Phân phối giá trị h_Coord |
| **Xu hướng** (4) | `trend_slope`, `trend_intercept`, `trend_r2`, `trend_resid_std` | Dịch chuyển có xu hướng tuyến tính không? |
| **Phổ tần số** (5) | `dominant_freq`, `spectral_entropy`, `energy_low`, `energy_mid`, `energy_high` | Thành phần tần số nào chiếm ưu thế? |
| **Cấu trúc thời gian** (4) | `autocorr_lag1/5/30`, `hurst` | Tính nhớ, persistent vs. mean-reverting |
| **Chất lượng tín hiệu** (3) | `outlier_ratio`, `signal_range`, `snr` | Mức độ nhiễu, biên độ dao động |

> *`valid_ratio` bị loại vì phương sai ≈ 0 (tất cả giờ đều có 100% dữ liệu).*

### Pipeline

```
hourly_matrix (191, 3600)  +  hampel_data (191, 3600)
      │
      ▼  extract_feature_matrix()
feature_df (191, 21)
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

### Kết quả theo k

| k | HAC Sil | HAC Cal | HAC Dav | GMM Sil | GMM Cal | GMM AIC |
|---|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| **2** | **0.234** | **63.11** | **1.598** | **0.251** | **59.26** | **−1425** |
| 3 | 0.148 | 52.57 | 1.763 | 0.178 | 55.78 | −1816 |
| 4 | 0.150 | 46.42 | 1.748 | 0.135 | 49.41 | −1927 |

> **→ Đề xuất: k = 2** (cả HAC lẫn GMM đều đạt Silhouette cao nhất tại k=2).
> AIC giảm theo k (đặc trưng của GMM), nhưng Silhouette tăng ngược lại → k=2 là điểm cân bằng.

### Diễn giải 2 cụm (HAC, k=2)

| Cụm | Số giờ | Đặc trưng nổi bật (chuẩn hóa) | Diễn giải vật lý |
|-----|--------|-------------------------------|-----------------|
| **Cluster 0** | 121 (63.4%) | `energy_high` ↑ (+0.50), `energy_low` ↓ (−0.50), `autocorr_lag1` ↓ (−0.50) | **Dao động tần số cao, ít tương quan liên tiếp** → tín hiệu nhiễu ngắn hạn, không ổn định |
| **Cluster 1** | 70 (36.6%) | `energy_low` ↑ (+0.87), `autocorr_lag1` ↑ (+0.87), `energy_high` ↓ (−0.87) | **Dao động tần số thấp, tương quan cao** → tín hiệu trơn, xu hướng dài hạn ổn định |

---

## 5. So sánh hai phương pháp

### Bảng so sánh tổng hợp

| Tiêu chí | Phương pháp 1 | Phương pháp 2 |
|----------|:---:|:---:|
| **Silhouette tốt nhất** | **0.555** | 0.251 |
| **k tối ưu** | 4 | 2 |
| **Không gian phân cụm** | 2D t-SNE (từ 360 chiều) | 21 đặc trưng vật lý |
| **PCA 2D giải thích** | — | 59.8% |
| **Khả năng giải thích** | Thấp | **Cao** |
| **Bất biến pha** | Không | **Có** |
| **Phụ thuộc giá trị tuyệt đối** | **Có** | Không |
| **Tốc độ tính toán** | Chậm (t-SNE 2 lần) | Nhanh |
| **Cần tinh chỉnh tham số** | Nhiều (perplexity, metric) | Ít |

### Tại sao Silhouette của P.pháp 2 thấp hơn?

Điều này **không có nghĩa là P.pháp 2 kém hơn**. Có 3 lý do:

**1. P.pháp 1 cluster theo giá trị tuyệt đối:**
```
t-SNE chiếu (191, 360) → (191, 2) giữ khoảng cách L1.
Các giờ có h_Coord ~ 14.5m tự nhiên tách biệt với nhóm ~ 15.2m.
→ Khoảng cách Euclidean giữa cụm lớn → Silhouette cao.
→ Nhưng hai giờ cùng cụm chỉ vì cùng mức h,
   dù một giờ ổn định và một giờ rung lắc mạnh.
```

**2. P.pháp 2 cluster theo hành vi trong không gian 21 chiều:**
```
21 đặc trưng mô tả nhiều khía cạnh khác nhau (tần số, xu hướng, nhiễu...).
Ranh giới hành vi tự nhiên mờ hơn ranh giới giá trị tuyệt đối.
→ Silhouette thấp hơn nhưng mỗi cụm có ý nghĩa vật lý rõ ràng.
```

**3. PCA 2D chỉ giải thích 59.8% phương sai:**
```
Metrics (Silhouette) tính trên X_scaled (21D).
PCA 2D mất 40.2% thông tin → scatter plot 2D không phản ánh
đầy đủ khoảng cách thực trong không gian đặc trưng.
```

### Nhận xét chi tiết

#### Ưu điểm Phương pháp 1
- Silhouette cao (0.555): các cụm được phân tách tốt trong không gian t-SNE
- Phát hiện được sự phân tầng theo mức độ dịch chuyển tuyệt đối (4 mức h)
- Phù hợp khi mục tiêu là **nhận dạng giai đoạn dịch chuyển** (settling, loading, unloading...)

#### Nhược điểm Phương pháp 1
- **Mất thứ tự thời gian:** t-SNE không bảo toàn cấu trúc toàn cục; hai điểm gần nhau trong 2D có thể xa nhau trong thực tế
- **Không tái lập (non-deterministic):** t-SNE cho kết quả khác nhau mỗi lần chạy
- **Nhạy cảm với trung bình:** hai chuỗi cùng giá trị trung bình nhưng hành vi khác nhau (một phẳng, một dao động) bị xếp cùng cụm
- **Không giải thích được:** nhãn "Cluster 2" không nói lên điều gì về bản chất vật lý
- **Chi phí tính toán:** t-SNE O(n²), chạy 2 lần trên 191 mẫu

#### Ưu điểm Phương pháp 2
- **Giải thích được:** Cluster 1 = "giờ có tần số thấp, ổn định"; Cluster 0 = "giờ nhiễu, tần số cao"
- **Bất biến pha và mức:** hai chuỗi cùng hành vi nhưng khác mức h vẫn vào cùng cụm
- **Tái lập hoàn toàn:** không có thành phần ngẫu nhiên (ngoài GMM seed)
- **Dễ tích hợp kiến thức chuyên môn:** có thể thêm/bỏ/điều chỉnh trọng số đặc trưng
- **Hiệu quả:** tính đặc trưng O(n), không cần t-SNE

#### Nhược điểm Phương pháp 2
- Silhouette thấp hơn (0.25): ranh giới cụm không sắc nét trong 21D
- PCA 2D chỉ giải thích 59.8%: visualization 2D mất thông tin
- Phụ thuộc vào lựa chọn đặc trưng: cần hiểu biết domain để chọn đặc trưng phù hợp
- Hurst exponent và spectral entropy cần đủ dữ liệu để ước lượng chính xác

---

## 6. Đề xuất hướng cải tiến

### 6.1 Kết hợp hai phương pháp (Ensemble Clustering)

Thay vì chọn một trong hai, kết hợp nhãn từ cả hai phương pháp:

```python
# Bước 1: Lấy nhãn từ P.pháp 1 (HAC k=4) và P.pháp 2 (HAC k=2)
labels_m1 = hac_method1.labels_   # [0,1,2,3, ...]  → phân tầng theo mức h
labels_m2 = hac_method2.labels_   # [0,1, ...]       → phân loại theo hành vi

# Bước 2: Tạo nhãn tổ hợp
combined = labels_m1 * 2 + labels_m2  # tối đa 4×2=8 tổ hợp
# → "Cluster 3 (mức cao) + Cluster 1 (ổn định)"
# → "Cluster 0 (mức thấp) + Cluster 0 (nhiễu)"

# Bước 3: Gộp các tổ hợp ít mẫu
from sklearn.cluster import AgglomerativeClustering
# Áp dụng lại clustering trên không gian kết hợp
```

**Lợi ích:** giữ được cả thông tin về *mức* và *hành vi* trong cùng nhãn.

---

### 6.2 Cải tiến tập đặc trưng

#### a) Thêm đặc trưng biến đổi theo thời gian (sliding window)

Thay vì dùng 1 giá trị trên toàn giờ, tính đặc trưng trên cửa sổ trượt:

```python
# Chia 1 giờ (3600s) thành 6 cửa sổ 10 phút (600s)
# Với mỗi cửa sổ tính mean, std, slope
# → thêm 6×3 = 18 đặc trưng về tính biến đổi nội giờ

windows = np.array_split(hour_data, 6)
for i, w in enumerate(windows):
    features[f'mean_w{i}'] = w.mean()
    features[f'std_w{i}']  = w.std()
    features[f'slope_w{i}'] = linregress(t_w, w).slope
```

#### b) Thêm đặc trưng tương quan giữa trục

Tích hợp thêm `X_Coord` và `Y_Coord`:

```python
# Tương quan Pearson giữa h và x trong cùng giờ
features['corr_hx'] = np.corrcoef(h_data, x_data)[0, 1]
features['corr_hy'] = np.corrcoef(h_data, y_data)[0, 1]

# Hướng dịch chuyển chủ đạo (PCA trên xyz)
pca_xyz = PCA(n_components=1).fit(xyz_matrix)
features['primary_direction'] = pca_xyz.explained_variance_ratio_[0]
```

#### c) Đặc trưng từ phân tích wavelet

Phân tích wavelet giúp tách được cả tần số lẫn thời điểm xảy ra:

```python
import pywt
coeffs = pywt.wavedec(hour_data, 'db4', level=5)
# Energy tại mỗi level → 5 đặc trưng mới
features['wavelet_energy_l{i}'] = np.sum(c**2) for i, c in enumerate(coeffs)
```

---

### 6.3 Cải tiến giảm chiều

#### Thay t-SNE bằng UMAP

UMAP khắc phục các nhược điểm của t-SNE:

| | t-SNE | UMAP |
|--|--|--|
| Bảo toàn cấu trúc toàn cục | Kém | Tốt hơn |
| Tái lập | Không | Có |
| Tốc độ | O(n²) | O(n log n) |
| Có thể dự đoán điểm mới | Không | **Có** |

```python
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
data_umap = reducer.fit_transform(data_scaled)
# Có thể dự đoán giờ mới mà không cần train lại
new_point = reducer.transform(new_hour_scaled)
```

---

### 6.4 Cải tiến lựa chọn k tối ưu

Bổ sung 2 tiêu chí hiện chưa có:

#### a) Gap Statistic

So sánh inertia thực tế với inertia của phân phối chuẩn:

```python
def gap_statistic(X, k_range, n_refs=10):
    gaps = []
    for k in k_range:
        # Inertia thực
        km = KMeans(n_clusters=k).fit(X)
        inertia_real = km.inertia_

        # Inertia tham chiếu (dữ liệu ngẫu nhiên)
        ref_inertias = []
        for _ in range(n_refs):
            X_rand = np.random.uniform(X.min(0), X.max(0), X.shape)
            ref_inertias.append(KMeans(n_clusters=k).fit(X_rand).inertia_)

        gap = np.log(np.mean(ref_inertias)) - np.log(inertia_real)
        gaps.append(gap)

    # k tối ưu: gap lớn nhất
    return k_range[np.argmax(gaps)]
```

#### b) BIC/AIC cho GMM (đã có, cần tích hợp vào voting)

```python
# Hiện tại voting chỉ dùng Silhouette, Calinski, Davies
# Thêm: k = argmin(BIC)
best_k_bic = k_values[np.argmin([gmm_results[k]['bic'] for k in k_values])]
```

---

### 6.5 Xác nhận kết quả (Validation)

Các bước kiểm định chưa được thực hiện:

#### a) Stability analysis – kiểm tra tính ổn định cụm

```python
from sklearn.utils import resample

stability_scores = []
for _ in range(100):
    # Bootstrap 80% dữ liệu
    X_boot = resample(X_scaled, n_samples=int(0.8 * len(X_scaled)))
    labels_boot = HAC(n_clusters=4).fit_predict(X_boot)

    # So sánh với kết quả đầy đủ bằng Adjusted Rand Index
    labels_full = HAC(n_clusters=4).fit_predict(X_scaled[:len(X_boot)])
    ari = adjusted_rand_score(labels_full, labels_boot)
    stability_scores.append(ari)

print(f"Stability (ARI): {np.mean(stability_scores):.3f} ± {np.std(stability_scores):.3f}")
# ARI > 0.8: cụm ổn định; < 0.5: cụm không đáng tin cậy
```

#### b) Phân tích temporal – kiểm tra tính liên tục theo thời gian

```python
# Nếu cụm có ý nghĩa vật lý, các giờ cùng cụm nên có xu hướng
# xuất hiện theo chuỗi thời gian liên tục, không ngẫu nhiên

# Kiểm tra bằng runs test
from statsmodels.stats.runs import runs_test
_, p_value = runs_test(labels_timeseries)
# p < 0.05: có cấu trúc thời gian → cụm có ý nghĩa
```

---

### 6.6 Tóm tắt lộ trình cải tiến

```
Hiện tại
   │
   ├─ Ngắn hạn (dễ thực hiện)
   │     ├─ Cài tslearn → dùng DTW thật sự
   │     ├─ Thêm UMAP thay t-SNE
   │     └─ Thêm Gap Statistic + BIC vào voting
   │
   ├─ Trung hạn
   │     ├─ Sliding window features (6 cửa sổ 10 phút)
   │     ├─ Wavelet features (pywt)
   │     ├─ Ensemble clustering (kết hợp P.pháp 1 + 2)
   │     └─ Stability analysis (bootstrap ARI)
   │
   └─ Dài hạn
         ├─ Tích hợp X_Coord, Y_Coord (clustering 3D)
         ├─ Temporal validation (runs test)
         └─ Autoencoder-based clustering (deep features)
```
