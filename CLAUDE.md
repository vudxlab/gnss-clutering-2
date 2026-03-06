# CLAUDE.md – Project Rules

## Project Overview
GNSS Clustering – Phan cum chuoi thoi gian dich chuyen GNSS theo tung doan gio.
- Du lieu: 191 doan 1 gio (3600 diem/gio, 1Hz) tu 17 tram GNSS
- 5 phuong phap phan cum: PP1 (Raw t-SNE), PP2 (Feature-Based), PP2v2 (Feature-Based V2), M3A (Conv1D Autoencoder), M3B (Moment Foundation Model)
- Ho tro multi-axis: phan tich dong thoi X_Coord, Y_Coord, h_Coord (--axes xyh)

## Environment
- Conda environment: `torch-cuda12.8`
- Run commands: `conda run -n torch-cuda12.8 python <script>`
- Python packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, tqdm, torch, pywt, hdbscan, umap-learn, momentfm

## Auto Commit Rules
- **Tu dong commit** sau moi lan sua doi code hoac fix loi ma user yeu cau, **neu test/chay thanh cong**.
- Quy trinh: sua code → chay test/verify → neu thanh cong → `git add` cac file da sua → `git commit` voi message mo ta thay doi.
- Neu test/chay that bai: **KHONG commit**, tiep tuc fix cho den khi thanh cong.
- Commit message bang tieng Viet hoac tieng Anh, ngan gon, mo ta ro thay doi.
- Khong commit file trong `data/` va `result/` (da co trong .gitignore).

## Project Structure
```
gnss_clustering/              # Package chinh
  config.py                   # Cau hinh, hyperparameter, multi-axis constants
  data_loader.py              # Tai CSV, tao ma tran ngay/gio (single + multi-axis)
  preprocessing.py            # Hampel, Kalman filter
  feature_extraction.py       # PP1: Scale → PCA → t-SNE
  feature_engineering.py      # PP2 + PP2v2: dac trung per-axis + cross-axis features
  clustering.py               # HAC, GMM, DBSCAN, KMeans
  deep_clustering.py          # M3A: Conv1D Autoencoder (multi-channel), M3B: Moment (per-axis concat)
  optimization.py             # Tim k toi uu (voting)
  stability.py                # Bootstrap stability + temporal coherence
  visualization.py            # Tat ca bieu do
  __init__.py                 # Exports
main.py                       # Pipeline 1 buoc
step1_find_k.py               # Buoc 1: Tim k toi uu
step2_cluster.py              # Buoc 2: Phan cum + stability analysis (tat ca 5 PP)
notebook/
  Clustering_GNSS_3e.ipynb    # Notebook phan tich
  station_layout.ipynb        # Vi tri tram GNSS (3D + 2D)
result/                       # Output hinh anh (theo subfolder)
  00_eda/                     # Bieu do EDA (01_ - 09_)
  01_preprocessing/           # Bieu do tien xu ly (10_ - 13_)
  02_pp1/                     # PP1: t-SNE clustering (15_ - 17_)
  03_pp2/                     # PP2: Feature-Based (F01_ - F05_)
  04_pp2v2/                   # PP2v2: Feature-Based V2 (F2_*, F*_v2)
  05_m3a/                     # M3A: Conv1D Autoencoder (M3_01 - M3_05)
  06_m3b/                     # M3B: Moment Foundation Model (M3_*_moment_*)
  07_stability/               # Stability Analysis (S01_, S02_)
docs/
  DU_LIEU.md                  # Mo ta du lieu
  HUONG_DAN.md                # Huong dan su dung
  RESULTS.md                  # Ket qua phan cum
```

## Cac phuong phap phan cum

### PP1 – Raw t-SNE
- Single-axis: Scale → PCA 50D → t-SNE 2D → HAC, GMM, KMeans, DBSCAN
- Multi-axis: Concat data_filtered cac truc → (191, 360×N) → PCA 50D → t-SNE 2D

### PP2 – Feature-Based
- Single-axis: 22 dac trung vat ly (thong ke, xu huong, tan so, cau truc thoi gian, chat luong)
- Multi-axis: 22 features/truc × N + 8 cross-axis features (correlation, magnitude, vertical_ratio)
- StandardScaler → PCA 2D → HAC, GMM, DBSCAN

### PP2v2 – Feature-Based V2 (cai tien)
- Single-axis: 40 dac trung (22 co ban + wavelet, complexity, stationarity)
- Multi-axis: 40 features/truc × N + 8 cross-axis features
- PowerTransformer + RobustScaler
- Silhouette-guided feature weighting
- HAC, GMM, HDBSCAN, Ensemble (co-association matrix)
- UMAP 2D de visualize

### M3A – Conv1D Autoencoder
- Encoder: Conv1D 4 tang (in_channels→16→32→64→128, stride=2) + FC → latent 32D
- Single-axis: in_channels=1, input (n, 1, 360)
- Multi-axis: in_channels=N, input (n, N, 360) – Conv1D tu hoc cross-channel patterns
- Train MSE loss, Adam, 100 epochs
- Cluster tren latent space: HAC, GMM, HDBSCAN

### M3B – Moment Foundation Model
- AutonLab/MOMENT-1-large (zero-shot, 1024D embeddings)
- Single-axis: 1024D → PCA 50D
- Multi-axis: extract embedding per-axis, concat → (1024×N)D → PCA 50D
- Cluster tren embedding space: HAC, GMM, HDBSCAN

## Multi-axis Support
- Truc kha dung: x (X_Coord), y (Y_Coord), h (h_Coord)
- Mac dinh: `--axes h` (chi h_Coord, backward compatible)
- Multi-axis: `--axes xyh`, `--axes xy`, `--axes xh`, `--axes yh`
- Moi truc tao ma tran rieng (191, 3600), tien xu ly doc lap
- Cross-axis features: corr_xy, corr_xh, corr_yh, magnitude_3d, horiz_magnitude, vertical_ratio
- Cache rieng theo truc: `gnss_hourly_matrix_x.npy`, `gnss_hourly_matrix_y.npy`, etc.

## Documentation Rules
- Sau moi lan thay doi code (them method, sua logic, fix bug), **cap nhat cac file docs da ton tai** (CLAUDE.md, README.md, ...) de phan anh dung trang thai hien tai.
- Cap nhat bao gom: project structure, ket qua metrics, huong dan su dung, mo ta phuong phap.
- **KHONG tu dong tao file doc moi** (*.md, README, ...) tru khi user yeu cau ro rang.

## Testing / Verification
- Chay tat ca (single-axis): `conda run -n torch-cuda12.8 python step2_cluster.py --k1 4 --k2 2 --k3 2 --no-display`
- Chay tat ca (multi-axis):  `conda run -n torch-cuda12.8 python step2_cluster.py --k1 4 --k2 2 --k3 2 --axes xyh --no-display`
- Chi chay PP1: them `--method1-only`
- Chi chay PP2 + PP2v2: them `--method2-only`
- Chi chay M3A + M3B: them `--method3-only`
- Tai lai du lieu: them `--no-cache`

## Pipeline step2_cluster.py
3 nhom phuong phap doc lap:
- `--k1`: so cum cho PP1 (Raw t-SNE)
- `--k2`: so cum cho PP2, PP2v2 (Feature-Based)
- `--k3`: so cum cho M3A, M3B (Deep Learning)
- `--axes`: truc su dung (mac dinh `h`, co the la `xyh`, `xy`, `xh`, `yh`)

```
[1/4] Tai / cache du lieu
[2/4] Tien xu ly (Hampel → reshape → Kalman)
[3/8] PP1 – Raw t-SNE                      ← dung k1, --method1-only
[4/8] PP2 – Feature-Based                  ← dung k2, --method2-only
[5/8] PP2v2 – Feature-Based V2             ← dung k2, --method2-only
[6/8] M3A – Conv1D Autoencoder             ← dung k3, --method3-only
[7/8] M3B – Moment Foundation Model        ← dung k3, --method3-only
[8/8] Stability Analysis (Bootstrap ARI + Temporal coherence)
```
