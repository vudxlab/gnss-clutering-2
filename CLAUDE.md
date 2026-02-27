# CLAUDE.md – Project Rules

## Project Overview
GNSS Clustering – Phan cum chuoi thoi gian dich chuyen GNSS theo tung doan gio.
- Du lieu: 191 doan 1 gio (3600 diem/gio, 1Hz) tu 17 tram GNSS
- 5 phuong phap phan cum: PP1 (Raw t-SNE), PP2 (Feature-Based), PP2v2 (Feature-Based V2), M3A (Conv1D Autoencoder), M3B (Moment Foundation Model)

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
  config.py                   # Cau hinh, hyperparameter
  data_loader.py              # Tai CSV, tao ma tran ngay/gio
  preprocessing.py            # Hampel, Kalman filter
  feature_extraction.py       # PP1: Scale → PCA → t-SNE
  feature_engineering.py      # PP2 + PP2v2: 22 dac trung co ban + 18 mo rong
  clustering.py               # HAC, GMM, DBSCAN, KMeans
  deep_clustering.py          # M3A: Conv1D Autoencoder, M3B: Moment Foundation Model
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
docs/
  DU_LIEU.md                  # Mo ta du lieu
  HUONG_DAN.md                # Huong dan su dung
  RESULTS.md                  # Ket qua phan cum
```

## Cac phuong phap phan cum

### PP1 – Raw t-SNE
- Scale → PCA 50D → t-SNE 2D → HAC, GMM, KMeans, DBSCAN

### PP2 – Feature-Based
- 22 dac trung vat ly (thong ke, xu huong, tan so, cau truc thoi gian, chat luong)
- StandardScaler → PCA 2D → HAC, GMM, DBSCAN

### PP2v2 – Feature-Based V2 (cai tien)
- 40 dac trung (22 co ban + wavelet, complexity, stationarity)
- PowerTransformer + RobustScaler
- Silhouette-guided feature weighting
- HAC, GMM, HDBSCAN, Ensemble (co-association matrix)
- UMAP 2D de visualize

### M3A – Conv1D Autoencoder
- Encoder: Conv1D 4 tang (1→16→32→64→128, stride=2) + FC → latent 32D
- Train MSE loss, Adam, 100 epochs
- Cluster tren latent space: HAC, GMM, HDBSCAN

### M3B – Moment Foundation Model
- AutonLab/MOMENT-1-large (zero-shot, 1024D embeddings)
- PCA reduction 1024D → 50D (vi n_samples < n_dim)
- Cluster tren embedding space: HAC, GMM, HDBSCAN

## Documentation Rules
- Sau moi lan thay doi code (them method, sua logic, fix bug), **cap nhat cac file docs da ton tai** (CLAUDE.md, README.md, ...) de phan anh dung trang thai hien tai.
- Cap nhat bao gom: project structure, ket qua metrics, huong dan su dung, mo ta phuong phap.
- **KHONG tu dong tao file doc moi** (*.md, README, ...) tru khi user yeu cau ro rang.

## Testing / Verification
- Chay tat ca: `conda run -n torch-cuda12.8 python step2_cluster.py --k1 4 --k2 2 --no-display`
- Chi chay PP1: them `--method1-only`
- Chi chay PP2 + PP2v2 + M3A + M3B: them `--method2-only`
- Tai lai du lieu: them `--no-cache`

## Pipeline step2_cluster.py
```
[1/4] Tai / cache du lieu
[2/4] Tien xu ly (Hampel → reshape → Kalman)
[3/4] PP1 – Raw t-SNE (--method1-only de chi chay PP1)
[4/6] PP2 – Feature-Based
[5/6] PP2v2 – Feature-Based V2
[6/7] M3A – Conv1D Autoencoder
[7/8] M3B – Moment Foundation Model
[8/8] Stability Analysis (Bootstrap ARI + Temporal coherence)
```
