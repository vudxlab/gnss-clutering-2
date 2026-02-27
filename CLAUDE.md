# CLAUDE.md – Project Rules

## Project Overview
GNSS Clustering – Phan cum chuoi thoi gian dich chuyen GNSS theo tung doan gio.

## Environment
- Conda environment: `torch-cuda12.8`
- Run commands: `conda run -n torch-cuda12.8 python <script>`
- Python packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, tqdm

## Auto Commit Rules
- **Tu dong commit** sau moi lan sua doi code hoac fix loi ma user yeu cau, **neu test/chay thanh cong**.
- Quy trinh: sua code → chay test/verify → neu thanh cong → `git add` cac file da sua → `git commit` voi message mo ta thay doi.
- Neu test/chay that bai: **KHONG commit**, tiep tuc fix cho den khi thanh cong.
- Commit message bang tieng Viet hoac tieng Anh, ngan gon, mo ta ro thay doi.
- Khong commit file trong `data/` va `result/` (da co trong .gitignore).

## Project Structure
```
gnss_clustering/          # Package chinh
  config.py               # Cau hinh, hyperparameter
  data_loader.py          # Tai CSV, tao ma tran ngay/gio
  preprocessing.py        # Hampel, Kalman filter
  feature_extraction.py   # PP1: Scale → PCA → t-SNE
  feature_engineering.py  # PP2: 21 dac trung vat ly
  clustering.py           # HAC, GMM, DBSCAN, KMeans
  optimization.py         # Tim k toi uu (voting)
  stability.py            # Bootstrap stability + temporal coherence
  visualization.py        # Tat ca bieu do
main.py                   # Pipeline 1 buoc
step1_find_k.py           # Buoc 1: Tim k toi uu
step2_cluster.py          # Buoc 2: Phan cum + stability analysis
```

## Testing / Verification
- Chay test nhanh: `conda run -n torch-cuda12.8 python step2_cluster.py --k1 4 --k2 2 --no-display`
- Chi chay PP1: them `--method1-only`
- Chi chay PP2: them `--method2-only`
- Tai lai du lieu: them `--no-cache`
