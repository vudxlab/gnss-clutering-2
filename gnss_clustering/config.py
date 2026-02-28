"""
Cau hinh chung cho project GNSS Clustering
"""

import os

# ── Seed ────────────────────────────────────────────────────────────────────
SEED = 23

# ── Thu muc goc ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
RESULT_DIR  = os.path.join(BASE_DIR, 'result')

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Thu muc con cho tung phuong phap ──────────────────────────────────────
RESULT_SUBDIR_EDA       = '00_eda'
RESULT_SUBDIR_PREPROC   = '01_preprocessing'
RESULT_SUBDIR_PP1       = '02_pp1'
RESULT_SUBDIR_PP2       = '03_pp2'
RESULT_SUBDIR_PP2V2     = '04_pp2v2'
RESULT_SUBDIR_M3A       = '05_m3a'
RESULT_SUBDIR_M3B       = '06_m3b'
RESULT_SUBDIR_STABILITY = '07_stability'

# ── Du lieu dau vao ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(DATA_DIR, 'full_gnss_2e.csv')

# ── File trung gian (luu trong data/) ────────────────────────────────────────
MATRIX_FILE       = os.path.join(DATA_DIR, 'gnss_daily_matrix.npy')
DATES_FILE        = os.path.join(DATA_DIR, 'gnss_dates.npy')
HOURLY_MATRIX_FILE = os.path.join(DATA_DIR, 'gnss_hourly_matrix.npy')
HOURLY_INFO_FILE  = os.path.join(DATA_DIR, 'gnss_hourly_info.csv')

# ── Thong so ma tran ─────────────────────────────────────────────────────────
SECONDS_PER_DAY  = 86400
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY    = 24

# ── Nguong loc du lieu thieu (%) ─────────────────────────────────────────────
MISSING_THRESHOLD = 0

# ── Tien xu ly ───────────────────────────────────────────────────────────────
HAMPEL_WINDOW_SIZE    = 50
HAMPEL_N_SIGMAS       = 1
RESHAPE_WINDOW_SIZE   = 10   # cua so trung binh dong de giam chieu

KALMAN_PROCESS_VARIANCE     = 1e-5
KALMAN_MEASUREMENT_VARIANCE = 1e-1

BUTTERWORTH_CUTOFF = 0.1
BUTTERWORTH_ORDER  = 2

# ── PCA ──────────────────────────────────────────────────────────────────────
PCA_N_COMPONENTS = 50

# ── t-SNE ────────────────────────────────────────────────────────────────────
TSNE_N_COMPONENTS      = 2
TSNE_PERPLEXITY        = 30
TSNE_LEARNING_RATE     = 20
TSNE_EARLY_EXAGGERATION = 12
TSNE_METRIC            = 'l1'

# ── Phan cum ─────────────────────────────────────────────────────────────────
DEFAULT_N_CLUSTERS   = 2
K_RANGE              = (2, 11)

DBSCAN_MIN_SAMPLES   = 4
DBSCAN_EPS_PERCENTILE = 90

TS_KMEANS_METRIC     = 'dtw'
TS_KMEANS_MAX_ITER   = 20

GMM_MAX_ITER = 100

# ── Stability Analysis ─────────────────────────────────────────────────────
STABILITY_N_ITERATIONS  = 100   # So lan bootstrap
STABILITY_SAMPLE_RATIO  = 0.8   # Ty le mau moi lan (80%)

# ── Feature Engineering v2 ─────────────────────────────────────────────────
WAVELET_NAME          = 'db4'
WAVELET_MAX_LEVEL     = 4
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES   = 3
UMAP_N_NEIGHBORS      = 15
UMAP_MIN_DIST         = 0.1

# ── Matplotlib ───────────────────────────────────────────────────────────────
MATPLOTLIB_STYLE = 'seaborn-v0_8'
FIGURE_SIZE      = (12, 8)
FONT_SIZE        = 10
FIGURE_DPI       = 150      # DPI khi luu hinh
