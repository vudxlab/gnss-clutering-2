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

# ── Matplotlib ───────────────────────────────────────────────────────────────
MATPLOTLIB_STYLE = 'seaborn-v0_8'
FIGURE_SIZE      = (12, 8)
FONT_SIZE        = 10
FIGURE_DPI       = 150      # DPI khi luu hinh
