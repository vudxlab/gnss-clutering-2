"""
BUOC 1 – Tim so cum toi uu (k-search)
======================================

Chay script nay truoc de xac dinh so cum k toi uu cho tung phuong phap.
Ket qua la cac bieu do va bang so sanh, giup ban dua ra quyet dinh k.

Sau khi co k, chay tiep step2_cluster.py:
    python step2_cluster.py --k1 4 --k2 2

Chay:
    python step1_find_k.py                      # dung k_range mac dinh (2..10)
    python step1_find_k.py --k-min 2 --k-max 8  # tuy chinh khoang k
    python step1_find_k.py --method1-only        # chi chay Phuong phap 1
    python step1_find_k.py --method2-only        # chi chay Phuong phap 2
    python step1_find_k.py --no-display          # khong mo cua so (server/headless)
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Parse arguments                                                             #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='B1 – Tim so cum toi uu GNSS Clustering')
parser.add_argument('--k-min',        type=int,  default=None, help='k nho nhat (mac dinh: config.K_RANGE[0])')
parser.add_argument('--k-max',        type=int,  default=None, help='k lon nhat +1 (mac dinh: config.K_RANGE[1])')
parser.add_argument('--method1-only', action='store_true', help='Chi chay Phuong phap 1')
parser.add_argument('--method2-only', action='store_true', help='Chi chay Phuong phap 2')
parser.add_argument('--no-display',   action='store_true', help='Khong hien thi cua so (Agg backend)')
parser.add_argument('--no-cache',     action='store_true', help='Bo qua cache, tai lai du lieu tu CSV')
args = parser.parse_args()

# --------------------------------------------------------------------------- #
#  Matplotlib backend                                                          #
# --------------------------------------------------------------------------- #
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#  Import package                                                              #
# --------------------------------------------------------------------------- #
from gnss_clustering import config
from gnss_clustering.data_loader import (
    load_data, create_daily_matrix, create_hourly_matrix, load_cached_matrices,
)
from gnss_clustering.preprocessing import preprocess_pipeline
from gnss_clustering.feature_extraction import extract_features
from gnss_clustering.optimization import find_optimal_clusters
from gnss_clustering.feature_engineering import find_optimal_clusters_features
from gnss_clustering import visualization as viz

plt.style.use(config.MATPLOTLIB_STYLE)
RD = config.RESULT_DIR


# --------------------------------------------------------------------------- #
#  Helper                                                                      #
# --------------------------------------------------------------------------- #
def _build_hourly_info_df(daily_matrix, unique_dates):
    """Xay dung hourly_info_df day du (truoc khi loc) tu daily_matrix."""
    rows = []
    for di, date in enumerate(unique_dates):
        for hour in range(config.HOURS_PER_DAY):
            s0 = hour * config.SECONDS_PER_HOUR
            s1 = s0 + config.SECONDS_PER_HOUR
            hd = daily_matrix[di, s0:s1]
            nan_c = np.isnan(hd).sum()
            rows.append({
                'date': date, 'hour': hour, 'day_idx': di,
                'missing_percentage': nan_c / config.SECONDS_PER_HOUR * 100,
                'valid_points': config.SECONDS_PER_HOUR - nan_c,
                'datetime': f"{date} {hour:02d}:00:00",
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("BUOC 1 – TIM SO CUM TOI UU")
    print(f"  Ket qua luu vao: {RD}")
    print("=" * 70)

    # ── k_range ────────────────────────────────────────────────────────────
    k_min = args.k_min if args.k_min is not None else config.K_RANGE[0]
    k_max = args.k_max if args.k_max is not None else config.K_RANGE[1]
    k_range = (k_min, k_max)
    print(f"\nKhoang k: {k_range[0]} den {k_range[1]-1}")

    # ── 1. Tai du lieu ──────────────────────────────────────────────────────
    print("\n[1/4] Tai / kiem tra cache du lieu...")
    cache_ok = (
        not args.no_cache
        and os.path.exists(config.HOURLY_MATRIX_FILE)
        and os.path.exists(config.HOURLY_INFO_FILE)
        and os.path.exists(config.MATRIX_FILE)
        and os.path.exists(config.DATES_FILE)
    )

    if cache_ok:
        print("  Cache ton tai – dang tai...")
        daily_matrix, unique_dates, hourly_matrix, valid_hours_info = load_cached_matrices()
        print(f"  hourly_matrix: {hourly_matrix.shape}, "
              f"valid hours: {len(valid_hours_info)}")
    else:
        print("  Khong co cache – tai tu CSV...")
        df = load_data()
        daily_matrix, unique_dates = create_daily_matrix(df, save=True)
        hourly_matrix, valid_hours_info = create_hourly_matrix(
            daily_matrix, unique_dates, save=True
        )
        print(f"  hourly_matrix: {hourly_matrix.shape}, "
              f"valid hours: {len(valid_hours_info)}")

        # Ve bieu do EDA (chi khi tai moi)
        print("\n  Ve bieu do EDA...")
        hourly_info_df = _build_hourly_info_df(daily_matrix, unique_dates)
        viz.plot_daily_heatmap(daily_matrix, unique_dates, save=True, result_dir=RD)
        viz.plot_daily_timeseries(daily_matrix, unique_dates, save=True, result_dir=RD)
        viz.plot_hourly_heatmap(hourly_matrix, valid_hours_info,
                                hourly_info_df=hourly_info_df, save=True, result_dir=RD)
        viz.plot_hourly_overview(hourly_matrix, valid_hours_info, save=True, result_dir=RD)
        viz.plot_hourly_analysis(hourly_matrix, valid_hours_info,
                                 hourly_info_df=hourly_info_df, save=True, result_dir=RD)
        viz.plot_sample_hours(hourly_matrix, valid_hours_info, save=True, result_dir=RD)
        viz.plot_first_n_hours(hourly_matrix, valid_hours_info, n=20, save=True, result_dir=RD)

    # ── 2. Tien xu ly ───────────────────────────────────────────────────────
    print("\n[2/4] Tien xu ly (Hampel → reshape → Kalman)...")
    data_filtered, hampel_data, _ = preprocess_pipeline(hourly_matrix)

    # ── 3. Phuong phap 1 – Raw t-SNE ────────────────────────────────────────
    if not args.method2_only:
        print("\n[3/4] PHUONG PHAP 1 – Tim k tren khong gian t-SNE")
        print("-" * 50)
        print("  Trich xuat dac trung (scale → PCA → t-SNE)... (co the mat vai phut)")
        data_tsne, data_scaled, _ = extract_features(data_filtered)

        opt1 = find_optimal_clusters(
            data=data_tsne,
            data_scaled=data_scaled,
            k_range=k_range,
            methods=['timeseries_kmeans', 'hac', 'dbscan', 'gmm'],
            save=True, result_dir=RD,
        )
        k1 = opt1['recommended_k'] or config.DEFAULT_N_CLUSTERS
        print(f"\n  [PP1] So cum de xuat: k = {k1}")
    else:
        print("\n[3/4] Bo qua Phuong phap 1 (--method2-only)")
        data_tsne = data_scaled = None
        k1 = config.DEFAULT_N_CLUSTERS
        opt1 = None

    # ── 4. Phuong phap 2 – Feature-Based ────────────────────────────────────
    if not args.method1_only:
        print("\n[4/4] PHUONG PHAP 2 – Tim k tren khong gian dac trung")
        print("-" * 50)
        opt2 = find_optimal_clusters_features(
            hourly_matrix=hourly_matrix,
            hampel_data=hampel_data,
            k_range=k_range,
            save=True, result_dir=RD,
        )
        k2 = opt2['recommended_k']
        print(f"\n  [PP2] So cum de xuat: k = {k2}")
    else:
        print("\n[4/4] Bo qua Phuong phap 2 (--method1-only)")
        k2 = config.DEFAULT_N_CLUSTERS
        opt2 = None

    # ── Tom tat ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOM TAT KET QUA – BUOC 1")
    print("=" * 70)
    print(f"{'Phuong phap':<30} {'k de xuat':>12} {'So phieu':>10}")
    print("-" * 55)
    if opt1:
        print(f"{'PP1 – Raw t-SNE (HAC/GMM/KMeans)':<30} {k1:>12} "
              f"{opt1['vote_count']:>10}")
    if opt2:
        print(f"{'PP2 – Feature-Based (HAC/GMM)':<30} {k2:>12} "
              f"{opt2['vote_count']:>10}")
    print("-" * 55)
    print(f"\nTiep theo, chay phan cum chi tiet (Buoc 2):")
    print(f"  python step2_cluster.py --k1 {k1} --k2 {k2}")
    print(f"\nHinh anh da luu vao: {RD}")
    print("  14_optimal_k_analysis.png  (Phuong phap 1)")
    print("  F00_optimal_k_features.png (Phuong phap 2)")

    return {'k1': k1, 'k2': k2, 'opt1': opt1, 'opt2': opt2}


if __name__ == '__main__':
    main()
