"""
BUOC 2 – Phan cum chi tiet va xuat ket qua
============================================

Chay sau step1_find_k.py voi k da chon.

Chay:
    python step2_cluster.py --k1 4 --k2 2
    python step2_cluster.py --k1 4 --k2 2 --method1-only
    python step2_cluster.py --k1 4 --k2 2 --method2-only
    python step2_cluster.py --k1 4 --k2 2 --no-display

Neu khong truyen k, script dung gia tri mac dinh trong config.py.
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
parser = argparse.ArgumentParser(description='B2 – Phan cum chi tiet GNSS Clustering')
parser.add_argument('--k1',           type=int,  default=None,
                    help='So cum cho Phuong phap 1 (default: config.DEFAULT_N_CLUSTERS)')
parser.add_argument('--k2',           type=int,  default=None,
                    help='So cum cho Phuong phap 2 (default: config.DEFAULT_N_CLUSTERS)')
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
from gnss_clustering.clustering import run_all
from gnss_clustering.feature_engineering import run_feature_based_pipeline
from gnss_clustering import visualization as viz

plt.style.use(config.MATPLOTLIB_STYLE)
RD = config.RESULT_DIR


# --------------------------------------------------------------------------- #
#  Helper                                                                      #
# --------------------------------------------------------------------------- #
def _build_hourly_info_df(daily_matrix, unique_dates):
    rows = []
    for di, date in enumerate(unique_dates):
        for hour in range(config.HOURS_PER_DAY):
            s0 = hour * config.SECONDS_PER_HOUR
            hd = daily_matrix[di, s0:s0 + config.SECONDS_PER_HOUR]
            nan_c = np.isnan(hd).sum()
            rows.append({
                'date': date, 'hour': hour, 'day_idx': di,
                'missing_percentage': nan_c / config.SECONDS_PER_HOUR * 100,
                'valid_points': config.SECONDS_PER_HOUR - nan_c,
                'datetime': f"{date} {hour:02d}:00:00",
            })
    return pd.DataFrame(rows)


def _print_metrics_table(clustering_results, title="BANG SO SANH METRICS"):
    print(f"\n{title}:")
    hdr = f"{'Thuat toan':<22} {'k':>4} {'Silhouette':>12} {'Calinski':>12} {'Davies':>10}"
    print(hdr)
    print("-" * len(hdr))
    for method, res in clustering_results.items():
        n_k = res.get('n_clusters', '?')
        sil = res.get('silhouette', float('nan'))
        cal = res.get('calinski_harabasz', float('nan'))
        dav = res.get('davies_bouldin', float('nan'))
        print(f"{method:<22} {n_k:>4} {sil:>12.4f} {cal:>12.2f} {dav:>10.4f}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    k1 = args.k1 if args.k1 is not None else config.DEFAULT_N_CLUSTERS
    k2 = args.k2 if args.k2 is not None else config.DEFAULT_N_CLUSTERS

    print("=" * 70)
    print("BUOC 2 – PHAN CUM CHI TIET")
    if not args.method2_only:
        print(f"  Phuong phap 1 (Raw t-SNE):   k = {k1}")
    if not args.method1_only:
        print(f"  Phuong phap 2 (Feature):     k = {k2}")
    print(f"  Ket qua luu vao: {RD}")
    print("=" * 70)

    # ── 1. Tai / cache du lieu ──────────────────────────────────────────────
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

    # Ve bieu do tien xu ly
    viz.plot_z_comparison_batch(hourly_matrix, hampel_data, n=25, save=True, result_dir=RD)
    viz.plot_multiple_series(hourly_matrix[:25], n_cols=5, row_height=2, fig_width=20,
                             title='Ma tran theo gio – goc (25 hang dau)',
                             save=True, result_dir=RD, filename='11_raw_hourly_grid')
    viz.plot_multiple_series(hampel_data[:25], n_cols=5, row_height=2, fig_width=20,
                             title='Ma tran theo gio – sau Hampel (25 hang dau)',
                             save=True, result_dir=RD, filename='12_hampel_grid')
    viz.plot_multiple_series(data_filtered[:25], n_cols=5, row_height=2, fig_width=20,
                             title='Ma tran theo gio – sau Kalman filter (25 hang dau)',
                             save=True, result_dir=RD, filename='13_kalman_grid')

    results_summary = {}

    # ── 3. Phuong phap 1 – Raw t-SNE ────────────────────────────────────────
    if not args.method2_only:
        print(f"\n[3/4] PHUONG PHAP 1 – Phan cum voi k={k1} (Raw t-SNE)")
        print("-" * 50)
        print("  Trich xuat dac trung (scale → PCA → t-SNE)... (co the mat vai phut)")
        data_tsne, data_scaled, _ = extract_features(data_filtered)

        clustering_results_1 = run_all(data_tsne, data_scaled=data_scaled, n_clusters=k1)

        # Chuan hoa tuple -> dict
        for key in list(clustering_results_1.keys()):
            if isinstance(clustering_results_1[key], tuple):
                clustering_results_1[key] = clustering_results_1[key][0]

        _print_metrics_table(clustering_results_1, "PHUONG PHAP 1 – KET QUA METRICS")

        # Bieu do scatter + metric bars
        viz.plot_clustering_results(clustering_results_1, data_tsne,
                                    save=True, result_dir=RD)
        # Bieu do line plot tung cum
        viz.plot_clusters_lineplot_all_methods(clustering_results_1, data_scaled,
                                               save=True, result_dir=RD)

        results_summary['method1'] = {
            'k': k1,
            'clustering_results': clustering_results_1,
        }
        print(f"\n  Bieu do PP1 da luu:")
        print("    15_clustering_scatter.png")
        print("    16_clustering_metrics.png")
        print("    17_lineplot_*.png")
    else:
        print("\n[3/4] Bo qua Phuong phap 1 (--method2-only)")

    # ── 4. Phuong phap 2 – Feature-Based ────────────────────────────────────
    if not args.method1_only:
        print(f"\n[4/4] PHUONG PHAP 2 – Phan cum voi k={k2} (Feature-Based)")
        print("-" * 50)
        fb_results = run_feature_based_pipeline(
            hourly_matrix=hourly_matrix,
            hampel_data=hampel_data,
            valid_hours_info=valid_hours_info,
            n_clusters=k2,
            result_dir=RD,
        )
        _print_metrics_table(fb_results['clustering_results'],
                             "PHUONG PHAP 2 – KET QUA METRICS")

        results_summary['method2'] = {
            'k': k2,
            'clustering_results': fb_results['clustering_results'],
        }
        print(f"\n  Bieu do PP2 da luu:")
        print("    F01_feature_boxplot.png")
        print("    F02_pca_loadings.png")
        print("    F03_scatter_*.png")
        print("    F04_cluster_profiles_*.png")
        print("    F05_cluster_ts_*.png")
    else:
        print("\n[4/4] Bo qua Phuong phap 2 (--method1-only)")

    # ── Tom tat ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOM TAT KET QUA CUOI CUNG – BUOC 2")
    print("=" * 70)

    if 'method1' in results_summary:
        print(f"\n>> Phuong phap 1 (k={k1}) – HAC / GMM / KMeans / DBSCAN tren t-SNE:")
        for m, r in results_summary['method1']['clustering_results'].items():
            n_k = r.get('n_clusters', '?')
            sil = r.get('silhouette', float('nan'))
            cal = r.get('calinski_harabasz', float('nan'))
            dav = r.get('davies_bouldin', float('nan'))
            print(f"   {m:<22} k={n_k:>2}  Sil={sil:.4f}  Cal={cal:.1f}  Dav={dav:.4f}")

    if 'method2' in results_summary:
        print(f"\n>> Phuong phap 2 (k={k2}) – HAC / GMM / DBSCAN tren khong gian dac trung:")
        for m, r in results_summary['method2']['clustering_results'].items():
            n_k = r.get('n_clusters', '?')
            sil = r.get('silhouette', float('nan'))
            cal = r.get('calinski_harabasz', float('nan'))
            dav = r.get('davies_bouldin', float('nan'))
            print(f"   {m:<22} k={n_k:>2}  Sil={sil:.4f}  Cal={cal:.1f}  Dav={dav:.4f}")

    print(f"\nTat ca hinh anh da luu vao: {RD}")
    print("HOAN THANH!")

    return results_summary


if __name__ == '__main__':
    main()
