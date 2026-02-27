"""
BUOC 2 – Phan cum chi tiet va xuat ket qua
============================================

Chay sau step1_find_k.py voi k da chon.

3 nhom phuong phap:
  Method 1: Raw t-SNE clustering (--k1)
  Method 2: Feature-Based clustering – PP2, PP2v2 (--k2)
  Method 3: Deep Learning clustering – M3A Conv1D AE, M3B Moment (--k3)

Chay:
    python step2_cluster.py --k1 4 --k2 2 --k3 2
    python step2_cluster.py --k1 4 --method1-only
    python step2_cluster.py --k2 2 --method2-only
    python step2_cluster.py --k3 2 --method3-only
    python step2_cluster.py --k1 4 --k2 2 --k3 2 --no-display

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
parser.add_argument('--k3',           type=int,  default=None,
                    help='So cum cho Phuong phap 3 (default: config.DEFAULT_N_CLUSTERS)')
parser.add_argument('--method1-only', action='store_true', help='Chi chay Phuong phap 1')
parser.add_argument('--method2-only', action='store_true', help='Chi chay Phuong phap 2 (PP2 + PP2v2)')
parser.add_argument('--method3-only', action='store_true', help='Chi chay Phuong phap 3 (M3A + M3B)')
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
from gnss_clustering.feature_engineering import run_feature_based_pipeline, run_feature_based_pipeline_v2
from gnss_clustering.deep_clustering import run_autoencoder_pipeline, run_moment_pipeline
from gnss_clustering.stability import run_stability_analysis
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
    k3 = args.k3 if args.k3 is not None else config.DEFAULT_N_CLUSTERS

    # Xac dinh phuong phap nao se chay
    only = args.method1_only or args.method2_only or args.method3_only
    run_m1 = args.method1_only or not only
    run_m2 = args.method2_only or not only
    run_m3 = args.method3_only or not only

    print("=" * 70)
    print("BUOC 2 – PHAN CUM CHI TIET")
    if run_m1:
        print(f"  Phuong phap 1 (Raw t-SNE):       k = {k1}")
    if run_m2:
        print(f"  Phuong phap 2 (Feature-Based):    k = {k2}")
    if run_m3:
        print(f"  Phuong phap 3 (Deep Learning):    k = {k3}")
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
    if run_m1:
        print(f"\n[3/8] PHUONG PHAP 1 – Phan cum voi k={k1} (Raw t-SNE)")
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
        print("\n[3/8] Bo qua Phuong phap 1")

    # ── 4. Phuong phap 2 – Feature-Based (original) ─────────────────────────
    if run_m2:
        print(f"\n[4/8] PHUONG PHAP 2 – Phan cum voi k={k2} (Feature-Based)")
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
        print("\n[4/8] Bo qua Phuong phap 2")

    # ── 5. Phuong phap 2v2 – Feature-Based V2 (cai tien) ──────────────────
    if run_m2:
        print(f"\n[5/8] PHUONG PHAP 2v2 – Phan cum voi k={k2} (Feature-Based V2)")
        print("-" * 50)
        fb_v2_results = run_feature_based_pipeline_v2(
            hourly_matrix=hourly_matrix,
            hampel_data=hampel_data,
            valid_hours_info=valid_hours_info,
            n_clusters=k2,
            result_dir=RD,
        )
        _print_metrics_table(fb_v2_results['clustering_results'],
                             "PHUONG PHAP 2v2 – KET QUA METRICS")

        results_summary['method2v2'] = {
            'k': k2,
            'clustering_results': fb_v2_results['clustering_results'],
        }
        print(f"\n  Bieu do PP2v2 da luu:")
        print("    F2_01_feature_weights.png")
        print("    F2_03_scatter_*.png")
        print("    F2_06_co_association.png")
    else:
        print("\n[5/8] Bo qua Phuong phap 2v2")

    # ── 6. Method 3A – Conv1D Autoencoder ──────────────────────────────────
    if run_m3:
        print(f"\n[6/8] METHOD 3A – Conv1D Autoencoder (k={k3})")
        print("-" * 50)
        ae_results = run_autoencoder_pipeline(
            hourly_matrix=hourly_matrix,
            hampel_data=hampel_data,
            valid_hours_info=valid_hours_info,
            n_clusters=k3,
            latent_dim=32,
            epochs=100,
            result_dir=RD,
        )
        _print_metrics_table(ae_results['clustering_results'],
                             "METHOD 3A – KET QUA METRICS")

        results_summary['method3a'] = {
            'k': k3,
            'clustering_results': ae_results['clustering_results'],
        }
        print(f"\n  Bieu do M3A da luu:")
        print("    M3_01_training_loss.png")
        print("    M3_02_reconstruction.png")
        print("    M3_03_latent_scatter_*.png")
        print("    M3_05_cluster_ts_*.png")
    else:
        print("\n[6/8] Bo qua Method 3A")

    # ── 7. Method 3B – Moment Foundation Model ────────────────────────────
    if run_m3:
        print(f"\n[7/8] METHOD 3B – Moment Foundation Model (k={k3})")
        print("-" * 50)
        moment_results = run_moment_pipeline(
            hourly_matrix=hourly_matrix,
            hampel_data=hampel_data,
            valid_hours_info=valid_hours_info,
            n_clusters=k3,
            result_dir=RD,
        )
        _print_metrics_table(moment_results['clustering_results'],
                             "METHOD 3B – KET QUA METRICS")

        results_summary['method3b'] = {
            'k': k3,
            'clustering_results': moment_results['clustering_results'],
        }
        print(f"\n  Bieu do M3B da luu:")
        print("    M3_03_latent_scatter_moment_*.png")
        print("    M3_05_cluster_ts_moment_*.png")
    else:
        print("\n[7/8] Bo qua Method 3B")

    # ── 8. Stability Analysis ──────────────────────────────────────────────
    print("\n[8/8] STABILITY ANALYSIS")
    print("-" * 50)

    # PP1: stability tren khong gian t-SNE (HAC + GMM)
    if 'method1' in results_summary:
        cr1 = results_summary['method1']['clustering_results']
        labels_dict_1 = {}
        n_clusters_dict_1 = {}
        for m in ['HAC', 'GMM']:
            if m in cr1 and cr1[m] is not None:
                labels_dict_1[f'PP1_{m}'] = cr1[m]['labels']
                n_clusters_dict_1[f'PP1_{m}'] = cr1[m]['n_clusters']

        if labels_dict_1:
            stab1 = run_stability_analysis(
                X=data_tsne,
                labels_dict=labels_dict_1,
                n_clusters_dict=n_clusters_dict_1,
                n_iterations=config.STABILITY_N_ITERATIONS,
                sample_ratio=config.STABILITY_SAMPLE_RATIO,
                save=True, result_dir=RD,
            )
            results_summary['method1']['stability'] = stab1

    # PP2: stability tren khong gian dac trung (HAC + GMM)
    if 'method2' in results_summary:
        cr2 = results_summary['method2']['clustering_results']
        X_scaled_2 = fb_results['X_scaled']
        labels_dict_2 = {}
        n_clusters_dict_2 = {}
        for m in ['HAC', 'GMM']:
            if m in cr2 and cr2[m] is not None:
                labels_dict_2[f'PP2_{m}'] = cr2[m]['labels']
                n_clusters_dict_2[f'PP2_{m}'] = cr2[m]['n_clusters']

        if labels_dict_2:
            stab2 = run_stability_analysis(
                X=X_scaled_2,
                labels_dict=labels_dict_2,
                n_clusters_dict=n_clusters_dict_2,
                n_iterations=config.STABILITY_N_ITERATIONS,
                sample_ratio=config.STABILITY_SAMPLE_RATIO,
                save=True, result_dir=RD,
            )
            results_summary['method2']['stability'] = stab2

    # PP2v2: stability tren khong gian dac trung v2 (HAC + GMM + Ensemble)
    if 'method2v2' in results_summary:
        cr2v2 = results_summary['method2v2']['clustering_results']
        X_weighted_2v2 = fb_v2_results['X_weighted']
        labels_dict_2v2 = {}
        n_clusters_dict_2v2 = {}
        for m in ['HAC', 'GMM', 'Ensemble']:
            if m in cr2v2 and cr2v2[m] is not None:
                labels_dict_2v2[f'PP2v2_{m}'] = cr2v2[m]['labels']
                n_clusters_dict_2v2[f'PP2v2_{m}'] = cr2v2[m]['n_clusters']

        if labels_dict_2v2:
            stab2v2 = run_stability_analysis(
                X=X_weighted_2v2,
                labels_dict=labels_dict_2v2,
                n_clusters_dict=n_clusters_dict_2v2,
                n_iterations=config.STABILITY_N_ITERATIONS,
                sample_ratio=config.STABILITY_SAMPLE_RATIO,
                save=True, result_dir=RD,
            )
            results_summary['method2v2']['stability'] = stab2v2

    # M3A: stability tren latent space (HAC + GMM)
    if 'method3a' in results_summary:
        cr3a = results_summary['method3a']['clustering_results']
        Z_3a = ae_results['Z_scaled']
        labels_dict_3a = {}
        n_clusters_dict_3a = {}
        for m in ['HAC', 'GMM']:
            if m in cr3a and cr3a[m] is not None:
                labels_dict_3a[f'M3A_{m}'] = cr3a[m]['labels']
                n_clusters_dict_3a[f'M3A_{m}'] = cr3a[m]['n_clusters']

        if labels_dict_3a:
            stab3a = run_stability_analysis(
                X=Z_3a,
                labels_dict=labels_dict_3a,
                n_clusters_dict=n_clusters_dict_3a,
                n_iterations=config.STABILITY_N_ITERATIONS,
                sample_ratio=config.STABILITY_SAMPLE_RATIO,
                save=True, result_dir=RD,
            )
            results_summary['method3a']['stability'] = stab3a

    # M3B: stability tren embedding space (HAC + GMM)
    if 'method3b' in results_summary:
        cr3b = results_summary['method3b']['clustering_results']
        Z_3b = moment_results['Z_scaled']
        labels_dict_3b = {}
        n_clusters_dict_3b = {}
        for m in ['HAC', 'GMM']:
            if m in cr3b and cr3b[m] is not None:
                labels_dict_3b[f'M3B_{m}'] = cr3b[m]['labels']
                n_clusters_dict_3b[f'M3B_{m}'] = cr3b[m]['n_clusters']

        if labels_dict_3b:
            stab3b = run_stability_analysis(
                X=Z_3b,
                labels_dict=labels_dict_3b,
                n_clusters_dict=n_clusters_dict_3b,
                n_iterations=config.STABILITY_N_ITERATIONS,
                sample_ratio=config.STABILITY_SAMPLE_RATIO,
                save=True, result_dir=RD,
            )
            results_summary['method3b']['stability'] = stab3b

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

    if 'method2v2' in results_summary:
        print(f"\n>> Phuong phap 2v2 (k={k2}) – HAC / GMM / HDBSCAN / Ensemble (cai tien):")
        for m, r in results_summary['method2v2']['clustering_results'].items():
            n_k = r.get('n_clusters', '?')
            sil = r.get('silhouette', float('nan'))
            cal = r.get('calinski_harabasz', float('nan'))
            dav = r.get('davies_bouldin', float('nan'))
            print(f"   {m:<22} k={n_k:>2}  Sil={sil:.4f}  Cal={cal:.1f}  Dav={dav:.4f}")

    if 'method3a' in results_summary:
        print(f"\n>> Method 3A (k={k3}) – Conv1D Autoencoder + HAC / GMM / HDBSCAN:")
        for m, r in results_summary['method3a']['clustering_results'].items():
            n_k = r.get('n_clusters', '?')
            sil = r.get('silhouette', float('nan'))
            cal = r.get('calinski_harabasz', float('nan'))
            dav = r.get('davies_bouldin', float('nan'))
            print(f"   {m:<22} k={n_k:>2}  Sil={sil:.4f}  Cal={cal:.1f}  Dav={dav:.4f}")

    if 'method3b' in results_summary:
        print(f"\n>> Method 3B (k={k3}) – Moment Foundation Model + HAC / GMM / HDBSCAN:")
        for m, r in results_summary['method3b']['clustering_results'].items():
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
