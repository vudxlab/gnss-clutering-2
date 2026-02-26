"""
Pipeline chinh cho GNSS Clustering Project.

Chay:
    python main.py

Tuy chinh:
    - Sua cac thong so trong gnss_clustering/config.py
    - Hoac truyen doi so vao tung ham
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')   # doi thanh 'Agg' neu khong co display (server/CI)
import matplotlib.pyplot as plt

from gnss_clustering import config
from gnss_clustering.data_loader import (
    load_data,
    create_daily_matrix,
    create_hourly_matrix,
)
from gnss_clustering.preprocessing import preprocess_pipeline
from gnss_clustering.feature_extraction import extract_features
from gnss_clustering.clustering import run_all
from gnss_clustering.optimization import find_optimal_clusters
from gnss_clustering import visualization as viz

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.style.use(config.MATPLOTLIB_STYLE)
plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
plt.rcParams['font.size'] = config.FONT_SIZE

RD = config.RESULT_DIR   # thu muc luu hinh


def main():
    print("=" * 70)
    print("GNSS CLUSTERING PIPELINE")
    print(f"  Du lieu  : {config.DATA_DIR}")
    print(f"  Ket qua  : {config.RESULT_DIR}")
    print("=" * 70)

    # ── 1. Tai du lieu ───────────────────────────────────────────────────────
    print("\n[1/6] Tai du lieu...")
    df = load_data()

    # ── 2. Ma tran ngay / gio ────────────────────────────────────────────────
    print("\n[2/6] Tao ma tran ngay va gio...")
    daily_matrix, unique_dates = create_daily_matrix(df, save=True)
    hourly_matrix, valid_hours_info = create_hourly_matrix(
        daily_matrix, unique_dates, save=True
    )

    # Can hourly_info_df (toan bo, truoc loc) de ve bieu do so sanh
    import pandas as pd, numpy as np
    from gnss_clustering import config as cfg
    seconds_per_hour = cfg.SECONDS_PER_HOUR
    hours_per_day    = cfg.HOURS_PER_DAY
    hourly_info_list = []
    for di, date in enumerate(unique_dates):
        for hour in range(hours_per_day):
            s0 = hour * seconds_per_hour
            s1 = (hour + 1) * seconds_per_hour
            hd = daily_matrix[di, s0:s1]
            nan_c = np.isnan(hd).sum()
            hourly_info_list.append({
                'date': date, 'hour': hour, 'day_idx': di,
                'missing_percentage': nan_c / seconds_per_hour * 100,
                'valid_points': seconds_per_hour - nan_c,
                'datetime': f"{date} {hour:02d}:00:00"
            })
    hourly_info_df = pd.DataFrame(hourly_info_list)

    # --- Ve bieu do du lieu thu nghiem ---
    print("\n  Dang ve bieu do du lieu...")

    # Cell 7 – Heatmap ngay
    viz.plot_daily_heatmap(daily_matrix, unique_dates, save=True, result_dir=RD)

    # Cell 8 – Timeseries moi ngay
    viz.plot_daily_timeseries(daily_matrix, unique_dates, save=True, result_dir=RD)

    # Cell 12 – Heatmap gio + histogram ty le thieu + bar so gio hop le
    viz.plot_hourly_heatmap(
        hourly_matrix, valid_hours_info,
        hourly_info_df=hourly_info_df,
        save=True, result_dir=RD
    )

    # Cell 14 – 4-subplot tong quan (3D, histogram, boxplot, line mau)
    viz.plot_hourly_overview(hourly_matrix, valid_hours_info, save=True, result_dir=RD)

    # Cell 15 – 2x2: heatmap chi tiet, so sanh truoc/sau, violin, weekday
    viz.plot_hourly_analysis(
        hourly_matrix, valid_hours_info,
        hourly_info_df=hourly_info_df,
        save=True, result_dir=RD
    )

    # Cell 16 – 6 gio mau tot nhat
    viz.plot_sample_hours(hourly_matrix, valid_hours_info, save=True, result_dir=RD)

    # Cell 17 – Heatmap 20 gio dau + luoi subplot
    viz.plot_first_n_hours(hourly_matrix, valid_hours_info, n=20, save=True, result_dir=RD)

    # ── 3. Tien xu ly ────────────────────────────────────────────────────────
    print("\n[3/6] Tien xu ly du lieu...")
    data_filtered, hampel_data, reshape_data = preprocess_pipeline(hourly_matrix)

    # Cell 25 – So sanh truoc/sau Hampel (luoi)
    viz.plot_z_comparison_batch(hourly_matrix, hampel_data, n=25, save=True, result_dir=RD)

    # Cell 25 – Grid hourly_matrix goc
    viz.plot_multiple_series(
        hourly_matrix[:25], n_cols=5, row_height=2, fig_width=20,
        title='Ma tran theo gio – goc (25 hang dau)',
        save=True, result_dir=RD, filename='11_raw_hourly_grid'
    )
    # Grid sau Hampel
    viz.plot_multiple_series(
        hampel_data[:25], n_cols=5, row_height=2, fig_width=20,
        title='Ma tran theo gio – sau Hampel (25 hang dau)',
        save=True, result_dir=RD, filename='12_hampel_grid'
    )
    # Cell 28 – Grid sau Kalman
    viz.plot_multiple_series(
        data_filtered[:25], n_cols=5, row_height=2, fig_width=20,
        title='Ma tran theo gio – sau Kalman filter (25 hang dau)',
        save=True, result_dir=RD, filename='13_kalman_grid'
    )

    # ── 4. Trich xuat dac trung ──────────────────────────────────────────────
    print("\n[4/6] Trich xuat dac trung (scale → PCA → t-SNE)...")
    data_tsne, data_scaled, data_pca = extract_features(data_filtered)

    # ── 5. Tim so cum toi uu ─────────────────────────────────────────────────
    print("\n[5/6] Tim so cum toi uu...")
    opt_result = find_optimal_clusters(
        data=data_tsne,
        data_scaled=data_scaled,
        k_range=config.K_RANGE,
        methods=['timeseries_kmeans', 'hac', 'dbscan', 'gmm'],
        save=True, result_dir=RD
    )
    recommended_k = opt_result['recommended_k'] or config.DEFAULT_N_CLUSTERS
    print(f"\nSo cum de xuat: {recommended_k}")

    # ── 6. Phan cum va visualize ──────────────────────────────────────────────
    print(f"\n[6/6] Phan cum voi k={recommended_k}...")
    clustering_results = run_all(data_tsne, data_scaled=data_scaled, n_clusters=recommended_k)

    # Chuan hoa tuple -> dict
    for key in clustering_results:
        if isinstance(clustering_results[key], tuple):
            clustering_results[key] = clustering_results[key][0]

    # Bang so sanh
    print("\nBANG SO SANH METRICS:")
    hdr = f"{'Thuat toan':<22} {'So cum':>8} {'Silhouette':>12} {'Calinski':>12} {'Davies':>10}"
    print(hdr)
    print("-" * len(hdr))
    for method, res in clustering_results.items():
        print(f"{method:<22} {res['n_clusters']:>8} {res['silhouette']:>12.4f} "
              f"{res['calinski_harabasz']:>12.2f} {res['davies_bouldin']:>10.4f}")

    # Cell 53/54 – Scatter + metric bars
    best_labels = viz.plot_clustering_results(
        clustering_results, data_tsne, save=True, result_dir=RD
    )

    # Cell 56/57 – Line plot tung cum
    viz.plot_clusters_lineplot_all_methods(
        clustering_results, data_scaled, save=True, result_dir=RD
    )

    print(f"\nHOAN THANH! Tat ca hinh anh da luu vao: {RD}")
    return clustering_results, data_tsne, data_scaled


if __name__ == '__main__':
    main()
