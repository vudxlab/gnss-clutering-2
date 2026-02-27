"""
GNSS Clustering Package
=======================

Cac module:
  config              – Cau hinh chung (duong dan, hyperparameter)
  data_loader         – Tai CSV, tao ma tran ngay / gio
  preprocessing       – Hampel filter, lam min, reshape, Kalman
  feature_extraction  – StandardScaler, PCA, t-SNE  (phuong phap cu)
  clustering          – TimeSeriesKMeans, HAC, DBSCAN, GMM
  optimization        – Tim so cum toi uu (voting system)
  visualization       – Tat ca bieu do
  feature_engineering – Phuong phap moi: trich xuat vector dac trung vat ly
                        (thong ke, xu huong, tan so, cau truc thoi gian,
                        chat luong) va phan cum trong khong gian dac trung

Vi du su dung nhanh
-------------------
Phuong phap cu (cluster tren chuoi tho):
    from gnss_clustering import load_data, create_hourly_matrix
    from gnss_clustering import preprocess_pipeline, extract_features
    from gnss_clustering import run_all, plot_clustering_results

Phuong phap moi (cluster tren dac trung):
    from gnss_clustering import run_feature_based_pipeline
"""

# ── config ───────────────────────────────────────────────────────────────────
from . import config

# ── data_loader ──────────────────────────────────────────────────────────────
from .data_loader import (
    load_data,
    create_daily_matrix,
    create_hourly_matrix,
    load_cached_matrices,
    get_daily_data,
)

# ── preprocessing ─────────────────────────────────────────────────────────────
from .preprocessing import (
    hampel_filter,
    median_absolute_deviation,
    moving_average_2d,
    gaussian_filter_2d,
    butterworth_filter_2d,
    kalman_filter_2d,
    reshape_by_window,
    preprocess_pipeline,
)

# ── feature_extraction (phuong phap cu) ──────────────────────────────────────
from .feature_extraction import (
    scale_data,
    apply_pca,
    apply_tsne,
    extract_features,
)

# ── clustering ───────────────────────────────────────────────────────────────
from .clustering import (
    run_timeseries_kmeans,
    run_hierarchical_clustering,
    run_dbscan_clustering,
    run_gmm_clustering,
    run_all,
)

# ── optimization ─────────────────────────────────────────────────────────────
from .optimization import (
    find_optimal_clusters,
    create_optimization_plots,
)

# ── visualization ─────────────────────────────────────────────────────────────
from .visualization import (
    plot_daily_heatmap,
    plot_daily_timeseries,
    plot_hourly_heatmap,
    plot_hourly_overview,
    plot_hourly_analysis,
    plot_sample_hours,
    plot_first_n_hours,
    plot_z_comparison,
    plot_z_comparison_batch,
    plot_multiple_series,
    plot_clustering_results,
    plot_clusters_lineplot_all_methods,
)

# ── feature_engineering (phuong phap moi) ─────────────────────────────────────
from .feature_engineering import (
    extract_feature_matrix,
    preprocess_features,
    preprocess_features_v2,
    reduce_features_pca,
    reduce_features_umap,
    find_optimal_clusters_features,
    plot_feature_importance,
    plot_feature_scatter,
    plot_feature_scatter_umap,
    plot_cluster_feature_profiles,
    plot_cluster_timeseries,
    run_feature_based_pipeline,
    run_feature_based_pipeline_v2,
)

# ── stability ────────────────────────────────────────────────────────────────
from .stability import (
    bootstrap_stability,
    temporal_coherence,
    run_stability_analysis,
    plot_stability_results,
)

__all__ = [
    # config
    "config",
    # data_loader
    "load_data", "create_daily_matrix", "create_hourly_matrix",
    "load_cached_matrices", "get_daily_data",
    # preprocessing
    "hampel_filter", "median_absolute_deviation",
    "moving_average_2d", "gaussian_filter_2d",
    "butterworth_filter_2d", "kalman_filter_2d",
    "reshape_by_window", "preprocess_pipeline",
    # feature_extraction
    "scale_data", "apply_pca", "apply_tsne", "extract_features",
    # clustering
    "run_timeseries_kmeans", "run_hierarchical_clustering",
    "run_dbscan_clustering", "run_gmm_clustering", "run_all",
    # optimization
    "find_optimal_clusters", "create_optimization_plots",
    # visualization
    "plot_daily_heatmap", "plot_daily_timeseries",
    "plot_hourly_heatmap", "plot_hourly_overview", "plot_hourly_analysis",
    "plot_sample_hours", "plot_first_n_hours",
    "plot_z_comparison", "plot_z_comparison_batch", "plot_multiple_series",
    "plot_clustering_results", "plot_clusters_lineplot_all_methods",
    # feature_engineering
    "extract_feature_matrix", "preprocess_features", "preprocess_features_v2",
    "reduce_features_pca", "reduce_features_umap",
    "find_optimal_clusters_features",
    "plot_feature_importance", "plot_feature_scatter", "plot_feature_scatter_umap",
    "plot_cluster_feature_profiles", "plot_cluster_timeseries",
    "run_feature_based_pipeline", "run_feature_based_pipeline_v2",
    # stability
    "bootstrap_stability", "temporal_coherence",
    "run_stability_analysis", "plot_stability_results",
]
