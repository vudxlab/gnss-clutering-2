"""
Module các thuật toán phân cụm:
- TimeSeriesKMeans (DTW), với fallback sang KMeans
- Hierarchical Agglomerative Clustering (HAC)
- DBSCAN
- Gaussian Mixture Model (GMM)
"""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from . import config


def _compute_metrics(data, labels):
    """Tính 3 chỉ số đánh giá phân cụm."""
    return (
        silhouette_score(data, labels),
        calinski_harabasz_score(data, labels),
        davies_bouldin_score(data, labels),
    )


def run_timeseries_kmeans(data, data_scaled=None, n_clusters=None, metric=None,
                          max_iter=None, random_state=None, n_jobs=1):
    """
    TimeSeriesKMeans (DTW). Nếu tslearn chưa được cài, fallback sang sklearn KMeans.

    Parameters
    ----------
    data : np.ndarray
        Dữ liệu đầu vào để clustering (thường là data_tsne).
    data_scaled : np.ndarray, optional
        Dữ liệu chuẩn hóa để tính metrics. Nếu None thì dùng data.
    n_clusters : int
    metric : str  ('dtw' hoặc 'euclidean')
    max_iter : int
    random_state : int
    n_jobs : int

    Returns
    -------
    result : dict
        labels, silhouette, calinski_harabasz, davies_bouldin, n_clusters
    method_name : str
    """
    if n_clusters is None:
        n_clusters = config.DEFAULT_N_CLUSTERS
    if metric is None:
        metric = config.TS_KMEANS_METRIC
    if max_iter is None:
        max_iter = config.TS_KMEANS_MAX_ITER
    if random_state is None:
        random_state = config.SEED
    if data_scaled is None:
        data_scaled = data

    print("\nTIMESERIES K-MEANS (DTW)")
    print("-" * 40)

    try:
        from tslearn.clustering import TimeSeriesKMeans

        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric=metric,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        labels = model.fit_predict(data)
        sil, cal, dav = _compute_metrics(data_scaled, labels)

        result = {
            'labels': labels,
            'silhouette': sil,
            'calinski_harabasz': cal,
            'davies_bouldin': dav,
            'n_clusters': len(np.unique(labels)),
        }
        print(f"Hoàn thành! Silhouette: {sil:.3f}, số cụm: {result['n_clusters']}")
        return result, "TimeSeriesKMeans"

    except Exception as e:
        print(f"Lỗi TimeSeriesKMeans: {e}")
        print("Fallback sang sklearn KMeans...")
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(data_scaled)
        sil, cal, dav = _compute_metrics(data_scaled, labels)
        result = {
            'labels': labels,
            'silhouette': sil,
            'calinski_harabasz': cal,
            'davies_bouldin': dav,
            'n_clusters': len(np.unique(labels)),
        }
        print(f"KMeans fallback - Silhouette: {sil:.3f}, số cụm: {result['n_clusters']}")
        return result, "KMeans_Alternative"


def run_hierarchical_clustering(data, n_clusters=None, linkage='ward'):
    """
    Hierarchical Agglomerative Clustering.

    Parameters
    ----------
    data : np.ndarray
    n_clusters : int
    linkage : str  ('ward', 'complete', 'average', 'single')

    Returns
    -------
    result : dict hoặc None
    """
    if n_clusters is None:
        n_clusters = config.DEFAULT_N_CLUSTERS

    print("\nHIERARCHICAL AGGLOMERATIVE CLUSTERING")
    print("-" * 40)

    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(data)
        sil, cal, dav = _compute_metrics(data, labels)

        result = {
            'labels': labels,
            'silhouette': sil,
            'calinski_harabasz': cal,
            'davies_bouldin': dav,
            'n_clusters': len(np.unique(labels)),
        }
        print(f"Hoàn thành! Silhouette: {sil:.3f}, số cụm: {result['n_clusters']}")
        return result

    except Exception as e:
        print(f"Lỗi HAC: {e}")
        return None


def run_dbscan_clustering(data, min_samples=None, eps_percentile=None):
    """
    DBSCAN với eps tự động chọn theo percentile khoảng cách k-NN.

    Parameters
    ----------
    data : np.ndarray
    min_samples : int
    eps_percentile : float

    Returns
    -------
    result : dict hoặc None
    """
    if min_samples is None:
        min_samples = config.DBSCAN_MIN_SAMPLES
    if eps_percentile is None:
        eps_percentile = config.DBSCAN_EPS_PERCENTILE

    print("\nDBSCAN (DENSITY-BASED)")
    print("-" * 40)

    try:
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances_sorted = np.sort(distances[:, min_samples - 1])
        eps_auto = np.percentile(distances_sorted, eps_percentile)

        model = DBSCAN(eps=eps_auto, min_samples=min_samples)
        labels = model.fit_predict(data)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)

        if n_clusters > 1:
            mask = labels != -1
            if np.sum(mask) > 1:
                sil, cal, dav = _compute_metrics(data[mask], labels[mask])
            else:
                sil = cal = dav = -1
        else:
            sil = cal = dav = -1

        result = {
            'labels': labels,
            'silhouette': sil,
            'calinski_harabasz': cal,
            'davies_bouldin': dav,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps_auto,
        }

        print(f"Hoàn thành! eps={eps_auto:.4f}, số cụm: {n_clusters}, noise: {n_noise}")
        if sil > -1:
            print(f"Silhouette: {sil:.3f}")
        return result

    except Exception as e:
        print(f"Lỗi DBSCAN: {e}")
        return None


def run_gmm_clustering(data, n_clusters=None, random_state=None, max_iter=None):
    """
    Gaussian Mixture Model clustering.

    Parameters
    ----------
    data : np.ndarray
    n_clusters : int
    random_state : int
    max_iter : int

    Returns
    -------
    result : dict hoặc None
    """
    if n_clusters is None:
        n_clusters = config.DEFAULT_N_CLUSTERS
    if random_state is None:
        random_state = config.SEED
    if max_iter is None:
        max_iter = config.GMM_MAX_ITER

    print("\nGAUSSIAN MIXTURE MODEL")
    print("-" * 40)

    try:
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=random_state,
            max_iter=max_iter,
        )
        model.fit(data)
        labels = model.predict(data)
        sil, cal, dav = _compute_metrics(data, labels)

        result = {
            'labels': labels,
            'silhouette': sil,
            'calinski_harabasz': cal,
            'davies_bouldin': dav,
            'n_clusters': len(np.unique(labels)),
            'aic': model.aic(data),
            'bic': model.bic(data),
            'log_likelihood': model.score(data),
        }

        print(f"Hoàn thành! Silhouette: {sil:.3f}, AIC: {result['aic']:.2f}, BIC: {result['bic']:.2f}")
        return result

    except Exception as e:
        print(f"Lỗi GMM: {e}")
        return None


def run_all(data, data_scaled=None, n_clusters=None):
    """
    Chạy toàn bộ 4 thuật toán và trả về dict kết quả.

    Parameters
    ----------
    data : np.ndarray
        Dữ liệu đã giảm chiều (thường là data_tsne).
    data_scaled : np.ndarray, optional
        Dữ liệu chuẩn hóa để TimeSeriesKMeans tính metrics.
    n_clusters : int

    Returns
    -------
    clustering_results : dict
        Key: tên thuật toán, Value: dict kết quả.
    """
    if n_clusters is None:
        n_clusters = config.DEFAULT_N_CLUSTERS

    clustering_results = {}

    ts_result, _ = run_timeseries_kmeans(data, data_scaled=data_scaled, n_clusters=n_clusters)
    clustering_results['TimeSeriesKMeans'] = ts_result

    clustering_results['HAC'] = run_hierarchical_clustering(data, n_clusters=n_clusters)

    clustering_results['GMM'] = run_gmm_clustering(data, n_clusters=n_clusters)

    clustering_results['DBSCAN'] = run_dbscan_clustering(data)

    # Chuẩn hóa: nếu kết quả là tuple, lấy phần tử đầu
    for key in clustering_results:
        if isinstance(clustering_results[key], tuple):
            clustering_results[key] = clustering_results[key][0]

    return clustering_results
