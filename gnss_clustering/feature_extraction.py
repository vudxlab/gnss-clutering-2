"""
Module trích xuất đặc trưng và giảm chiều dữ liệu:
- Chuẩn hóa (StandardScaler)
- PCA
- t-SNE
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import config


def scale_data(data):
    """
    Chuẩn hóa dữ liệu bằng StandardScaler (zero mean, unit variance theo từng feature).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    data_scaled : np.ndarray, same shape
    scaler : StandardScaler (đã fit)
    """
    scaler = StandardScaler()
    # fit_transform theo cột (feature), sau đó transpose lại
    data_scaled = scaler.fit_transform(data.T).T
    print(f"Đã chuẩn hóa dữ liệu: {data_scaled.shape}")
    return data_scaled, scaler


def apply_pca(data, n_components=None, random_state=None):
    """
    Giảm chiều bằng PCA.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    n_components : int
    random_state : int

    Returns
    -------
    data_pca : np.ndarray, shape (n_samples, n_components)
    pca : PCA (đã fit)
    """
    if n_components is None:
        n_components = min(config.PCA_N_COMPONENTS, data.shape[0], data.shape[1])
    if random_state is None:
        random_state = config.SEED

    print(f"Đang giảm chiều bằng PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=random_state)
    data_pca = pca.fit_transform(data)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA hoàn thành: {data_pca.shape}, tổng phương sai giải thích: {explained:.1f}%")
    return data_pca, pca


def apply_tsne(data, n_components=None, perplexity=None, learning_rate=None,
               early_exaggeration=None, metric=None, random_state=None, n_iter=2):
    """
    Giảm chiều bằng t-SNE. Mặc định chạy fit_transform 2 lần (như notebook gốc).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    n_components : int
    perplexity : float
    learning_rate : float
    early_exaggeration : float
    metric : str
    random_state : int
    n_iter : int
        Số lần lặp fit_transform (notebook gốc chạy 2 lần).

    Returns
    -------
    data_tsne : np.ndarray, shape (n_samples, n_components)
    tsne : TSNE object
    """
    if n_components is None:
        n_components = config.TSNE_N_COMPONENTS
    if perplexity is None:
        perplexity = min(config.TSNE_PERPLEXITY, len(data) - 1)
    if learning_rate is None:
        learning_rate = config.TSNE_LEARNING_RATE
    if early_exaggeration is None:
        early_exaggeration = config.TSNE_EARLY_EXAGGERATION
    if metric is None:
        metric = config.TSNE_METRIC
    if random_state is None:
        random_state = config.SEED

    print(f"Đang áp dụng t-SNE (perplexity={perplexity}, metric={metric})...")
    tsne = TSNE(
        n_components=n_components,
        init='pca',
        random_state=random_state,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        metric=metric,
    )

    current = data
    for i in range(n_iter):
        current = tsne.fit_transform(current)
        print(f"  t-SNE lần {i + 1}: {current.shape}")

    print("t-SNE hoàn thành.")
    return current, tsne


def extract_features(data_filtered, pca_components=None, tsne_n_iter=2):
    """
    Pipeline trích xuất đặc trưng đầy đủ: scale → PCA → t-SNE.

    Parameters
    ----------
    data_filtered : np.ndarray
    pca_components : int
    tsne_n_iter : int

    Returns
    -------
    data_tsne : np.ndarray, shape (n_samples, 2)
    data_scaled : np.ndarray
    data_pca : np.ndarray
    """
    data_scaled, _ = scale_data(data_filtered)
    data_pca, _ = apply_pca(data_scaled, n_components=pca_components)
    data_tsne, _ = apply_tsne(data_scaled, n_iter=tsne_n_iter)
    return data_tsne, data_scaled, data_pca
