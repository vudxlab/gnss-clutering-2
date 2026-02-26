"""
Feature-Based Clustering approach for GNSS displacement time series.

Thay vi cluster truc tiep tren chuoi gia tri tho (360 chieu),
module nay trich xuat vector dac trung vat ly co y nghia (~20 chieu)
tu moi doan du lieu mot gio:

  Group 1 – Thong ke phan phoi  : mean, std, skewness, kurtosis, IQR
  Group 2 – Xu huong (trend)    : slope, intercept, R2, residual_std
  Group 3 – Tan so (spectral)   : dominant_freq, spectral_entropy,
                                   low/mid/high energy ratio, bandwidth
  Group 4 – Cau truc thoi gian  : autocorr_lag1/lag5/lag30, Hurst exponent
  Group 5 – Chat luong tin hieu  : valid_ratio, outlier_ratio (Hampel),
                                   SNR, range

Uu diem so voi phuong phap cu:
  - Khong phu thuoc pha (phase-invariant)
  - Chieu thap, moi chieu co y nghia vat ly ro rang
  - De giai thich ket qua phan cum
  - Khong can t-SNE: PCA 2D da du de visualize
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

from . import config


# ============================================================================
# Feature extractors
# ============================================================================

def _statistical_features(x):
    """
    Group 1 – Dac trung phan phoi (5 dac trung).
    x: 1-D array, khong co NaN.
    """
    return {
        'mean':     float(np.mean(x)),
        'std':      float(np.std(x)),
        'skewness': float(stats.skew(x)),
        'kurtosis': float(stats.kurtosis(x)),
        'iqr':      float(np.percentile(x, 75) - np.percentile(x, 25)),
    }


def _trend_features(x):
    """
    Group 2 – Xu huong tuyen tinh (4 dac trung).
    Fit y = a*t + b; tra ve slope, intercept, R2, residual_std.
    """
    t = np.arange(len(x), dtype=float)
    slope, intercept, r_value, _, _ = stats.linregress(t, x)
    residuals = x - (slope * t + intercept)
    return {
        'trend_slope':     float(slope),
        'trend_intercept': float(intercept),
        'trend_r2':        float(r_value ** 2),
        'trend_resid_std': float(np.std(residuals)),
    }


def _spectral_features(x, fs=1.0):
    """
    Group 3 – Dac trung tan so (5 dac trung).
    fs: tan so lay mau (Hz). Voi du lieu 1 mau/giay thi fs=1.
    """
    n = len(x)
    freqs = rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(rfft(x - x.mean()))   # bo DC

    # Dominant frequency
    dominant_idx = np.argmax(fft_mag[1:]) + 1   # bo f=0
    dominant_freq = float(freqs[dominant_idx])

    # Spectral entropy (Shannon)
    psd = fft_mag ** 2
    psd_norm = psd / (psd.sum() + 1e-12)
    spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))

    # Energy in 3 dai tan so (low / mid / high)
    total_e = psd.sum() + 1e-12
    cutoffs = [fs * 0.1, fs * 0.3]  # 10% va 30% Nyquist
    low  = psd[freqs <  cutoffs[0]].sum() / total_e
    mid  = psd[(freqs >= cutoffs[0]) & (freqs < cutoffs[1])].sum() / total_e
    high = psd[freqs >= cutoffs[1]].sum() / total_e

    return {
        'dominant_freq':     dominant_freq,
        'spectral_entropy':  spectral_entropy,
        'energy_low':        float(low),
        'energy_mid':        float(mid),
        'energy_high':       float(high),
    }


def _temporal_structure_features(x):
    """
    Group 4 – Cau truc thoi gian (4 dac trung).
    - Autocorrelation tai lag 1, 5, 30 (don vi mau)
    - Hurst exponent (R/S method) – do luu tru/ xu huong dai han
    """
    # Autocorrelation
    def autocorr(arr, lag):
        if len(arr) <= lag:
            return 0.0
        c = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    lags = [1, 5, 30]
    ac = {f'autocorr_lag{lag}': autocorr(x, lag) for lag in lags}

    # Hurst exponent (simplified R/S)
    hurst = _hurst_exponent(x)

    return {**ac, 'hurst': hurst}


def _hurst_exponent(x, min_n=10):
    """
    Tinh Hurst exponent bang phuong phap R/S.
    H < 0.5 : mean-reverting (anti-persistent)
    H ~ 0.5 : random walk
    H > 0.5 : trending (persistent)
    """
    n = len(x)
    if n < 20:
        return 0.5

    ns = []
    rs_vals = []
    for sub_n in [n // 4, n // 2, n]:
        if sub_n < min_n:
            continue
        sub = x[:sub_n]
        mean_sub = sub.mean()
        deviation = np.cumsum(sub - mean_sub)
        R = deviation.max() - deviation.min()
        S = sub.std()
        if S > 0:
            ns.append(sub_n)
            rs_vals.append(R / S)

    if len(ns) < 2:
        return 0.5

    log_n  = np.log(ns)
    log_rs = np.log(rs_vals)
    slope, _, _, _, _ = stats.linregress(log_n, log_rs)
    return float(np.clip(slope, 0.0, 1.0))


def _signal_quality_features(raw_row, filtered_row=None):
    """
    Group 5 – Chat luong tin hieu (4 dac trung).
    raw_row    : du lieu goc (co the co NaN)
    filtered_row: du lieu sau Hampel (khong NaN)
    """
    valid_mask = ~np.isnan(raw_row)
    valid_ratio = float(valid_mask.sum() / len(raw_row))

    # Bien do dao dong
    valid_vals = raw_row[valid_mask]
    sig_range = float(valid_vals.max() - valid_vals.min()) if len(valid_vals) > 1 else 0.0

    # Outlier ratio (ty le diem bi Hampel phat hien la ngoai lai)
    outlier_ratio = 0.0
    if filtered_row is not None and len(valid_vals) > 0:
        diff = np.abs(valid_vals - filtered_row[valid_mask])
        outlier_ratio = float((diff > 1e-9).sum() / len(valid_vals))

    # SNR uoc luong: mean / std (tranh chia 0)
    snr = float(np.abs(valid_vals.mean()) / (valid_vals.std() + 1e-12)) if len(valid_vals) > 1 else 0.0

    return {
        'valid_ratio':   valid_ratio,
        'outlier_ratio': outlier_ratio,
        'signal_range':  sig_range,
        'snr':           snr,
    }


# ============================================================================
# Main extraction function
# ============================================================================

def extract_feature_matrix(hourly_matrix, hampel_data=None, fs=1.0):
    """
    Trich xuat ma tran dac trung tu hourly_matrix.

    Parameters
    ----------
    hourly_matrix : np.ndarray, shape (n_hours, 3600)
        Du lieu theo gio (co the co NaN).
    hampel_data : np.ndarray, shape (n_hours, 3600), optional
        Du lieu sau Hampel filter (khong NaN). Neu None thi dung hourly_matrix.
    fs : float
        Tan so lay mau (mau/giay). Mac dinh 1.0 (1 mau/giay = du lieu raw).
        Neu dung reshape_data (1 mau / 10 giay) thi fs = 0.1.

    Returns
    -------
    feature_df : pd.DataFrame, shape (n_hours, n_features)
        Moi hang la vector dac trung cua 1 gio.
    feature_names : list of str
    """
    if hampel_data is None:
        hampel_data = hourly_matrix

    records = []
    for i in range(len(hourly_matrix)):
        raw = hourly_matrix[i]
        filt = hampel_data[i]

        # Lay diem hop le de tinh dac trung tren filt (sau Hampel, khong NaN)
        valid_mask = ~np.isnan(filt)
        x = filt[valid_mask]

        if len(x) < 20:
            # Qua it du lieu, dien 0
            row = {k: 0.0 for k in
                   ['mean','std','skewness','kurtosis','iqr',
                    'trend_slope','trend_intercept','trend_r2','trend_resid_std',
                    'dominant_freq','spectral_entropy','energy_low','energy_mid','energy_high',
                    'autocorr_lag1','autocorr_lag5','autocorr_lag30','hurst',
                    'valid_ratio','outlier_ratio','signal_range','snr']}
        else:
            row = {}
            row.update(_statistical_features(x))
            row.update(_trend_features(x))
            row.update(_spectral_features(x, fs=fs))
            row.update(_temporal_structure_features(x))
            row.update(_signal_quality_features(raw, filt))

        records.append(row)

    feature_df = pd.DataFrame(records)
    return feature_df, list(feature_df.columns)


# ============================================================================
# Preprocessing dac trung
# ============================================================================

def preprocess_features(feature_df):
    """
    Chuan hoa dac trung (StandardScaler) + loai bo cot co phuong sai gan 0.

    Returns
    -------
    feature_scaled : np.ndarray
    scaler : StandardScaler
    kept_cols : list of str
    """
    # Loai bo cot hang so (std ~ 0)
    stds = feature_df.std()
    kept_cols = stds[stds > 1e-8].index.tolist()
    X = feature_df[kept_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, kept_cols


def reduce_features_pca(X_scaled, n_components=2):
    """
    Giam chieu bang PCA. Voi ~20 dac trung thi 2D da giu duoc phan lon thong tin.
    """
    n_comp = min(n_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=config.SEED)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()
    print(f"PCA {n_comp} chieu: giai thich {explained[-1]*100:.1f}% phuong sai")
    return X_pca, pca


# ============================================================================
# Visualization dac trung
# ============================================================================

def plot_feature_importance(feature_df, kept_cols, pca, result_dir=None, save=True):
    """
    Ve 2 bieu do:
    1. Boxplot phan phoi tung dac trung (da chuan hoa)
    2. PCA loadings – dong gop cua tung dac trung vao PC1, PC2
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    X_std = StandardScaler().fit_transform(feature_df[kept_cols].values)

    # --- Bieu do 1: Boxplot ---
    fig1, ax1 = plt.subplots(figsize=(18, 5))
    ax1.boxplot(X_std, labels=kept_cols, vert=True)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Phan phoi cac dac trung (da chuan hoa)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dac trung')
    ax1.set_ylabel('Gia tri chuan hoa')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'F01_feature_boxplot.png')
        fig1.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()

    # --- Bieu do 2: PCA loadings ---
    if pca is not None and pca.n_components_ >= 2:
        loadings = pd.DataFrame(
            pca.components_[:2].T,
            index=kept_cols,
            columns=['PC1', 'PC2']
        )
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        for ax, pc in zip(axes2, ['PC1', 'PC2']):
            vals = loadings[pc].sort_values()
            colors = ['salmon' if v < 0 else 'steelblue' for v in vals]
            ax.barh(vals.index, vals.values, color=colors)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title(f'{pc} Loadings\n(Dong gop cua tung dac trung)',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Loading')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            path = os.path.join(result_dir, 'F02_pca_loadings.png')
            fig2.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"  [saved] {path}")
        plt.show()


def plot_feature_scatter(X_pca, labels, method_name, feature_names_short=None,
                         result_dir=None, save=True):
    """
    Scatter plot 2D (PC1 vs PC2) to mau theo nhan phan cum.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    unique_lbls = np.unique(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, lbl in enumerate(unique_lbls):
        mask = labels == lbl
        lbl_txt = 'Noise' if lbl == -1 else f'Cluster {lbl}'
        marker = 'x' if lbl == -1 else 'o'
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORS[i % len(COLORS)], s=80, marker=marker,
                   alpha=0.7, edgecolors='k', linewidths=0.3,
                   label=f'{lbl_txt} (n={mask.sum()})')

    ax.set_title(f'{method_name} – Feature-Based Clustering\n(PCA space)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        safe = method_name.replace(' ', '_').lower()
        path = os.path.join(result_dir, f'F03_scatter_{safe}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_cluster_feature_profiles(feature_df, kept_cols, labels, method_name,
                                   result_dir=None, save=True):
    """
    Radar / bar chart the hien gia tri trung binh cua tung dac trung trong moi cum.
    Giup giai thich y nghia vat ly cua moi cum.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    X = StandardScaler().fit_transform(feature_df[kept_cols].values)
    unique_lbls = sorted(set(labels))
    if -1 in unique_lbls:
        unique_lbls.remove(-1)

    n_feat = len(kept_cols)
    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    fig, ax = plt.subplots(figsize=(max(16, n_feat), 5))
    x_pos = np.arange(n_feat)
    width = 0.8 / max(len(unique_lbls), 1)

    for i, lbl in enumerate(unique_lbls):
        mask = labels == lbl
        means = X[mask].mean(axis=0)
        errs  = X[mask].std(axis=0)
        ax.bar(x_pos + i * width, means, width,
               label=f'Cluster {lbl} (n={mask.sum()})',
               color=COLORS[i % len(COLORS)], alpha=0.75, yerr=errs, capsize=2)

    ax.set_xticks(x_pos + width * (len(unique_lbls) - 1) / 2)
    ax.set_xticklabels(kept_cols, rotation=45, ha='right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Gia tri dac trung chuan hoa (mean ± std)')
    ax.set_title(f'{method_name} – Profile dac trung trung binh moi cum\n'
                 f'(> 0: cao hon TB toan bo; < 0: thap hon TB)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save:
        safe = method_name.replace(' ', '_').lower()
        path = os.path.join(result_dir, f'F04_cluster_profiles_{safe}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_cluster_timeseries(hourly_matrix, labels, valid_hours_info, method_name,
                             result_dir=None, save=True):
    """
    Ve chuoi thoi gian trung binh ± std cho tung cum (tren du lieu goc).
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    unique_lbls = sorted(set(labels))
    if -1 in unique_lbls:
        unique_lbls.remove(-1)

    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    t = np.arange(3600) / 60  # truc thoi gian theo phut

    fig, axes = plt.subplots(len(unique_lbls), 1,
                             figsize=(16, 4 * len(unique_lbls)), sharex=True)
    if len(unique_lbls) == 1:
        axes = [axes]

    for i, lbl in enumerate(unique_lbls):
        ax = axes[i]
        mask = labels == lbl
        idxs = np.where(mask)[0]

        for idx in idxs:
            s = hourly_matrix[idx, :]
            vm = ~np.isnan(s)
            if vm.sum() > 0:
                ax.plot(t[vm], s[vm], alpha=0.25, linewidth=0.5, color=COLORS[i % len(COLORS)])

        cdata = hourly_matrix[idxs, :]
        mean  = np.nanmean(cdata, axis=0)
        std   = np.nanstd(cdata, axis=0)
        vm2   = ~np.isnan(mean)
        if vm2.sum() > 0:
            ax.plot(t[vm2], mean[vm2], color='black', linewidth=2.5,
                    label='Mean', zorder=5)
            ax.fill_between(t[vm2], (mean-std)[vm2], (mean+std)[vm2],
                            alpha=0.2, color='gray', label='±1 std')

        ax.set_title(f'Cluster {lbl}  –  {mask.sum()} gio', fontsize=12, fontweight='bold')
        ax.set_ylabel('h_Coord (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Phut trong gio')
    fig.suptitle(f'{method_name} – Chuoi thoi gian trung binh tung cum (du lieu goc)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        safe = method_name.replace(' ', '_').lower()
        path = os.path.join(result_dir, f'F05_cluster_ts_{safe}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


# ============================================================================
# Pipeline tich hop
# ============================================================================

def run_feature_based_pipeline(hourly_matrix, hampel_data, valid_hours_info,
                                 n_clusters=4, result_dir=None):
    """
    Pipeline phan cum dua tren dac trung (Feature-Based Clustering):
      1. Trich xuat dac trung
      2. Chuan hoa + PCA 2D
      3. HAC, GMM, DBSCAN
      4. Visualize ket qua

    Parameters
    ----------
    hourly_matrix : np.ndarray, shape (n_hours, 3600)
    hampel_data   : np.ndarray, shape (n_hours, 3600)
    valid_hours_info : pd.DataFrame
    n_clusters    : int
    result_dir    : str

    Returns
    -------
    results : dict
        Chua feature_df, X_scaled, X_pca, clustering_results
    """
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import NearestNeighbors

    if result_dir is None:
        result_dir = config.RESULT_DIR

    print("=" * 60)
    print("FEATURE-BASED CLUSTERING PIPELINE")
    print("=" * 60)

    # --- Buoc 1: Trich xuat dac trung ---
    print("\n[1] Trich xuat dac trung...")
    feature_df, feat_names = extract_feature_matrix(hourly_matrix, hampel_data, fs=1.0)
    print(f"    Feature matrix: {feature_df.shape}  ({len(feat_names)} dac trung)")
    print(f"    Cac dac trung: {feat_names}")

    # --- Buoc 2: Chuan hoa ---
    print("\n[2] Chuan hoa dac trung...")
    X_scaled, scaler, kept_cols = preprocess_features(feature_df)
    print(f"    Giu lai {len(kept_cols)}/{len(feat_names)} dac trung (loai bo hang so)")

    # --- Buoc 3: PCA ---
    print("\n[3] PCA giam chieu...")
    X_pca, pca = reduce_features_pca(X_scaled, n_components=2)

    # Ve phan phoi dac trung va loadings
    plot_feature_importance(feature_df, kept_cols, pca, result_dir=result_dir)

    # --- Buoc 4: Phan cum ---
    print(f"\n[4] Phan cum (k={n_clusters})...")
    clustering_results = {}

    def _metrics(data, labels):
        unique = np.unique(labels[labels != -1])
        if len(unique) < 2:
            return -1, -1, -1
        mask = labels != -1
        if mask.sum() < 2:
            return -1, -1, -1
        return (silhouette_score(data[mask], labels[mask]),
                calinski_harabasz_score(data[mask], labels[mask]),
                davies_bouldin_score(data[mask], labels[mask]))

    # HAC
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    lbl_hac = hac.fit_predict(X_scaled)
    sil, cal, dav = _metrics(X_scaled, lbl_hac)
    clustering_results['HAC'] = {'labels': lbl_hac, 'silhouette': sil,
                                  'calinski_harabasz': cal, 'davies_bouldin': dav,
                                  'n_clusters': len(np.unique(lbl_hac))}
    print(f"    HAC     : Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}")

    # GMM
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full',
                          random_state=config.SEED)
    gmm.fit(X_scaled)
    lbl_gmm = gmm.predict(X_scaled)
    sil, cal, dav = _metrics(X_scaled, lbl_gmm)
    clustering_results['GMM'] = {'labels': lbl_gmm, 'silhouette': sil,
                                  'calinski_harabasz': cal, 'davies_bouldin': dav,
                                  'n_clusters': len(np.unique(lbl_gmm)),
                                  'aic': gmm.aic(X_scaled), 'bic': gmm.bic(X_scaled)}
    print(f"    GMM     : Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}")

    # DBSCAN (eps tu dong)
    nbrs = NearestNeighbors(n_neighbors=4).fit(X_scaled)
    dists, _ = nbrs.kneighbors(X_scaled)
    eps_auto = float(np.percentile(np.sort(dists[:, 3]), 90))
    dbs = DBSCAN(eps=eps_auto, min_samples=4)
    lbl_dbs = dbs.fit_predict(X_scaled)
    n_cls_dbs = len(np.unique(lbl_dbs[lbl_dbs != -1]))
    sil, cal, dav = _metrics(X_scaled, lbl_dbs)
    clustering_results['DBSCAN'] = {'labels': lbl_dbs, 'silhouette': sil,
                                     'calinski_harabasz': cal, 'davies_bouldin': dav,
                                     'n_clusters': n_cls_dbs,
                                     'n_noise': int((lbl_dbs == -1).sum()),
                                     'eps': eps_auto}
    print(f"    DBSCAN  : Sil={sil:.3f}, {n_cls_dbs} cum, "
          f"noise={int((lbl_dbs==-1).sum())}")

    # --- Buoc 5: Visualize ---
    print("\n[5] Visualize ket qua...")
    for method_name, res in clustering_results.items():
        plot_feature_scatter(X_pca, res['labels'], method_name, result_dir=result_dir)
        plot_cluster_feature_profiles(feature_df, kept_cols, res['labels'],
                                       method_name, result_dir=result_dir)
        plot_cluster_timeseries(hourly_matrix, res['labels'],
                                 valid_hours_info, method_name, result_dir=result_dir)

    # --- Bang so sanh ---
    print("\nBANG SO SANH (Feature-Based Clustering):")
    hdr = f"{'Method':<10} {'k':>4} {'Silhouette':>12} {'Calinski':>10} {'Davies':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m, r in clustering_results.items():
        print(f"{m:<10} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
              f"{r['calinski_harabasz']:>10.2f} {r['davies_bouldin']:>8.4f}")

    return {
        'feature_df':          feature_df,
        'kept_cols':           kept_cols,
        'X_scaled':            X_scaled,
        'X_pca':               X_pca,
        'pca':                 pca,
        'clustering_results':  clustering_results,
    }
