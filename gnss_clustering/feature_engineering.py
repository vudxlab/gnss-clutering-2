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
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.decomposition import PCA
import pywt
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


def _wavelet_features(x, wavelet=None, max_level=None):
    """
    Group 6 – Dac trung wavelet (~10 dac trung).
    Phan tich da phan giai bang DWT, trich xuat nang luong va entropy moi muc.
    """
    if wavelet is None:
        wavelet = config.WAVELET_NAME
    if max_level is None:
        max_level = config.WAVELET_MAX_LEVEL

    max_level = min(max_level, pywt.dwt_max_level(len(x), wavelet))
    if max_level < 1:
        return {f'wavelet_energy_d{i}': 0.0 for i in range(1, 5)}

    coeffs = pywt.wavedec(x, wavelet, level=max_level)
    # coeffs[0] = approximation, coeffs[1:] = details (fine to coarse)

    feats = {}
    total_energy = sum(np.sum(c ** 2) for c in coeffs) + 1e-12

    # Energy ratio for each detail level
    for i, c in enumerate(coeffs[1:], 1):
        level_energy = np.sum(c ** 2)
        feats[f'wavelet_energy_d{i}'] = float(level_energy / total_energy)
        feats[f'wavelet_entropy_d{i}'] = float(
            stats.entropy(np.abs(c) / (np.sum(np.abs(c)) + 1e-12))
        )

    # Approximation energy ratio
    feats['wavelet_energy_approx'] = float(np.sum(coeffs[0] ** 2) / total_energy)

    # Wavelet variance ratio: detail variance / total variance
    detail_var = sum(np.var(c) for c in coeffs[1:])
    total_var = np.var(x) + 1e-12
    feats['wavelet_detail_var_ratio'] = float(detail_var / total_var)

    return feats


def _complexity_features(x):
    """
    Group 7 – Do phuc tap (5 dac trung).
    """
    n = len(x)
    feats = {}

    # Sample entropy (simplified - count matching template patterns)
    m = 2  # template length
    r = 0.2 * np.std(x)  # tolerance
    if r < 1e-12 or n < m + 1:
        feats['sample_entropy'] = 0.0
    else:
        # Count matches for length m and m+1
        def _count_matches(data, template_len, tol):
            count = 0
            N = len(data) - template_len
            for i in range(N):
                for j in range(i + 1, N):
                    if np.max(np.abs(data[i:i+template_len] - data[j:j+template_len])) <= tol:
                        count += 1
            return count
        # Subsample for speed (max 500 points)
        if n > 500:
            idx = np.linspace(0, n - 1, 500, dtype=int)
            xs = x[idx]
        else:
            xs = x
        B = _count_matches(xs, m, r)
        A = _count_matches(xs, m + 1, r)
        feats['sample_entropy'] = float(-np.log((A + 1e-12) / (B + 1e-12)))

    # Permutation entropy
    order = 3
    if n >= order:
        perms = {}
        for i in range(n - order + 1):
            pattern = tuple(np.argsort(x[i:i+order]))
            perms[pattern] = perms.get(pattern, 0) + 1
        total = sum(perms.values())
        probs = np.array([v / total for v in perms.values()])
        feats['permutation_entropy'] = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        feats['permutation_entropy'] = 0.0

    # Zero crossing rate
    zero_mean = x - np.mean(x)
    feats['zero_crossing_rate'] = float(
        np.sum(np.diff(np.sign(zero_mean)) != 0) / (n - 1) if n > 1 else 0.0
    )

    # Mean absolute difference (1st derivative roughness)
    feats['mean_abs_diff'] = float(np.mean(np.abs(np.diff(x)))) if n > 1 else 0.0

    # Coefficient of variation
    mean_abs = np.abs(np.mean(x))
    feats['coeff_variation'] = float(np.std(x) / (mean_abs + 1e-12))

    return feats


def _stationarity_features(x):
    """
    Group 8 – Dac trung tinh dung (3 dac trung).
    Chia chuoi thanh cac doan va so sanh thong ke giua chung.
    """
    n = len(x)
    feats = {}

    # Split into 4 segments and compare means/stds
    n_seg = 4
    seg_len = n // n_seg
    if seg_len < 5:
        return {'mean_shift': 0.0, 'var_shift': 0.0, 'trend_strength': 0.0}

    segments = [x[i*seg_len:(i+1)*seg_len] for i in range(n_seg)]
    seg_means = [np.mean(s) for s in segments]
    seg_stds = [np.std(s) for s in segments]

    # Mean shift: std of segment means / overall std
    overall_std = np.std(x) + 1e-12
    feats['mean_shift'] = float(np.std(seg_means) / overall_std)

    # Variance shift: std of segment stds / mean of segment stds
    mean_seg_std = np.mean(seg_stds) + 1e-12
    feats['var_shift'] = float(np.std(seg_stds) / mean_seg_std)

    # Trend strength: 1 - var(residuals) / var(x)
    t = np.arange(n, dtype=float)
    slope, intercept, _, _, _ = stats.linregress(t, x)
    residuals = x - (slope * t + intercept)
    feats['trend_strength'] = float(
        max(0, 1 - np.var(residuals) / (np.var(x) + 1e-12))
    )

    return feats


# ============================================================================
# Helper
# ============================================================================

_BASE_FEATURE_KEYS = [
    'mean', 'std', 'skewness', 'kurtosis', 'iqr',
    'trend_slope', 'trend_intercept', 'trend_r2', 'trend_resid_std',
    'dominant_freq', 'spectral_entropy', 'energy_low', 'energy_mid', 'energy_high',
    'autocorr_lag1', 'autocorr_lag5', 'autocorr_lag30', 'hurst',
    'valid_ratio', 'outlier_ratio', 'signal_range', 'snr',
]

_EXTENDED_FEATURE_KEYS = [
    'wavelet_energy_d1', 'wavelet_entropy_d1',
    'wavelet_energy_d2', 'wavelet_entropy_d2',
    'wavelet_energy_d3', 'wavelet_entropy_d3',
    'wavelet_energy_d4', 'wavelet_entropy_d4',
    'wavelet_energy_approx', 'wavelet_detail_var_ratio',
    'sample_entropy', 'permutation_entropy',
    'zero_crossing_rate', 'mean_abs_diff', 'coeff_variation',
    'mean_shift', 'var_shift', 'trend_strength',
]


def _make_zero_row(extended=False):
    keys = _BASE_FEATURE_KEYS[:]
    if extended:
        keys += _EXTENDED_FEATURE_KEYS
    return {k: 0.0 for k in keys}


# ============================================================================
# Main extraction function
# ============================================================================

def extract_feature_matrix(hourly_matrix, hampel_data=None, fs=1.0, extended=False):
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
    extended : bool
        Neu True, them wavelet + complexity + stationarity features (~18 dac trung moi).

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
            # Qua it du lieu, dien 0 – dung dummy row tu lan dau co du lieu
            row = _make_zero_row(extended=extended)
        else:
            row = {}
            row.update(_statistical_features(x))
            row.update(_trend_features(x))
            row.update(_spectral_features(x, fs=fs))
            row.update(_temporal_structure_features(x))
            row.update(_signal_quality_features(raw, filt))
            if extended:
                row.update(_wavelet_features(x))
                row.update(_complexity_features(x))
                row.update(_stationarity_features(x))

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
# Tim so cum toi uu (Feature-Based)
# ============================================================================

def find_optimal_clusters_features(hourly_matrix, hampel_data, k_range=None,
                                    result_dir=None, save=True):
    """
    Tim so cum toi uu cho Feature-Based Clustering (Phuong phap 2).

    Loop HAC va GMM qua k_range, tinh Silhouette / Calinski / Davies-Bouldin,
    chon k toi uu bang voting, ve bieu do va in bang ket qua.

    Parameters
    ----------
    hourly_matrix : np.ndarray  shape (n_hours, 3600)
    hampel_data   : np.ndarray  shape (n_hours, 3600)
    k_range       : tuple (start, end_exclusive), default config.K_RANGE
    result_dir    : str, default config.RESULT_DIR
    save          : bool

    Returns
    -------
    dict
        recommended_k, vote_count, method_results, best_k_by_method,
        X_scaled, kept_cols, feature_df
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    if k_range is None:
        k_range = config.K_RANGE
    if result_dir is None:
        result_dir = config.RESULT_DIR

    print("TIM SO CUM TOI UU – FEATURE-BASED CLUSTERING (PHUONG PHAP 2)")
    print("=" * 60)

    # 1. Trich xuat + chuan hoa dac trung
    feature_df, _ = extract_feature_matrix(hourly_matrix, hampel_data, fs=1.0)
    X_scaled, _, kept_cols = preprocess_features(feature_df)
    print(f"  Feature matrix: {X_scaled.shape[0]} mau x {X_scaled.shape[1]} dac trung")

    k_values = list(range(k_range[0], k_range[1]))
    method_results = {
        'hac': {'k_values': [], 'silhouette': [], 'calinski': [], 'davies': []},
        'gmm': {'k_values': [], 'silhouette': [], 'calinski': [], 'davies': []},
    }

    for k in k_values:
        # HAC
        try:
            lbl = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, lbl)
            cal = calinski_harabasz_score(X_scaled, lbl)
            dav = davies_bouldin_score(X_scaled, lbl)
            method_results['hac']['k_values'].append(k)
            method_results['hac']['silhouette'].append(sil)
            method_results['hac']['calinski'].append(cal)
            method_results['hac']['davies'].append(dav)
        except Exception as e:
            print(f"  HAC k={k} Error: {str(e)[:60]}")

        # GMM
        try:
            lbl = GaussianMixture(n_components=k, covariance_type='full',
                                   random_state=config.SEED).fit_predict(X_scaled)
            if len(np.unique(lbl)) > 1:
                sil = silhouette_score(X_scaled, lbl)
                cal = calinski_harabasz_score(X_scaled, lbl)
                dav = davies_bouldin_score(X_scaled, lbl)
                method_results['gmm']['k_values'].append(k)
                method_results['gmm']['silhouette'].append(sil)
                method_results['gmm']['calinski'].append(cal)
                method_results['gmm']['davies'].append(dav)
        except Exception as e:
            print(f"  GMM k={k} Error: {str(e)[:60]}")

    # In bang metrics
    print(f"\n{'k':>4} {'HAC_Sil':>10} {'HAC_Cal':>10} {'HAC_Dav':>10} "
          f"{'GMM_Sil':>10} {'GMM_Cal':>10} {'GMM_Dav':>10}")
    print("-" * 68)
    hac = method_results['hac']
    gmm = method_results['gmm']
    for i, k in enumerate(hac['k_values']):
        g_idx = gmm['k_values'].index(k) if k in gmm['k_values'] else None
        g_sil = f"{gmm['silhouette'][g_idx]:.4f}" if g_idx is not None else "  N/A  "
        g_cal = f"{gmm['calinski'][g_idx]:.2f}" if g_idx is not None else "  N/A  "
        g_dav = f"{gmm['davies'][g_idx]:.4f}" if g_idx is not None else "  N/A  "
        print(f"{k:>4} {hac['silhouette'][i]:>10.4f} {hac['calinski'][i]:>10.2f} "
              f"{hac['davies'][i]:>10.4f} {g_sil:>10} {g_cal:>10} {g_dav:>10}")

    # Best k per metric per method
    best_k = {}
    for mname, mr in method_results.items():
        if not mr['k_values']:
            continue
        best_k[mname] = {
            'best_silhouette_k': mr['k_values'][int(np.argmax(mr['silhouette']))],
            'best_calinski_k':   mr['k_values'][int(np.argmax(mr['calinski']))],
            'best_davies_k':     mr['k_values'][int(np.argmin(mr['davies']))],
        }

    # Voting
    k_votes = {}
    for mname, bk in best_k.items():
        for metric_key, kv in bk.items():
            if metric_key.endswith('_k'):
                k_votes.setdefault(kv, []).append(f"{mname}_{metric_key}")
    sorted_votes = sorted(k_votes.items(), key=lambda x: len(x[1]), reverse=True)

    print("\nVOTING (Feature-Based):")
    for kv, votes in sorted_votes:
        print(f"  k={kv}: {len(votes)} phieu  [{', '.join(votes)}]")

    recommended_k = sorted_votes[0][0] if sorted_votes else config.DEFAULT_N_CLUSTERS
    vote_count    = len(sorted_votes[0][1]) if sorted_votes else 0
    print(f"\nDE XUAT: k = {recommended_k} ({vote_count} phieu)")

    # Ve bieu do
    _plot_feature_k_analysis(method_results, best_k, recommended_k, sorted_votes,
                              save=save, result_dir=result_dir)

    return {
        'recommended_k':   recommended_k,
        'vote_count':       vote_count,
        'voting_details':   sorted_votes,
        'method_results':   method_results,
        'best_k_by_method': best_k,
        'X_scaled':         X_scaled,
        'kept_cols':        kept_cols,
        'feature_df':       feature_df,
    }


def _plot_feature_k_analysis(method_results, best_k_by_method, recommended_k,
                               voting_details, save=True, result_dir=None):
    """Ve bieu do 2x3 tom tat ket qua tim k toi uu cho Feature-Based Clustering."""
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    colors = {'hac': 'steelblue', 'gmm': 'coral'}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def _plot_metric(ax, metric_key, title, ylabel, higher_better=True):
        for mname, mr in method_results.items():
            if not mr['k_values']:
                continue
            ax.plot(mr['k_values'], mr[metric_key], 'o-',
                    color=colors[mname], label=mname.upper(), linewidth=2, markersize=6)
        if recommended_k:
            ax.axvline(recommended_k, color='red', linestyle='--',
                       alpha=0.7, label=f'k={recommended_k} (de xuat)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('So cum k')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_metric(axes[0, 0], 'silhouette', 'Silhouette vs k\n(cao hon = tot hon)', 'Silhouette')
    _plot_metric(axes[0, 1], 'calinski',   'Calinski-Harabasz vs k\n(cao hon = tot hon)', 'CH Score')
    _plot_metric(axes[0, 2], 'davies',     'Davies-Bouldin vs k\n(thap hon = tot hon)',  'DB Score')

    # Bar chart: best k per metric per method
    ax = axes[1, 0]
    method_names = list(best_k_by_method.keys())
    metrics_keys = ['best_silhouette_k', 'best_calinski_k', 'best_davies_k']
    metric_labels = ['Silhouette', 'Calinski', 'Davies']
    x = np.arange(len(method_names))
    w = 0.25
    for i, (mk, ml) in enumerate(zip(metrics_keys, metric_labels)):
        vals = [best_k_by_method[m][mk] if m in best_k_by_method else 0
                for m in method_names]
        ax.bar(x + i * w, vals, w, label=ml, alpha=0.8)
    ax.set_title('K tot nhat theo tung metric', fontsize=13, fontweight='bold')
    ax.set_xlabel('Thuat toan')
    ax.set_ylabel('k tot nhat')
    ax.set_xticks(x + w)
    ax.set_xticklabels([m.upper() for m in method_names])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heatmap metrics at recommended_k
    ax = axes[1, 1]
    perf, mlabels = [], []
    for mname, mr in method_results.items():
        if not mr['k_values'] or recommended_k not in mr['k_values']:
            continue
        idx = mr['k_values'].index(recommended_k)
        perf.append([mr['silhouette'][idx],
                     mr['calinski'][idx] / max(mr['calinski']),
                     1 - mr['davies'][idx] / max(mr['davies'])])
        mlabels.append(mname.upper())
    if perf:
        im = ax.imshow(perf, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Silhouette', 'Calinski (norm)', '1-Davies (norm)'])
        ax.set_yticks(range(len(mlabels)))
        ax.set_yticklabels(mlabels)
        ax.set_title(f'Heatmap metrics tai k={recommended_k}', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax)
        for i in range(len(mlabels)):
            for j in range(3):
                ax.text(j, i, f'{perf[i][j]:.3f}', ha='center', va='center',
                        fontsize=10, fontweight='bold')

    # Voting bar
    ax = axes[1, 2]
    if voting_details:
        top = voting_details[:5]
        kv = [str(x[0]) for x in top]
        vc = [len(x[1]) for x in top]
        bars = ax.bar(range(len(kv)), vc, color='mediumseagreen', alpha=0.85)
        ax.set_title('Top 5 k theo so phieu bau', fontsize=13, fontweight='bold')
        ax.set_xlabel('Gia tri k')
        ax.set_ylabel('So phieu')
        ax.set_xticks(range(len(kv)))
        ax.set_xticklabels([f'k={v}' for v in kv])
        for bar, c in zip(bars, vc):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                    str(c), ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Tim so cum toi uu – Feature-Based Clustering (Phuong phap 2)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'F00_optimal_k_features.png')
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


# ============================================================================
# V2 – Advanced preprocessing & clustering
# ============================================================================

def preprocess_features_v2(feature_df):
    """
    Chuan hoa nang cao: PowerTransformer (Yeo-Johnson) + RobustScaler.
    - PowerTransformer giam lech (skew) va lam phan phoi gan Gaussian hon.
    - RobustScaler chuan hoa dung median/IQR, it bi anh huong boi outlier.

    Returns
    -------
    X_scaled : np.ndarray
    scaler_pipeline : tuple (PowerTransformer, RobustScaler)
    kept_cols : list of str
    """
    stds = feature_df.std()
    kept_cols = stds[stds > 1e-8].index.tolist()
    X = feature_df[kept_cols].values.copy()

    # Replace any remaining NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    X_pt = pt.fit_transform(X)

    rs = RobustScaler()
    X_scaled = rs.fit_transform(X_pt)

    return X_scaled, (pt, rs), kept_cols


def reduce_features_umap(X_scaled, n_components=2):
    """
    Giam chieu bang UMAP (chi de visualize, khong cluster tren khong gian nay).
    """
    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        random_state=config.SEED,
    )
    X_umap = reducer.fit_transform(X_scaled)
    print(f"UMAP {n_components} chieu: hoan thanh")
    return X_umap, reducer


def _silhouette_feature_weighting(X, labels, kept_cols):
    """
    Tinh trong so cho tung dac trung dua tren dong gop vao Silhouette Score.
    Permutation-based: xao tron tung cot va do muc giam Silhouette.

    Returns
    -------
    weights : np.ndarray, shape (n_features,)
    importance_df : pd.DataFrame
    """
    from sklearn.metrics import silhouette_score

    valid = labels != -1
    if valid.sum() < 10 or len(np.unique(labels[valid])) < 2:
        n_feat = X.shape[1]
        return np.ones(n_feat), pd.DataFrame({
            'feature': kept_cols, 'importance': np.zeros(n_feat)
        })

    base_sil = silhouette_score(X[valid], labels[valid])
    importances = np.zeros(X.shape[1])

    rng = np.random.RandomState(config.SEED)
    for j in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        perm_sil = silhouette_score(X_perm[valid], labels[valid])
        importances[j] = max(0, base_sil - perm_sil)

    # Normalize to [0.5, 1.5] range to avoid zero weights
    if importances.max() > 1e-12:
        importances_norm = importances / importances.max()
        weights = 0.5 + importances_norm  # range [0.5, 1.5]
    else:
        weights = np.ones(X.shape[1])

    importance_df = pd.DataFrame({
        'feature': kept_cols[:X.shape[1]],
        'importance': importances,
        'weight': weights,
    }).sort_values('importance', ascending=False)

    return weights, importance_df


def _ensemble_clustering(X, n_clusters, n_runs=10):
    """
    Ensemble clustering bang co-association matrix.
    Chay nhieu lan HAC va GMM voi perturbation nho, tao ma tran co-association,
    roi chay HAC tren ma tran do de ra nhan cuoi cung.

    Returns
    -------
    labels : np.ndarray
    co_assoc : np.ndarray, shape (n_samples, n_samples)
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.mixture import GaussianMixture

    n = X.shape[0]
    co_assoc = np.zeros((n, n))
    rng = np.random.RandomState(config.SEED)

    for run in range(n_runs):
        # Perturbation: add small Gaussian noise
        noise_scale = 0.05 * X.std(axis=0)
        X_noisy = X + rng.randn(*X.shape) * noise_scale

        # HAC
        lbl_hac = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward'
        ).fit_predict(X_noisy)
        for i in range(n):
            for j in range(i + 1, n):
                if lbl_hac[i] == lbl_hac[j]:
                    co_assoc[i, j] += 1
                    co_assoc[j, i] += 1

        # GMM
        lbl_gmm = GaussianMixture(
            n_components=n_clusters, covariance_type='full',
            random_state=config.SEED + run
        ).fit_predict(X_noisy)
        for i in range(n):
            for j in range(i + 1, n):
                if lbl_gmm[i] == lbl_gmm[j]:
                    co_assoc[i, j] += 1
                    co_assoc[j, i] += 1

    # Normalize
    co_assoc /= (2 * n_runs)
    np.fill_diagonal(co_assoc, 1.0)

    # Cluster on co-association matrix (distance = 1 - co_assoc)
    dist_matrix = 1.0 - co_assoc
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average'
    ).fit_predict(dist_matrix)

    return labels, co_assoc


def plot_feature_scatter_umap(X_2d, labels, method_name, dim_method='UMAP',
                               result_dir=None, save=True):
    """Scatter plot 2D (UMAP hoac PCA) to mau theo nhan phan cum."""
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
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS[i % len(COLORS)], s=80, marker=marker,
                   alpha=0.7, edgecolors='k', linewidths=0.3,
                   label=f'{lbl_txt} (n={mask.sum()})')

    ax.set_title(f'{method_name} – Feature-Based v2\n({dim_method} space)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{dim_method}1')
    ax.set_ylabel(f'{dim_method}2')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        safe = method_name.replace(' ', '_').lower()
        path = os.path.join(result_dir, f'F2_03_scatter_{safe}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def _plot_feature_weights(importance_df, result_dir=None, save=True):
    """Ve bieu do trong so dac trung."""
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    df_sorted = importance_df.sort_values('importance', ascending=True)
    colors = plt.cm.RdYlGn(df_sorted['weight'].values / df_sorted['weight'].max())
    ax.barh(df_sorted['feature'], df_sorted['importance'], color=colors)
    ax.set_xlabel('Importance (Silhouette drop)')
    ax.set_title('Feature Importance – Silhouette-Guided Weighting',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'F2_01_feature_weights.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def _plot_co_association(co_assoc, labels, result_dir=None, save=True):
    """Ve co-association matrix (reordered by cluster)."""
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    order = np.argsort(labels)
    co_sorted = co_assoc[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(co_sorted, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Co-association')
    ax.set_title('Ensemble Co-Association Matrix\n(sap xep theo nhan cum)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Mau (sorted)')
    ax.set_ylabel('Mau (sorted)')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'F2_06_co_association.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


# ============================================================================
# Pipeline V2 tich hop
# ============================================================================

def run_feature_based_pipeline_v2(hourly_matrix, hampel_data, valid_hours_info,
                                   n_clusters=4, result_dir=None):
    """
    Pipeline phan cum V2 (cai tien):
      1. Trich xuat dac trung mo rong (22 goc + 18 moi = ~40 dac trung)
      2. Chuan hoa nang cao (PowerTransformer + RobustScaler)
      3. Phan cum initial (HAC) tren khong gian full-dim
      4. Silhouette-guided feature weighting
      5. Tai phan cum tren khong gian co trong so
      6. HDBSCAN, GMM, Ensemble clustering
      7. UMAP 2D de visualize
      8. Ve tat ca bieu do

    Returns
    -------
    results : dict
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    import hdbscan

    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("FEATURE-BASED CLUSTERING V2 PIPELINE (CAI TIEN)")
    print("=" * 60)

    # --- Buoc 1: Trich xuat dac trung mo rong ---
    print("\n[1] Trich xuat dac trung mo rong (extended=True)...")
    feature_df, feat_names = extract_feature_matrix(
        hourly_matrix, hampel_data, fs=1.0, extended=True
    )
    print(f"    Feature matrix: {feature_df.shape}  ({len(feat_names)} dac trung)")

    # --- Buoc 2: Chuan hoa nang cao ---
    print("\n[2] Chuan hoa nang cao (PowerTransformer + RobustScaler)...")
    X_scaled, scaler_pipeline, kept_cols = preprocess_features_v2(feature_df)
    print(f"    Giu lai {len(kept_cols)}/{len(feat_names)} dac trung")

    # --- Buoc 3: Initial HAC tren full-dim ---
    print(f"\n[3] Initial HAC (k={n_clusters}) tren {X_scaled.shape[1]}D...")
    lbl_init = AgglomerativeClustering(
        n_clusters=n_clusters, linkage='ward'
    ).fit_predict(X_scaled)

    # --- Buoc 4: Silhouette-guided feature weighting ---
    print("\n[4] Silhouette-guided feature weighting...")
    weights, importance_df = _silhouette_feature_weighting(
        X_scaled, lbl_init, kept_cols
    )
    print(f"    Top 5 dac trung quan trong nhat:")
    for _, row in importance_df.head(5).iterrows():
        print(f"      {row['feature']:30s} importance={row['importance']:.4f}  "
              f"weight={row['weight']:.3f}")

    _plot_feature_weights(importance_df, result_dir=result_dir)

    # Apply weights
    X_weighted = X_scaled * weights[np.newaxis, :]

    # --- Buoc 5: Phan cum chinh thuc tren khong gian co trong so ---
    print(f"\n[5] Phan cum chinh thuc (k={n_clusters}) tren khong gian co trong so...")

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

    clustering_results = {}

    # HAC (weighted)
    lbl_hac = AgglomerativeClustering(
        n_clusters=n_clusters, linkage='ward'
    ).fit_predict(X_weighted)
    sil, cal, dav = _metrics(X_weighted, lbl_hac)
    clustering_results['HAC'] = {
        'labels': lbl_hac, 'silhouette': sil,
        'calinski_harabasz': cal, 'davies_bouldin': dav,
        'n_clusters': len(np.unique(lbl_hac)),
    }
    print(f"    HAC      : Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}")

    # GMM (weighted)
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type='full',
        random_state=config.SEED,
    )
    gmm.fit(X_weighted)
    lbl_gmm = gmm.predict(X_weighted)
    sil, cal, dav = _metrics(X_weighted, lbl_gmm)
    clustering_results['GMM'] = {
        'labels': lbl_gmm, 'silhouette': sil,
        'calinski_harabasz': cal, 'davies_bouldin': dav,
        'n_clusters': len(np.unique(lbl_gmm)),
        'aic': gmm.aic(X_weighted), 'bic': gmm.bic(X_weighted),
    }
    print(f"    GMM      : Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}")

    # HDBSCAN (weighted)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        metric='euclidean',
    )
    lbl_hdb = clusterer.fit_predict(X_weighted)
    n_cls_hdb = len(np.unique(lbl_hdb[lbl_hdb != -1]))
    sil, cal, dav = _metrics(X_weighted, lbl_hdb)
    clustering_results['HDBSCAN'] = {
        'labels': lbl_hdb, 'silhouette': sil,
        'calinski_harabasz': cal, 'davies_bouldin': dav,
        'n_clusters': n_cls_hdb,
        'n_noise': int((lbl_hdb == -1).sum()),
    }
    print(f"    HDBSCAN  : Sil={sil:.3f}, {n_cls_hdb} cum, "
          f"noise={int((lbl_hdb == -1).sum())}")

    # Ensemble clustering
    print(f"\n[6] Ensemble clustering (10 runs x 2 methods)...")
    lbl_ens, co_assoc = _ensemble_clustering(X_weighted, n_clusters, n_runs=10)
    sil, cal, dav = _metrics(X_weighted, lbl_ens)
    clustering_results['Ensemble'] = {
        'labels': lbl_ens, 'silhouette': sil,
        'calinski_harabasz': cal, 'davies_bouldin': dav,
        'n_clusters': len(np.unique(lbl_ens)),
    }
    print(f"    Ensemble : Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}")
    _plot_co_association(co_assoc, lbl_ens, result_dir=result_dir)

    # --- Buoc 7: UMAP 2D de visualize ---
    print("\n[7] UMAP 2D de visualize...")
    X_umap, umap_reducer = reduce_features_umap(X_weighted, n_components=2)

    # PCA 2D cung de so sanh
    X_pca, pca = reduce_features_pca(X_weighted, n_components=2)

    # --- Buoc 8: Ve bieu do ---
    print("\n[8] Visualize ket qua...")

    # Ve feature boxplot (extended)
    plot_feature_importance(feature_df, kept_cols, pca, result_dir=result_dir)

    for method_name, res in clustering_results.items():
        # UMAP scatter
        plot_feature_scatter_umap(
            X_umap, res['labels'], method_name,
            dim_method='UMAP', result_dir=result_dir,
        )
        # PCA scatter
        plot_feature_scatter(
            X_pca, res['labels'], f'{method_name}_v2',
            result_dir=result_dir,
        )
        # Cluster profiles
        plot_cluster_feature_profiles(
            feature_df, kept_cols, res['labels'],
            f'{method_name}_v2', result_dir=result_dir,
        )
        # Time series per cluster
        plot_cluster_timeseries(
            hourly_matrix, res['labels'],
            valid_hours_info, f'{method_name}_v2', result_dir=result_dir,
        )

    # --- Bang so sanh ---
    print("\nBANG SO SANH (Feature-Based Clustering V2):")
    hdr = f"{'Method':<12} {'k':>4} {'Silhouette':>12} {'Calinski':>10} {'Davies':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m, r in clustering_results.items():
        print(f"{m:<12} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
              f"{r['calinski_harabasz']:>10.2f} {r['davies_bouldin']:>8.4f}")

    return {
        'feature_df':         feature_df,
        'kept_cols':          kept_cols,
        'X_scaled':           X_scaled,
        'X_weighted':         X_weighted,
        'X_umap':             X_umap,
        'X_pca':              X_pca,
        'pca':                pca,
        'feature_weights':    weights,
        'importance_df':      importance_df,
        'clustering_results': clustering_results,
    }
