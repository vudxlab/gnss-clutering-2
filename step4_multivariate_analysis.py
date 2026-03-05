"""
BUOC 4 – Phan tich da bien (Multivariate) cho 2 tram 4E va 4W
================================================================

Phan tich dong thoi 3 toa do (X, Y, h) cua ca 2 tram GNSS:
  1. Tao ma tran da bien cho moi tram (moi gio = vector 3*3600 = 10800 chieu)
  2. Tien xu ly (Hampel + reshape + Kalman) cho tung kenh
  3. Pipeline PP1: Scale -> PCA -> t-SNE -> Clustering
  4. Phan tich tuong quan noi cum (intra-cluster correlation):
     - Tuong quan giua X, Y, h trong tung cum
     - Tuong quan giua 2 tram trong tung cum
     - Cross-correlation giua cac kenh

Chay:
    python step4_multivariate_analysis.py --no-display
    python step4_multivariate_analysis.py --k 4 --no-display
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='B4 – Phan tich da bien 4E + 4W')
parser.add_argument('--k', type=int, default=4, help='So cum (default: 4)')
parser.add_argument('--no-display', action='store_true', help='Khong hien thi cua so')
args = parser.parse_args()

# ---------------------------------------------------------------------------
#  Matplotlib backend
# ---------------------------------------------------------------------------
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Import
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate

from gnss_clustering import config
from gnss_clustering.preprocessing import hampel_filter, reshape_by_window, kalman_filter_2d

plt.style.use(config.MATPLOTLIB_STYLE)

RD = os.path.join(config.RESULT_DIR, '08_multivariate')
os.makedirs(RD, exist_ok=True)


# ============================================================================
#  1. TAI VA CHUAN BI DU LIEU DA BIEN
# ============================================================================

def load_station_data(filepath):
    """Tai du lieu 1 tram, tra ve DataFrame voi Timestamp, X, Y, h."""
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    df['Second_of_day'] = (
        df['Timestamp'].dt.hour * 3600
        + df['Timestamp'].dt.minute * 60
        + df['Timestamp'].dt.second
    )
    return df


def create_multivariate_hourly_matrix(df, coord_cols=['X_Coord', 'Y_Coord', 'h_Coord'],
                                      missing_threshold=5.0):
    """
    Tao ma tran theo gio cho nhieu bien.
    Cho phep toi da missing_threshold% du lieu thieu (se noi suy).
    """
    unique_dates = sorted(df['Date'].unique())
    hourly_data = {col: [] for col in coord_cols}
    hourly_info = []

    for date in unique_dates:
        day_data = df[df['Date'] == date]
        for hour in range(24):
            start_s = hour * 3600
            end_s = (hour + 1) * 3600

            hour_mask = (day_data['Second_of_day'] >= start_s) & (day_data['Second_of_day'] < end_s)
            hour_data = day_data[hour_mask]

            # Tao vector 3600 cho moi kenh
            vectors = {}
            valid = True
            max_missing = 0
            for col in coord_cols:
                vec = np.full(3600, np.nan)
                for _, row in hour_data.iterrows():
                    idx = int(row['Second_of_day']) - start_s
                    if 0 <= idx < 3600:
                        vec[idx] = row[col]
                nan_pct = np.isnan(vec).sum() / 3600 * 100
                max_missing = max(max_missing, nan_pct)
                if nan_pct > missing_threshold:
                    valid = False
                    break
                # Noi suy NaN bang linear interpolation
                if nan_pct > 0:
                    s = pd.Series(vec)
                    vec = s.interpolate(method='linear').ffill().bfill().values
                vectors[col] = vec

            if valid:
                for col in coord_cols:
                    hourly_data[col].append(vectors[col])
                hourly_info.append({
                    'date': date, 'hour': hour,
                    'datetime': f"{date} {hour:02d}:00:00",
                    'max_missing_pct': max_missing,
                })

    matrices = {col: np.array(hourly_data[col]) for col in coord_cols}
    info_df = pd.DataFrame(hourly_info)
    return matrices, info_df


def preprocess_channel(matrix):
    """Tien xu ly 1 kenh: Hampel -> reshape -> Kalman."""
    filtered, _ = hampel_filter(matrix)
    reshaped = reshape_by_window(filtered)
    smoothed = kalman_filter_2d(reshaped)
    return smoothed


def build_multivariate_feature_matrix(matrices_dict):
    """
    Ghep nhieu kenh thanh 1 ma tran dac trung.
    Moi hang = [X_chan | Y_chan | h_chan] (noi tiep).
    """
    channels = list(matrices_dict.values())
    return np.hstack(channels)


# ============================================================================
#  2. CLUSTERING PIPELINE
# ============================================================================

def run_clustering(data_tsne, data_scaled, k):
    """Chay KMeans, HAC, GMM tren t-SNE space."""
    results = {}

    # KMeans
    km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
    labels = km.fit_predict(data_tsne)
    sil = silhouette_score(data_tsne, labels)
    cal = calinski_harabasz_score(data_tsne, labels)
    dav = davies_bouldin_score(data_tsne, labels)
    results['KMeans'] = {'labels': labels, 'silhouette': sil,
                          'calinski_harabasz': cal, 'davies_bouldin': dav, 'n_clusters': k}

    # HAC
    hac = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hac.fit_predict(data_tsne)
    sil = silhouette_score(data_tsne, labels)
    cal = calinski_harabasz_score(data_tsne, labels)
    dav = davies_bouldin_score(data_tsne, labels)
    results['HAC'] = {'labels': labels, 'silhouette': sil,
                       'calinski_harabasz': cal, 'davies_bouldin': dav, 'n_clusters': k}

    # GMM
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                           random_state=config.SEED, max_iter=config.GMM_MAX_ITER)
    gmm.fit(data_tsne)
    labels = gmm.predict(data_tsne)
    sil = silhouette_score(data_tsne, labels)
    cal = calinski_harabasz_score(data_tsne, labels)
    dav = davies_bouldin_score(data_tsne, labels)
    results['GMM'] = {'labels': labels, 'silhouette': sil,
                       'calinski_harabasz': cal, 'davies_bouldin': dav, 'n_clusters': k,
                       'aic': gmm.aic(data_tsne), 'bic': gmm.bic(data_tsne)}

    return results


def pp1_pipeline(feature_matrix, k):
    """Pipeline PP1: Scale -> PCA -> t-SNE -> Clustering."""
    # Scale
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(feature_matrix.T).T
    print(f"  Scale: {data_scaled.shape}")

    # PCA
    n_comp = min(50, data_scaled.shape[0], data_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=config.SEED)
    data_pca = pca.fit_transform(data_scaled)
    print(f"  PCA: {data_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # t-SNE
    perplexity = min(config.TSNE_PERPLEXITY, len(data_scaled) - 1)
    tsne = TSNE(n_components=2, init='pca', random_state=config.SEED,
                perplexity=perplexity, learning_rate=config.TSNE_LEARNING_RATE,
                early_exaggeration=config.TSNE_EARLY_EXAGGERATION, metric=config.TSNE_METRIC)
    data_tsne = tsne.fit_transform(data_scaled)
    data_tsne = tsne.fit_transform(data_tsne)  # 2 lan nhu notebook
    print(f"  t-SNE: {data_tsne.shape}")

    # Clustering
    results = run_clustering(data_tsne, data_scaled, k)
    return data_tsne, data_scaled, results


# ============================================================================
#  3. PHAN TICH TUONG QUAN NOI CUM
# ============================================================================

def intra_cluster_correlation(matrices_dict, labels, coord_names=None):
    """
    Tinh tuong quan noi cum giua cac kenh (X, Y, h).
    Tra ve dict {cluster_id: correlation_matrix}.
    """
    if coord_names is None:
        coord_names = list(matrices_dict.keys())

    n_channels = len(coord_names)
    unique_labels = np.unique(labels)
    corr_results = {}

    for cid in unique_labels:
        mask = labels == cid
        n_in_cluster = np.sum(mask)

        # Ma tran tuong quan trung binh giua cac kenh
        corr_matrix = np.zeros((n_channels, n_channels))
        p_matrix = np.zeros((n_channels, n_channels))

        for i, ci in enumerate(coord_names):
            for j, cj in enumerate(coord_names):
                # Tinh tuong quan trung binh giua 2 kenh cho cac mau trong cum
                corrs = []
                for idx in np.where(mask)[0]:
                    series_i = matrices_dict[ci][idx]
                    series_j = matrices_dict[cj][idx]
                    valid = ~(np.isnan(series_i) | np.isnan(series_j))
                    if np.sum(valid) > 10:
                        r, p = pearsonr(series_i[valid], series_j[valid])
                        corrs.append(r)
                if corrs:
                    corr_matrix[i, j] = np.mean(corrs)
                    p_matrix[i, j] = np.mean([abs(c) for c in corrs])

        corr_results[cid] = {
            'corr_matrix': corr_matrix,
            'n_samples': n_in_cluster,
            'coord_names': coord_names,
        }

    return corr_results


def cross_station_correlation(matrices_e, matrices_w, labels_e, labels_w,
                               coord_names=['X_Coord', 'Y_Coord', 'h_Coord']):
    """
    Tinh tuong quan giua 2 tram cho cac gio co cung thoi gian.
    Dung datetimes chung de match.
    """
    results = {}
    for coord in coord_names:
        # Tinh tuong quan trung binh giua 2 tram cho kenh nay
        n = min(len(matrices_e[coord]), len(matrices_w[coord]))
        corrs = []
        for idx in range(n):
            se = matrices_e[coord][idx]
            sw = matrices_w[coord][idx]
            valid = ~(np.isnan(se) | np.isnan(sw))
            if np.sum(valid) > 10:
                r, _ = pearsonr(se[valid], sw[valid])
                corrs.append(r)
        results[coord] = {
            'mean_corr': np.mean(corrs) if corrs else 0,
            'std_corr': np.std(corrs) if corrs else 0,
            'n_pairs': len(corrs),
            'corrs': corrs,
        }
    return results


def intra_cluster_cross_station_corr(matrices_e, matrices_w, labels_combined,
                                      n_e, coord_names=['X_Coord', 'Y_Coord', 'h_Coord']):
    """
    Phan tich tuong quan giua 2 tram TRONG TUNG CUM.
    labels_combined: nhan cum cua ca 2 tram ghep lai.
    n_e: so mau cua tram E (de tach labels).
    """
    labels_e = labels_combined[:n_e]
    labels_w = labels_combined[n_e:]
    unique_labels = np.unique(labels_combined)

    results = {}
    for cid in unique_labels:
        mask_e = labels_e == cid
        mask_w = labels_w == cid
        n_e_in = np.sum(mask_e)
        n_w_in = np.sum(mask_w)

        cluster_corr = {}
        for coord in coord_names:
            # Tinh mean cua tung tram trong cum
            if n_e_in > 0 and n_w_in > 0:
                mean_e = np.nanmean(matrices_e[coord][mask_e], axis=0)
                mean_w = np.nanmean(matrices_w[coord][mask_w], axis=0)
                valid = ~(np.isnan(mean_e) | np.isnan(mean_w))
                if np.sum(valid) > 10:
                    r, p = pearsonr(mean_e[valid], mean_w[valid])
                    cluster_corr[coord] = {'r': r, 'p': p}
                else:
                    cluster_corr[coord] = {'r': np.nan, 'p': np.nan}
            else:
                cluster_corr[coord] = {'r': np.nan, 'p': np.nan}

        results[cid] = {
            'n_e': n_e_in, 'n_w': n_w_in,
            'correlations': cluster_corr,
        }
    return results


# ============================================================================
#  4. VISUALIZATION
# ============================================================================

def plot_combined_scatter(data_tsne_e, data_tsne_w, labels_e, labels_w,
                          method_name, save=True):
    """Ve scatter plot 2 tram cung luc."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    k = len(np.unique(labels_e))

    # Station E
    ax = axes[0]
    for j in range(k):
        mask = labels_e == j
        ax.scatter(data_tsne_e[mask, 0], data_tsne_e[mask, 1],
                   c=COLORS[j % len(COLORS)], s=50, alpha=0.7, label=f'C{j}')
    ax.set_title(f'4E – {method_name}', fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Station W
    ax = axes[1]
    for j in range(k):
        mask = labels_w == j
        ax.scatter(data_tsne_w[mask, 0], data_tsne_w[mask, 1],
                   c=COLORS[j % len(COLORS)], s=50, alpha=0.7, label=f'C{j}')
    ax.set_title(f'4W – {method_name}', fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cluster size comparison
    ax = axes[2]
    x = np.arange(k)
    width = 0.35
    sizes_e = [np.sum(labels_e == j) for j in range(k)]
    sizes_w = [np.sum(labels_w == j) for j in range(k)]
    ax.bar(x - width/2, sizes_e, width, label='4E', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, sizes_w, width, label='4W', color='coral', alpha=0.7)
    ax.set_xlabel('Cum')
    ax.set_ylabel('So mau')
    ax.set_title('So sanh kich thuoc cum', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{j}' for j in range(k)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Phan cum da bien (X, Y, h) – {method_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, f'MV01_scatter_{method_name}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_joint_scatter(data_tsne_combined, labels_combined, n_e, method_name, save=True):
    """Ve scatter plot clustering chung 2 tram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    k = len(np.unique(labels_combined))
    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Color by cluster
    ax = axes[0]
    for j in range(k):
        mask = labels_combined == j
        ax.scatter(data_tsne_combined[mask, 0], data_tsne_combined[mask, 1],
                   c=COLORS[j % len(COLORS)], s=40, alpha=0.6, label=f'C{j}')
    ax.set_title(f'Phan cum chung – {method_name}', fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Color by station
    ax = axes[1]
    ax.scatter(data_tsne_combined[:n_e, 0], data_tsne_combined[:n_e, 1],
               c='steelblue', s=40, alpha=0.6, label='4E', marker='o')
    ax.scatter(data_tsne_combined[n_e:, 0], data_tsne_combined[n_e:, 1],
               c='coral', s=40, alpha=0.6, label='4W', marker='^')
    ax.set_title('Phan bo theo tram', fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Clustering chung 4E + 4W (da bien X, Y, h) – {method_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, f'MV02_joint_scatter_{method_name}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_intra_cluster_corr(corr_results, station_name, save=True):
    """Ve ma tran tuong quan noi cum cho 1 tram."""
    n_clusters = len(corr_results)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 4))
    if n_clusters == 1:
        axes = [axes]

    for i, (cid, res) in enumerate(corr_results.items()):
        ax = axes[i]
        corr = res['corr_matrix']
        names = [n.replace('_Coord', '') for n in res['coord_names']]

        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)

        for r in range(len(names)):
            for c in range(len(names)):
                val = corr[r, c]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)

        ax.set_title(f'Cum {cid} (n={res["n_samples"]})', fontweight='bold')

    fig.suptitle(f'{station_name} – Tuong quan noi cum giua X, Y, h',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        safe = station_name.replace(' ', '_')
        path = os.path.join(RD, f'MV03_intra_corr_{safe}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_cross_station_corr_by_cluster(cross_results, save=True):
    """Ve tuong quan giua 2 tram theo tung cum."""
    unique_labels = sorted(cross_results.keys())
    n_clusters = len(unique_labels)
    coord_names = ['X_Coord', 'Y_Coord', 'h_Coord']
    short_names = ['X', 'Y', 'h']

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 4))
    if n_clusters == 1:
        axes = [axes]

    COLORS = ['steelblue', 'coral', 'mediumseagreen']

    for i, cid in enumerate(unique_labels):
        ax = axes[i]
        res = cross_results[cid]
        corrs = res['correlations']
        vals = [corrs[c]['r'] for c in coord_names]
        bars = ax.bar(short_names, vals, color=COLORS, alpha=0.7, edgecolor='black')

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f'Cum {cid}\n(4E: {res["n_e"]}, 4W: {res["n_w"]})', fontweight='bold')
        ax.set_ylabel('Pearson r (4E vs 4W)')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Tuong quan giua 2 tram (4E vs 4W) trong tung cum',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, 'MV04_cross_station_corr.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_channel_timeseries_by_cluster(matrices, labels, station_name,
                                        coord_names=['X_Coord', 'Y_Coord', 'h_Coord'],
                                        save=True):
    """Ve chuoi thoi gian trung binh cua tung kenh theo cum."""
    unique_labels = sorted(np.unique(labels))
    n_clusters = len(unique_labels)
    n_channels = len(coord_names)

    fig, axes = plt.subplots(n_channels, n_clusters, figsize=(5 * n_clusters, 4 * n_channels))
    if n_clusters == 1:
        axes = axes.reshape(-1, 1)
    if n_channels == 1:
        axes = axes.reshape(1, -1)

    COLORS = ['red', 'blue', 'green', 'orange', 'purple']
    short_names = [n.replace('_Coord', '') for n in coord_names]

    for ci, coord in enumerate(coord_names):
        for cj, cid in enumerate(unique_labels):
            ax = axes[ci, cj]
            mask = labels == cid
            data = matrices[coord][mask]

            t = np.arange(data.shape[1])

            # Ve tung chuoi (mo nhat)
            for row in data:
                valid = ~np.isnan(row)
                if np.any(valid):
                    ax.plot(t[valid], row[valid], alpha=0.15, linewidth=0.3, color=COLORS[cj % len(COLORS)])

            # Ve mean + std
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            valid = ~np.isnan(mean)
            if np.any(valid):
                ax.plot(t[valid], mean[valid], color='black', linewidth=2, label='Mean')
                ax.fill_between(t[valid], (mean-std)[valid], (mean+std)[valid],
                                alpha=0.2, color='gray')

            ax.set_title(f'{short_names[ci]} – Cum {cid} (n={np.sum(mask)})',
                        fontsize=9, fontweight='bold')
            if ci == n_channels - 1:
                ax.set_xlabel('Time points')
            if cj == 0:
                ax.set_ylabel(short_names[ci])
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{station_name} – Chuoi thoi gian theo cum va kenh',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        safe = station_name.replace(' ', '_')
        path = os.path.join(RD, f'MV05_timeseries_{safe}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_metrics_comparison(results_e, results_w, results_combined, save=True):
    """So sanh metrics giua 3 cach chay."""
    methods = list(results_e.keys())
    configs = ['4E rieng', '4W rieng', '4E+4W chung']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric in enumerate(['silhouette', 'calinski_harabasz', 'davies_bouldin']):
        ax = axes[idx]
        x = np.arange(len(methods))
        width = 0.25

        vals_e = [results_e[m][metric] for m in methods]
        vals_w = [results_w[m][metric] for m in methods]
        vals_c = [results_combined[m][metric] for m in methods]

        ax.bar(x - width, vals_e, width, label='4E', color='steelblue', alpha=0.7)
        ax.bar(x, vals_w, width, label='4W', color='coral', alpha=0.7)
        ax.bar(x + width, vals_c, width, label='4E+4W', color='mediumseagreen', alpha=0.7)

        title_map = {
            'silhouette': 'Silhouette (cao = tot)',
            'calinski_harabasz': 'Calinski-Harabasz (cao = tot)',
            'davies_bouldin': 'Davies-Bouldin (thap = tot)',
        }
        ax.set_title(title_map[metric], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('So sanh metrics: 4E rieng vs 4W rieng vs 4E+4W chung',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, 'MV06_metrics_comparison.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


# ============================================================================
#  MAIN
# ============================================================================

def main():
    k = args.k
    print("=" * 70)
    print(f"PHAN TICH DA BIEN (X, Y, h) – 2 TRAM 4E + 4W (k={k})")
    print("=" * 70)

    # ── 1. Tai du lieu 2 tram ──────────────────────────────────────────────
    print("\n[1/6] Tai du lieu...")
    df_e = load_station_data(os.path.join(config.DATA_DIR, 'full_gnss_2e.csv'))
    df_w = load_station_data(os.path.join(config.DATA_DIR, 'full_gnss_2w.csv'))
    print(f"  4E: {len(df_e):,} dong, {df_e['Date'].nunique()} ngay")
    print(f"  4W: {len(df_w):,} dong, {df_w['Date'].nunique()} ngay")

    # ── 2. Tao ma tran da bien theo gio ───────────────────────────────────
    print("\n[2/6] Tao ma tran da bien theo gio...")
    coord_cols = ['X_Coord', 'Y_Coord', 'h_Coord']

    matrices_e, info_e = create_multivariate_hourly_matrix(df_e, coord_cols)
    print(f"  4E: {len(info_e)} gio hop le, shape={matrices_e['h_Coord'].shape}")

    matrices_w, info_w = create_multivariate_hourly_matrix(df_w, coord_cols)
    print(f"  4W: {len(info_w)} gio hop le, shape={matrices_w['h_Coord'].shape}")

    # ── 3. Tien xu ly tung kenh ──────────────────────────────────────────
    print("\n[3/6] Tien xu ly...")
    proc_e = {}
    proc_w = {}
    for col in coord_cols:
        print(f"  Processing {col}...")
        proc_e[col] = preprocess_channel(matrices_e[col])
        proc_w[col] = preprocess_channel(matrices_w[col])

    print(f"  Processed shape: {proc_e['h_Coord'].shape}")

    # ── 4. Clustering rieng tung tram ────────────────────────────────────
    print(f"\n[4/6] Clustering rieng tung tram (k={k})...")

    # 4E
    print("\n  --- 4E ---")
    feat_e = build_multivariate_feature_matrix(proc_e)
    print(f"  Feature matrix 4E: {feat_e.shape}")
    tsne_e, scaled_e, results_e = pp1_pipeline(feat_e, k)

    for m, r in results_e.items():
        print(f"    {m}: Sil={r['silhouette']:.4f}")

    # 4W
    print("\n  --- 4W ---")
    feat_w = build_multivariate_feature_matrix(proc_w)
    print(f"  Feature matrix 4W: {feat_w.shape}")
    tsne_w, scaled_w, results_w = pp1_pipeline(feat_w, k)

    for m, r in results_w.items():
        print(f"    {m}: Sil={r['silhouette']:.4f}")

    # ── 5. Clustering chung 2 tram ───────────────────────────────────────
    print(f"\n[5/6] Clustering chung 4E + 4W (k={k})...")
    n_e = len(feat_e)

    # Ghep 2 tram
    feat_combined = np.vstack([feat_e, feat_w])
    print(f"  Combined feature matrix: {feat_combined.shape}")
    tsne_combined, scaled_combined, results_combined = pp1_pipeline(feat_combined, k)

    for m, r in results_combined.items():
        print(f"    {m}: Sil={r['silhouette']:.4f}")

    # Chon thuat toan tot nhat
    best_method = max(results_combined.keys(),
                      key=lambda m: results_combined[m]['silhouette'])
    best_labels = results_combined[best_method]['labels']
    print(f"\n  Best method (combined): {best_method}")

    # ── 6. Phan tich tuong quan noi cum ──────────────────────────────────
    print("\n[6/6] Phan tich tuong quan noi cum...")

    # Tuong quan giua X, Y, h trong tung cum – moi tram
    best_e = max(results_e.keys(), key=lambda m: results_e[m]['silhouette'])
    best_w = max(results_w.keys(), key=lambda m: results_w[m]['silhouette'])

    corr_e = intra_cluster_correlation(proc_e, results_e[best_e]['labels'], coord_cols)
    corr_w = intra_cluster_correlation(proc_w, results_w[best_w]['labels'], coord_cols)

    print("\n  Tuong quan noi cum 4E:")
    for cid, res in corr_e.items():
        names = [n.replace('_Coord', '') for n in res['coord_names']]
        print(f"    Cum {cid} (n={res['n_samples']}):")
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j > i:
                    print(f"      {ni}-{nj}: r={res['corr_matrix'][i,j]:.3f}")

    print("\n  Tuong quan noi cum 4W:")
    for cid, res in corr_w.items():
        names = [n.replace('_Coord', '') for n in res['coord_names']]
        print(f"    Cum {cid} (n={res['n_samples']}):")
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j > i:
                    print(f"      {ni}-{nj}: r={res['corr_matrix'][i,j]:.3f}")

    # Tuong quan giua 2 tram trong tung cum
    cross_corr = intra_cluster_cross_station_corr(
        proc_e, proc_w, best_labels, n_e, coord_cols)

    print("\n  Tuong quan 4E vs 4W theo cum:")
    for cid, res in cross_corr.items():
        print(f"    Cum {cid} (4E:{res['n_e']}, 4W:{res['n_w']}):")
        for coord, cr in res['correlations'].items():
            short = coord.replace('_Coord', '')
            print(f"      {short}: r={cr['r']:.3f}, p={cr['p']:.4f}")

    # ── Visualization ──────────────────────────────────────────────────
    print("\n  Ve bieu do...")

    # Scatter plots rieng
    plot_combined_scatter(tsne_e, tsne_w, results_e[best_e]['labels'],
                          results_w[best_w]['labels'], best_e)

    # Joint scatter
    plot_joint_scatter(tsne_combined, best_labels, n_e, best_method)

    # Intra-cluster correlation
    plot_intra_cluster_corr(corr_e, '4E')
    plot_intra_cluster_corr(corr_w, '4W')

    # Cross-station correlation by cluster
    plot_cross_station_corr_by_cluster(cross_corr)

    # Timeseries by cluster
    plot_channel_timeseries_by_cluster(proc_e, results_e[best_e]['labels'], '4E', coord_cols)
    plot_channel_timeseries_by_cluster(proc_w, results_w[best_w]['labels'], '4W', coord_cols)

    # Metrics comparison
    plot_metrics_comparison(results_e, results_w, results_combined)

    # ── Tom tat ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOM TAT KET QUA")
    print("=" * 70)

    print(f"\nSo cum: k={k}")
    print(f"\n4E ({len(info_e)} gio, {best_e}):")
    for m, r in results_e.items():
        print(f"  {m}: Sil={r['silhouette']:.4f}, Cal={r['calinski_harabasz']:.1f}, "
              f"Dav={r['davies_bouldin']:.4f}")

    print(f"\n4W ({len(info_w)} gio, {best_w}):")
    for m, r in results_w.items():
        print(f"  {m}: Sil={r['silhouette']:.4f}, Cal={r['calinski_harabasz']:.1f}, "
              f"Dav={r['davies_bouldin']:.4f}")

    print(f"\n4E+4W chung ({n_e}+{len(feat_w)}={len(feat_combined)} gio, {best_method}):")
    for m, r in results_combined.items():
        print(f"  {m}: Sil={r['silhouette']:.4f}, Cal={r['calinski_harabasz']:.1f}, "
              f"Dav={r['davies_bouldin']:.4f}")

    print(f"\nTat ca hinh anh da luu vao: {RD}")
    print("HOAN THANH!")


if __name__ == '__main__':
    main()
