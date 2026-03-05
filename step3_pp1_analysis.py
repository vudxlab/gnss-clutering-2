"""
BUOC 3 – Phan tich do nhay tham so & do on dinh cho PP1 (Raw t-SNE)
====================================================================

a) Phan tich do nhay tham so (Parameter Sensitivity):
   - KMeans, HAC, GMM: quet nhieu gia tri k
   - GMM: kiem tra cac dang ma tran hiep phuong sai (full, tied, diag, spherical)
   - DBSCAN: quet nhieu gia tri eps va MinPts

b) Phan tich do on dinh (Stability Analysis):
   - Moi thuat toan chay lap nhieu lan voi khoi tao / cau hinh khac nhau
   - Tinh ARI giua cac lan chay de danh gia do nhat quan

Chay:
    python step3_pp1_analysis.py --no-display
    python step3_pp1_analysis.py --no-cache --no-display
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
parser = argparse.ArgumentParser(description='B3 – Phan tich do nhay & on dinh PP1')
parser.add_argument('--no-display', action='store_true', help='Khong hien thi cua so (Agg backend)')
parser.add_argument('--no-cache', action='store_true', help='Bo qua cache, tai lai du lieu tu CSV')
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
#  Import package
# ---------------------------------------------------------------------------
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from gnss_clustering import config
from gnss_clustering.data_loader import load_data, create_daily_matrix, create_hourly_matrix, load_cached_matrices
from gnss_clustering.preprocessing import preprocess_pipeline
from gnss_clustering.feature_extraction import extract_features

plt.style.use(config.MATPLOTLIB_STYLE)

RD = config.RESULT_DIR
RD_PP1 = os.path.join(RD, config.RESULT_SUBDIR_PP1)
os.makedirs(RD_PP1, exist_ok=True)


# ============================================================================
#  A. PHAN TICH DO NHAY THAM SO
# ============================================================================

def sensitivity_k_sweep(data_tsne, k_range=range(2, 11)):
    """
    Quet nhieu gia tri k cho KMeans, HAC, GMM.
    Tra ve DataFrame voi cac metrics cho moi (algorithm, k).
    """
    print("\n" + "=" * 60)
    print("A1. SENSITIVITY – K SWEEP (KMeans, HAC, GMM)")
    print("=" * 60)

    records = []
    for k in k_range:
        # KMeans
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
        labels_km = km.fit_predict(data_tsne)
        sil = silhouette_score(data_tsne, labels_km)
        cal = calinski_harabasz_score(data_tsne, labels_km)
        dav = davies_bouldin_score(data_tsne, labels_km)
        records.append({'algorithm': 'KMeans', 'k': k,
                        'silhouette': sil, 'calinski_harabasz': cal, 'davies_bouldin': dav})

        # HAC
        hac = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_hac = hac.fit_predict(data_tsne)
        sil = silhouette_score(data_tsne, labels_hac)
        cal = calinski_harabasz_score(data_tsne, labels_hac)
        dav = davies_bouldin_score(data_tsne, labels_hac)
        records.append({'algorithm': 'HAC', 'k': k,
                        'silhouette': sil, 'calinski_harabasz': cal, 'davies_bouldin': dav})

        # GMM (full covariance)
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               random_state=config.SEED, max_iter=config.GMM_MAX_ITER)
        gmm.fit(data_tsne)
        labels_gmm = gmm.predict(data_tsne)
        sil = silhouette_score(data_tsne, labels_gmm)
        cal = calinski_harabasz_score(data_tsne, labels_gmm)
        dav = davies_bouldin_score(data_tsne, labels_gmm)
        records.append({'algorithm': 'GMM', 'k': k,
                        'silhouette': sil, 'calinski_harabasz': cal, 'davies_bouldin': dav,
                        'aic': gmm.aic(data_tsne), 'bic': gmm.bic(data_tsne)})

        print(f"  k={k}: KMeans Sil={records[-3]['silhouette']:.3f}, "
              f"HAC Sil={records[-2]['silhouette']:.3f}, "
              f"GMM Sil={records[-1]['silhouette']:.3f}")

    df = pd.DataFrame(records)
    return df


def sensitivity_gmm_covariance(data_tsne, k_range=range(2, 11)):
    """
    Quet cac dang ma tran hiep phuong sai cua GMM: full, tied, diag, spherical.
    """
    print("\n" + "=" * 60)
    print("A2. SENSITIVITY – GMM COVARIANCE TYPES")
    print("=" * 60)

    cov_types = ['full', 'tied', 'diag', 'spherical']
    records = []

    for cov_type in cov_types:
        for k in k_range:
            try:
                gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                                       random_state=config.SEED, max_iter=config.GMM_MAX_ITER)
                gmm.fit(data_tsne)
                labels = gmm.predict(data_tsne)
                sil = silhouette_score(data_tsne, labels)
                cal = calinski_harabasz_score(data_tsne, labels)
                dav = davies_bouldin_score(data_tsne, labels)
                records.append({
                    'covariance_type': cov_type, 'k': k,
                    'silhouette': sil, 'calinski_harabasz': cal, 'davies_bouldin': dav,
                    'aic': gmm.aic(data_tsne), 'bic': gmm.bic(data_tsne),
                    'log_likelihood': gmm.score(data_tsne),
                })
            except Exception as e:
                print(f"  GMM({cov_type}, k={k}) failed: {e}")
                continue

        print(f"  {cov_type}: {len([r for r in records if r['covariance_type']==cov_type])} configs OK")

    df = pd.DataFrame(records)
    return df


def sensitivity_dbscan(data_tsne, min_pts_range=range(2, 10), n_eps=20):
    """
    Quet nhieu gia tri eps va MinPts cho DBSCAN.
    eps duoc chon tu phan vi k-NN distances.
    """
    print("\n" + "=" * 60)
    print("A3. SENSITIVITY – DBSCAN (eps x MinPts)")
    print("=" * 60)

    records = []

    for min_pts in min_pts_range:
        # Tinh khoang cach k-NN de xac dinh dai eps hop ly
        nbrs = NearestNeighbors(n_neighbors=min_pts).fit(data_tsne)
        distances, _ = nbrs.kneighbors(data_tsne)
        k_distances = np.sort(distances[:, min_pts - 1])

        # Quet eps tu percentile 50 den 99
        eps_values = np.percentile(k_distances, np.linspace(50, 99, n_eps))
        eps_values = np.unique(np.round(eps_values, 6))

        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=min_pts)
            labels = db.fit_predict(data_tsne)

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels != -1])
            n_noise = int(np.sum(labels == -1))

            if n_clusters >= 2:
                mask = labels != -1
                if np.sum(mask) > n_clusters:
                    sil = silhouette_score(data_tsne[mask], labels[mask])
                else:
                    sil = -1
            else:
                sil = -1

            records.append({
                'min_pts': min_pts, 'eps': eps,
                'n_clusters': n_clusters, 'n_noise': n_noise,
                'noise_ratio': n_noise / len(labels),
                'silhouette': sil,
            })

        n_valid = len([r for r in records if r['min_pts'] == min_pts and r['silhouette'] > -1])
        print(f"  MinPts={min_pts}: {n_valid} configs co >= 2 cum")

    df = pd.DataFrame(records)
    return df


# ============================================================================
#  B. PHAN TICH DO ON DINH
# ============================================================================

def stability_multi_init(data_tsne, k, n_runs=30):
    """
    Chay moi thuat toan nhieu lan voi khoi tao khac nhau,
    tinh ARI giua tat ca cac cap ket qua de do do on dinh.
    """
    print("\n" + "=" * 60)
    print(f"B. STABILITY ANALYSIS (k={k}, {n_runs} runs)")
    print("=" * 60)

    results = {}

    # --- KMeans ---
    print("  KMeans...")
    km_labels_list = []
    km_sil_list = []
    for i in range(n_runs):
        km = KMeans(n_clusters=k, random_state=i, n_init=10)
        labels = km.fit_predict(data_tsne)
        km_labels_list.append(labels)
        km_sil_list.append(silhouette_score(data_tsne, labels))

    km_ari = _pairwise_ari(km_labels_list)
    results['KMeans'] = {
        'labels_list': km_labels_list, 'sil_list': km_sil_list,
        'ari_scores': km_ari,
        'ari_mean': float(np.mean(km_ari)), 'ari_std': float(np.std(km_ari)),
        'sil_mean': float(np.mean(km_sil_list)), 'sil_std': float(np.std(km_sil_list)),
    }
    print(f"    ARI = {results['KMeans']['ari_mean']:.3f} +/- {results['KMeans']['ari_std']:.3f}")

    # --- HAC (deterministic, sweep linkages) ---
    print("  HAC (4 linkage variants)...")
    linkages = ['ward', 'complete', 'average', 'single']
    hac_labels_list = []
    hac_sil_list = []
    hac_linkage_names = []
    for linkage in linkages:
        hac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = hac.fit_predict(data_tsne)
        hac_labels_list.append(labels)
        hac_sil_list.append(silhouette_score(data_tsne, labels))
        hac_linkage_names.append(linkage)

    hac_ari = _pairwise_ari(hac_labels_list)
    results['HAC'] = {
        'labels_list': hac_labels_list, 'sil_list': hac_sil_list,
        'linkage_names': hac_linkage_names,
        'ari_scores': hac_ari,
        'ari_mean': float(np.mean(hac_ari)), 'ari_std': float(np.std(hac_ari)),
        'sil_mean': float(np.mean(hac_sil_list)), 'sil_std': float(np.std(hac_sil_list)),
    }
    print(f"    ARI = {results['HAC']['ari_mean']:.3f} +/- {results['HAC']['ari_std']:.3f}")

    # --- GMM ---
    print("  GMM...")
    gmm_labels_list = []
    gmm_sil_list = []
    for i in range(n_runs):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               random_state=i, max_iter=config.GMM_MAX_ITER)
        gmm.fit(data_tsne)
        labels = gmm.predict(data_tsne)
        gmm_labels_list.append(labels)
        gmm_sil_list.append(silhouette_score(data_tsne, labels))

    gmm_ari = _pairwise_ari(gmm_labels_list)
    results['GMM'] = {
        'labels_list': gmm_labels_list, 'sil_list': gmm_sil_list,
        'ari_scores': gmm_ari,
        'ari_mean': float(np.mean(gmm_ari)), 'ari_std': float(np.std(gmm_ari)),
        'sil_mean': float(np.mean(gmm_sil_list)), 'sil_std': float(np.std(gmm_sil_list)),
    }
    print(f"    ARI = {results['GMM']['ari_mean']:.3f} +/- {results['GMM']['ari_std']:.3f}")

    # --- DBSCAN (sweep eps quanh gia tri tu dong) ---
    print("  DBSCAN (sweep eps)...")
    min_pts = config.DBSCAN_MIN_SAMPLES
    nbrs = NearestNeighbors(n_neighbors=min_pts).fit(data_tsne)
    distances, _ = nbrs.kneighbors(data_tsne)
    k_distances = np.sort(distances[:, min_pts - 1])

    # Lay 15 gia tri eps quanh percentile 70-95
    eps_values = np.percentile(k_distances, np.linspace(70, 95, 15))
    eps_values = np.unique(np.round(eps_values, 6))

    dbscan_labels_list = []
    dbscan_sil_list = []
    dbscan_eps_list = []
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_pts)
        labels = db.fit_predict(data_tsne)
        n_cls = len(np.unique(labels[labels != -1]))
        if n_cls >= 2:
            mask = labels != -1
            if np.sum(mask) > n_cls:
                dbscan_labels_list.append(labels)
                dbscan_sil_list.append(silhouette_score(data_tsne[mask], labels[mask]))
                dbscan_eps_list.append(eps)

    if len(dbscan_labels_list) >= 2:
        dbscan_ari = _pairwise_ari(dbscan_labels_list)
    else:
        dbscan_ari = []

    results['DBSCAN'] = {
        'labels_list': dbscan_labels_list, 'sil_list': dbscan_sil_list,
        'eps_list': dbscan_eps_list,
        'ari_scores': dbscan_ari,
        'ari_mean': float(np.mean(dbscan_ari)) if dbscan_ari else 0.0,
        'ari_std': float(np.std(dbscan_ari)) if dbscan_ari else 0.0,
        'sil_mean': float(np.mean(dbscan_sil_list)) if dbscan_sil_list else 0.0,
        'sil_std': float(np.std(dbscan_sil_list)) if dbscan_sil_list else 0.0,
    }
    print(f"    ARI = {results['DBSCAN']['ari_mean']:.3f} +/- {results['DBSCAN']['ari_std']:.3f} "
          f"({len(dbscan_labels_list)} valid configs)")

    return results


def _pairwise_ari(labels_list):
    """Tinh ARI giua tat ca cac cap ket qua."""
    n = len(labels_list)
    ari_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            ari_scores.append(ari)
    return ari_scores


# ============================================================================
#  VISUALIZATION
# ============================================================================

def plot_k_sensitivity(df_k, save=True, result_dir=None):
    """Ve bieu do do nhay theo k cho 3 thuat toan."""
    if result_dir is None:
        result_dir = RD_PP1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ('silhouette', 'Silhouette Score (cao = tot)', True),
        ('calinski_harabasz', 'Calinski-Harabasz Score (cao = tot)', True),
        ('davies_bouldin', 'Davies-Bouldin Score (thap = tot)', False),
    ]

    for ax, (metric, title, higher_better) in zip(axes, metrics):
        for algo in ['KMeans', 'HAC', 'GMM']:
            sub = df_k[df_k['algorithm'] == algo]
            ax.plot(sub['k'], sub[metric], 'o-', label=algo, linewidth=2, markersize=6)

        ax.set_xlabel('So cum k', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df_k['k'].unique()))

        # Danh dau k tot nhat
        for algo in ['KMeans', 'HAC', 'GMM']:
            sub = df_k[df_k['algorithm'] == algo]
            if higher_better:
                best_idx = sub[metric].idxmax()
            else:
                best_idx = sub[metric].idxmin()
            best_k = sub.loc[best_idx, 'k']
            best_val = sub.loc[best_idx, metric]
            ax.annotate(f'k={best_k}', xy=(best_k, best_val),
                       fontsize=8, fontweight='bold', color='red',
                       textcoords="offset points", xytext=(5, 10),
                       arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    fig.suptitle('PP1 – Do nhay tham so k (KMeans, HAC, GMM)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'SA01_k_sensitivity.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_gmm_covariance(df_gmm_cov, save=True, result_dir=None):
    """Ve bieu do so sanh cac dang covariance cua GMM."""
    if result_dir is None:
        result_dir = RD_PP1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cov_types = df_gmm_cov['covariance_type'].unique()
    colors = {'full': 'steelblue', 'tied': 'coral', 'diag': 'mediumseagreen', 'spherical': 'orchid'}

    # Silhouette vs k
    ax = axes[0, 0]
    for cov in cov_types:
        sub = df_gmm_cov[df_gmm_cov['covariance_type'] == cov]
        ax.plot(sub['k'], sub['silhouette'], 'o-', label=cov,
                color=colors.get(cov, 'gray'), linewidth=2, markersize=6)
    ax.set_xlabel('So cum k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette theo k va covariance type', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BIC vs k
    ax = axes[0, 1]
    for cov in cov_types:
        sub = df_gmm_cov[df_gmm_cov['covariance_type'] == cov]
        ax.plot(sub['k'], sub['bic'], 'o-', label=cov,
                color=colors.get(cov, 'gray'), linewidth=2, markersize=6)
    ax.set_xlabel('So cum k')
    ax.set_ylabel('BIC (thap = tot)')
    ax.set_title('BIC theo k va covariance type', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AIC vs k
    ax = axes[1, 0]
    for cov in cov_types:
        sub = df_gmm_cov[df_gmm_cov['covariance_type'] == cov]
        ax.plot(sub['k'], sub['aic'], 'o-', label=cov,
                color=colors.get(cov, 'gray'), linewidth=2, markersize=6)
    ax.set_xlabel('So cum k')
    ax.set_ylabel('AIC (thap = tot)')
    ax.set_title('AIC theo k va covariance type', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log-likelihood vs k
    ax = axes[1, 1]
    for cov in cov_types:
        sub = df_gmm_cov[df_gmm_cov['covariance_type'] == cov]
        ax.plot(sub['k'], sub['log_likelihood'], 'o-', label=cov,
                color=colors.get(cov, 'gray'), linewidth=2, markersize=6)
    ax.set_xlabel('So cum k')
    ax.set_ylabel('Log-likelihood (cao = tot)')
    ax.set_title('Log-likelihood theo k va covariance type', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('PP1 – GMM: Anh huong cua dang ma tran hiep phuong sai',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'SA02_gmm_covariance.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_dbscan_sensitivity(df_dbscan, save=True, result_dir=None):
    """Ve bieu do do nhay DBSCAN (eps x MinPts)."""
    if result_dir is None:
        result_dir = RD_PP1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    min_pts_values = sorted(df_dbscan['min_pts'].unique())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(min_pts_values)))

    # So cum vs eps
    ax = axes[0]
    for i, mp in enumerate(min_pts_values):
        sub = df_dbscan[df_dbscan['min_pts'] == mp]
        ax.plot(sub['eps'], sub['n_clusters'], 'o-', label=f'MinPts={mp}',
                color=cmap[i], markersize=3, linewidth=1)
    ax.set_xlabel('eps')
    ax.set_ylabel('So cum')
    ax.set_title('So cum theo eps va MinPts', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Noise ratio vs eps
    ax = axes[1]
    for i, mp in enumerate(min_pts_values):
        sub = df_dbscan[df_dbscan['min_pts'] == mp]
        ax.plot(sub['eps'], sub['noise_ratio'] * 100, 'o-', label=f'MinPts={mp}',
                color=cmap[i], markersize=3, linewidth=1)
    ax.set_xlabel('eps')
    ax.set_ylabel('Ty le noise (%)')
    ax.set_title('Ty le noise theo eps va MinPts', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Silhouette vs eps (chi configs co >= 2 cum)
    ax = axes[2]
    df_valid = df_dbscan[df_dbscan['silhouette'] > -1]
    for i, mp in enumerate(min_pts_values):
        sub = df_valid[df_valid['min_pts'] == mp]
        if len(sub) > 0:
            ax.plot(sub['eps'], sub['silhouette'], 'o-', label=f'MinPts={mp}',
                    color=cmap[i], markersize=3, linewidth=1)
    ax.set_xlabel('eps')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette theo eps (chi configs >= 2 cum)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('PP1 – DBSCAN: Do nhay tham so eps va MinPts',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'SA03_dbscan_sensitivity.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_stability_results(stability_results, save=True, result_dir=None):
    """Ve bieu do do on dinh cua cac thuat toan."""
    if result_dir is None:
        result_dir = RD_PP1

    methods = list(stability_results.keys())
    n_methods = len(methods)

    # Figure 1: Boxplot + Histogram ARI
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
    COLORS = ['steelblue', 'coral', 'mediumseagreen', 'orchid']

    for i, method in enumerate(methods):
        ax = axes[i]
        ari_scores = stability_results[method]['ari_scores']
        if ari_scores:
            ax.hist(ari_scores, bins=min(20, max(5, len(ari_scores) // 3)),
                    color=COLORS[i % len(COLORS)], alpha=0.7, edgecolor='black', linewidth=0.5)
            mean_val = stability_results[method]['ari_mean']
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                       label=f'Mean={mean_val:.3f}')
            ax.axvspan(0.75, 1.0, alpha=0.1, color='green')
            ax.axvspan(0.0, 0.5, alpha=0.1, color='red')
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Pairwise ARI')
        ax.set_ylabel('Tan suat')
        ax.set_xlim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Boxplot tong hop
    ax_box = axes[-1]
    box_data = [stability_results[m]['ari_scores'] for m in methods
                if stability_results[m]['ari_scores']]
    box_labels = [m for m in methods if stability_results[m]['ari_scores']]
    if box_data:
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
        ax_box.axhline(0.75, color='green', linestyle='--', alpha=0.5, label='On dinh tot (0.75)')
        ax_box.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Tuong doi (0.50)')
    ax_box.set_title('So sanh do on dinh', fontsize=12, fontweight='bold')
    ax_box.set_ylabel('Pairwise ARI')
    ax_box.set_ylim(-0.1, 1.1)
    ax_box.legend(fontsize=8)
    ax_box.grid(True, alpha=0.3)

    fig.suptitle('PP1 – Do on dinh cua cac thuat toan\n'
                 '(Pairwise ARI giua nhieu lan chay voi khoi tao/cau hinh khac nhau)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'SA04_stability_ari.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()

    # Figure 2: Silhouette variation
    fig2, axes2 = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes2 = [axes2]

    for i, method in enumerate(methods):
        ax = axes2[i]
        sil_list = stability_results[method]['sil_list']
        if sil_list:
            x = range(len(sil_list))
            ax.bar(x, sil_list, color=COLORS[i % len(COLORS)], alpha=0.7)
            ax.axhline(np.mean(sil_list), color='red', linestyle='--', linewidth=2,
                       label=f'Mean={np.mean(sil_list):.3f}')
            ax.fill_between(range(len(sil_list)),
                           np.mean(sil_list) - np.std(sil_list),
                           np.mean(sil_list) + np.std(sil_list),
                           alpha=0.15, color='red', label=f'Std={np.std(sil_list):.3f}')
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Lan chay / Config')
        ax.set_ylabel('Silhouette Score')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle('PP1 – Bien dong Silhouette qua nhieu lan chay',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'SA05_stability_silhouette.png')
        fig2.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def print_sensitivity_summary(df_k, df_gmm_cov, df_dbscan):
    """In bang tom tat ket qua do nhay."""
    print("\n" + "=" * 70)
    print("TOM TAT – DO NHAY THAM SO PP1")
    print("=" * 70)

    # K sweep - best k per algorithm
    print("\n--- K tot nhat (theo Silhouette) ---")
    for algo in ['KMeans', 'HAC', 'GMM']:
        sub = df_k[df_k['algorithm'] == algo]
        best = sub.loc[sub['silhouette'].idxmax()]
        print(f"  {algo:<10}: k={int(best['k'])} (Sil={best['silhouette']:.4f}, "
              f"Cal={best['calinski_harabasz']:.1f}, Dav={best['davies_bouldin']:.4f})")

    # GMM covariance
    print("\n--- GMM covariance tot nhat (theo BIC) ---")
    for cov in df_gmm_cov['covariance_type'].unique():
        sub = df_gmm_cov[df_gmm_cov['covariance_type'] == cov]
        best = sub.loc[sub['bic'].idxmin()]
        print(f"  {cov:<12}: k={int(best['k'])} (BIC={best['bic']:.1f}, "
              f"Sil={best['silhouette']:.4f})")

    # DBSCAN
    print("\n--- DBSCAN tot nhat (theo Silhouette, >= 2 cum) ---")
    df_valid = df_dbscan[df_dbscan['silhouette'] > -1]
    if len(df_valid) > 0:
        best = df_valid.loc[df_valid['silhouette'].idxmax()]
        print(f"  MinPts={int(best['min_pts'])}, eps={best['eps']:.4f}: "
              f"k={int(best['n_clusters'])}, noise={best['noise_ratio']*100:.1f}%, "
              f"Sil={best['silhouette']:.4f}")
    else:
        print("  Khong co cau hinh DBSCAN nao cho >= 2 cum")


def print_stability_summary(stability_results):
    """In bang tom tat ket qua do on dinh."""
    print("\n" + "=" * 70)
    print("TOM TAT – DO ON DINH PP1")
    print("=" * 70)
    hdr = f"{'Thuat toan':<12} {'ARI mean':>10} {'ARI std':>10} {'Sil mean':>10} {'Sil std':>10} {'On dinh?':>10}"
    print(hdr)
    print("-" * len(hdr))
    for method, res in stability_results.items():
        stable = "Co" if res['ari_mean'] >= 0.75 else ("TB" if res['ari_mean'] >= 0.5 else "Khong")
        print(f"{method:<12} {res['ari_mean']:>10.3f} {res['ari_std']:>10.3f} "
              f"{res['sil_mean']:>10.3f} {res['sil_std']:>10.3f} {stable:>10}")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHAN TICH DO NHAY THAM SO & DO ON DINH – PP1 (Raw t-SNE)")
    print("=" * 70)

    # ── 1. Tai du lieu ──────────────────────────────────────────────────────
    print("\n[1/5] Tai du lieu...")
    cache_ok = (
        not args.no_cache
        and os.path.exists(config.HOURLY_MATRIX_FILE)
        and os.path.exists(config.HOURLY_INFO_FILE)
    )

    if cache_ok:
        print("  Cache ton tai – dang tai...")
        _, _, hourly_matrix, valid_hours_info = load_cached_matrices()
    else:
        print("  Khong co cache – tai tu CSV...")
        df = load_data()
        daily_matrix, unique_dates = create_daily_matrix(df, save=True)
        hourly_matrix, valid_hours_info = create_hourly_matrix(
            daily_matrix, unique_dates, save=True)

    print(f"  hourly_matrix: {hourly_matrix.shape}")

    # ── 2. Tien xu ly ──────────────────────────────────────────────────────
    print("\n[2/5] Tien xu ly (Hampel -> reshape -> Kalman)...")
    data_filtered, _, _ = preprocess_pipeline(hourly_matrix)

    # ── 3. Trich xuat dac trung (PP1 pipeline) ────────────────────────────
    print("\n[3/5] Trich xuat dac trung (Scale -> PCA -> t-SNE)...")
    data_tsne, data_scaled, _ = extract_features(data_filtered)
    print(f"  data_tsne shape: {data_tsne.shape}")

    # ── 4. Phan tich do nhay tham so ──────────────────────────────────────
    print("\n[4/5] Phan tich do nhay tham so...")

    k_range = range(2, 11)
    df_k = sensitivity_k_sweep(data_tsne, k_range=k_range)
    df_gmm_cov = sensitivity_gmm_covariance(data_tsne, k_range=k_range)
    df_dbscan = sensitivity_dbscan(data_tsne, min_pts_range=range(2, 10), n_eps=20)

    # Ve bieu do
    plot_k_sensitivity(df_k, save=True, result_dir=RD_PP1)
    plot_gmm_covariance(df_gmm_cov, save=True, result_dir=RD_PP1)
    plot_dbscan_sensitivity(df_dbscan, save=True, result_dir=RD_PP1)

    print_sensitivity_summary(df_k, df_gmm_cov, df_dbscan)

    # ── 5. Phan tich do on dinh ──────────────────────────────────────────
    print("\n[5/5] Phan tich do on dinh...")

    # Chon k tot nhat theo silhouette trung binh cua 3 thuat toan
    k_sil_avg = df_k.groupby('k')['silhouette'].mean()
    best_k = int(k_sil_avg.idxmax())
    print(f"  Chon k={best_k} (silhouette trung binh cao nhat) cho stability analysis")

    stability_results = stability_multi_init(data_tsne, k=best_k, n_runs=30)
    plot_stability_results(stability_results, save=True, result_dir=RD_PP1)
    print_stability_summary(stability_results)

    # Luu ket qua CSV
    df_k.to_csv(os.path.join(RD_PP1, 'SA_k_sweep.csv'), index=False)
    df_gmm_cov.to_csv(os.path.join(RD_PP1, 'SA_gmm_covariance.csv'), index=False)
    df_dbscan.to_csv(os.path.join(RD_PP1, 'SA_dbscan_sensitivity.csv'), index=False)
    print(f"\n  Ket qua CSV da luu vao: {RD_PP1}")

    print("\n" + "=" * 70)
    print("HOAN THANH!")
    print("=" * 70)


if __name__ == '__main__':
    main()
