"""
BUOC 5 – Phan tich PP1 hoan chinh cho 2 tram 4E va 4W
======================================================

Ket hop 4 buoc:
  1. Thong ke chi tiet 2 bo du lieu (data info)
  2. Tim k toi uu (step1 – find_optimal_clusters)
  3. Phan cum chi tiet + visualization (step2 – run_all)
  4. Phan tich do nhay tham so + do on dinh (step3 – sensitivity/stability)
  5. Phan tich da bien (step4 – multivariate X, Y, h + correlation)

Chay:
    python step5_pp1_complete.py --no-display
    python step5_pp1_complete.py --k 4 --no-display
    python step5_pp1_complete.py --missing-thresh 20.0 --no-display
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PP1 hoan chinh – 2 tram 4E + 4W')
parser.add_argument('--k', type=int, default=None,
                    help='So cum (neu khong truyen, se tu dong tim k toi uu)')
parser.add_argument('--missing-thresh', type=float, default=20.0,
                    help='Nguong %% missing cho phep (default: 20.0)')
parser.add_argument('--no-display', action='store_true',
                    help='Khong hien thi cua so (Agg backend)')
parser.add_argument('--skip-sensitivity', action='store_true',
                    help='Bo qua phan tich do nhay tham so')
parser.add_argument('--skip-multivariate', action='store_true',
                    help='Bo qua phan tich da bien (X, Y, h)')
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
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
#  Import
# ---------------------------------------------------------------------------
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

from gnss_clustering import config
from gnss_clustering.preprocessing import hampel_filter, reshape_by_window, kalman_filter_2d

plt.style.use(config.MATPLOTLIB_STYLE)

RD = os.path.join(config.RESULT_DIR, '09_pp1_complete')
os.makedirs(RD, exist_ok=True)


# ============================================================================
#  HELPER FUNCTIONS
# ============================================================================

def load_station_data(filepath):
    """Tai du lieu 1 tram tu CSV."""
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    df['Second_of_day'] = (
        df['Timestamp'].dt.hour * 3600
        + df['Timestamp'].dt.minute * 60
        + df['Timestamp'].dt.second
    )
    return df


def print_data_info(df, station_name):
    """In thong ke chi tiet cua 1 bo du lieu."""
    print(f"\n  {'='*50}")
    print(f"  THONG TIN DU LIEU – {station_name}")
    print(f"  {'='*50}")
    print(f"  Tong so dong:          {len(df):,}")
    print(f"  Khoang thoi gian:      {df['Timestamp'].min()} -> {df['Timestamp'].max()}")

    dates = sorted(df['Date'].unique())
    print(f"  So ngay co du lieu:    {len(dates)}")
    print(f"  Ngay dau:              {dates[0]}")
    print(f"  Ngay cuoi:             {dates[-1]}")

    # Thong ke so dong moi ngay
    daily_counts = df.groupby('Date').size()
    print(f"  Dong/ngay – mean:      {daily_counts.mean():.0f}")
    print(f"  Dong/ngay – min:       {daily_counts.min()}")
    print(f"  Dong/ngay – max:       {daily_counts.max()}")

    # Thong ke toa do
    cols = ['X_Coord', 'Y_Coord', 'h_Coord']
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        print(f"\n  {'Toa do':<12} {'mean':>14} {'std':>14} {'min':>14} {'max':>14}")
        print(f"  {'-'*56}")
        for col in available_cols:
            short = col.replace('_Coord', '')
            vals = df[col].dropna()
            print(f"  {short:<12} {vals.mean():>14.6f} {vals.std():>14.6f} "
                  f"{vals.min():>14.6f} {vals.max():>14.6f}")

    # Thong ke missing theo gio
    print(f"\n  Phan tich du lieu thieu theo gio:")
    n_hours = 0
    missing_pcts = []
    for date in dates:
        day_data = df[df['Date'] == date]
        for hour in range(24):
            start_s = hour * 3600
            end_s = (hour + 1) * 3600
            mask = (day_data['Second_of_day'] >= start_s) & (day_data['Second_of_day'] < end_s)
            n_pts = mask.sum()
            pct = (3600 - n_pts) / 3600 * 100
            missing_pcts.append(pct)
            n_hours += 1

    missing_pcts = np.array(missing_pcts)
    print(f"  Tong so gio:           {n_hours}")
    print(f"  Gio 0% thieu:          {np.sum(missing_pcts == 0)}")
    print(f"  Gio <5% thieu:         {np.sum(missing_pcts < 5)}")
    print(f"  Gio <10% thieu:        {np.sum(missing_pcts < 10)}")
    print(f"  Gio <20% thieu:        {np.sum(missing_pcts < 20)}")
    print(f"  Missing% – median:     {np.median(missing_pcts):.2f}%")
    print(f"  Missing% – mean:       {np.mean(missing_pcts):.2f}%")
    print(f"  Missing% – max:        {np.max(missing_pcts):.2f}%")


def create_hourly_matrix_from_df(df, coord='h_Coord', missing_threshold=0.0):
    """Tao ma tran theo gio tu DataFrame, cho phep noi suy NaN."""
    unique_dates = sorted(df['Date'].unique())
    hourly_data = []
    hourly_info = []

    for date in unique_dates:
        day_data = df[df['Date'] == date]
        for hour in range(24):
            start_s = hour * 3600
            end_s = (hour + 1) * 3600
            vec = np.full(3600, np.nan)
            mask = (day_data['Second_of_day'] >= start_s) & (day_data['Second_of_day'] < end_s)
            for _, row in day_data[mask].iterrows():
                idx = int(row['Second_of_day']) - start_s
                if 0 <= idx < 3600:
                    vec[idx] = row[coord]

            nan_pct = np.isnan(vec).sum() / 3600 * 100
            if nan_pct <= missing_threshold:
                # Noi suy NaN
                if nan_pct > 0:
                    s = pd.Series(vec)
                    vec = s.interpolate(method='linear').ffill().bfill().values
                hourly_data.append(vec)
                hourly_info.append({
                    'date': date, 'hour': hour,
                    'datetime': f"{date} {hour:02d}:00:00",
                    'missing_pct': nan_pct,
                })

    matrix = np.array(hourly_data) if hourly_data else np.array([]).reshape(0, 3600)
    info = pd.DataFrame(hourly_info)
    return matrix, info


def create_multivariate_hourly_matrix(df, coord_cols, missing_threshold=20.0):
    """Tao ma tran da bien theo gio."""
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


def preprocess_pipeline_simple(hourly_matrix):
    """Tien xu ly: Hampel -> reshape -> Kalman (tra ve data_filtered)."""
    filtered, _ = hampel_filter(hourly_matrix)
    reshaped = reshape_by_window(filtered)
    smoothed = kalman_filter_2d(reshaped)
    return smoothed


def pp1_extract_features(data_filtered):
    """PP1: Scale -> PCA -> t-SNE (2 lan)."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_filtered.T).T

    n_comp = min(config.PCA_N_COMPONENTS, data_scaled.shape[0], data_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=config.SEED)
    data_pca = pca.fit_transform(data_scaled)
    print(f"    PCA: {data_pca.shape}, explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    perplexity = min(config.TSNE_PERPLEXITY, len(data_scaled) - 1)
    tsne = TSNE(n_components=2, init='pca', random_state=config.SEED,
                perplexity=perplexity, learning_rate=config.TSNE_LEARNING_RATE,
                early_exaggeration=config.TSNE_EARLY_EXAGGERATION,
                metric=config.TSNE_METRIC)
    data_tsne = tsne.fit_transform(data_scaled)
    data_tsne = tsne.fit_transform(data_tsne)  # 2 lan nhu notebook
    print(f"    t-SNE: {data_tsne.shape}")
    return data_tsne, data_scaled


def run_clustering(data_tsne, k):
    """Chay KMeans, HAC, GMM tren t-SNE space."""
    results = {}

    km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
    labels = km.fit_predict(data_tsne)
    results['KMeans'] = _make_result(data_tsne, labels, k)

    hac = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hac.fit_predict(data_tsne)
    results['HAC'] = _make_result(data_tsne, labels, k)

    gmm = GaussianMixture(n_components=k, covariance_type='full',
                           random_state=config.SEED, max_iter=config.GMM_MAX_ITER)
    gmm.fit(data_tsne)
    labels = gmm.predict(data_tsne)
    res = _make_result(data_tsne, labels, k)
    res['aic'] = gmm.aic(data_tsne)
    res['bic'] = gmm.bic(data_tsne)
    results['GMM'] = res

    # DBSCAN
    nbrs = NearestNeighbors(n_neighbors=config.DBSCAN_MIN_SAMPLES).fit(data_tsne)
    distances, _ = nbrs.kneighbors(data_tsne)
    k_dist = np.sort(distances[:, config.DBSCAN_MIN_SAMPLES - 1])
    eps = np.percentile(k_dist, config.DBSCAN_EPS_PERCENTILE)
    db = DBSCAN(eps=eps, min_samples=config.DBSCAN_MIN_SAMPLES)
    labels = db.fit_predict(data_tsne)
    n_cls = len(np.unique(labels[labels != -1]))
    if n_cls >= 2:
        mask = labels != -1
        results['DBSCAN'] = _make_result(data_tsne[mask], labels[mask], n_cls)
        results['DBSCAN']['labels'] = labels
        results['DBSCAN']['n_noise'] = int(np.sum(labels == -1))
    else:
        results['DBSCAN'] = {'labels': labels, 'n_clusters': n_cls,
                              'silhouette': -1, 'calinski_harabasz': 0, 'davies_bouldin': 99}

    return results


def _make_result(data, labels, k):
    return {
        'labels': labels,
        'n_clusters': k,
        'silhouette': silhouette_score(data, labels),
        'calinski_harabasz': calinski_harabasz_score(data, labels),
        'davies_bouldin': davies_bouldin_score(data, labels),
    }


def _pairwise_ari(labels_list):
    n = len(labels_list)
    ari = []
    for i in range(n):
        for j in range(i + 1, n):
            ari.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    return ari


def _print_metrics(results, title):
    print(f"\n  {title}:")
    hdr = f"    {'Thuat toan':<14} {'k':>4} {'Silhouette':>12} {'Calinski':>12} {'Davies':>10}"
    print(hdr)
    print(f"    {'-'*52}")
    for m, r in results.items():
        print(f"    {m:<14} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
              f"{r['calinski_harabasz']:>12.1f} {r['davies_bouldin']:>10.4f}")


# ============================================================================
#  FIND OPTIMAL K (from step1)
# ============================================================================

def find_optimal_k(data_tsne, data_scaled, k_range=range(2, 11)):
    """Tim k toi uu bang voting giua KMeans, HAC, GMM."""
    print("\n  Tim k toi uu (k sweep + voting)...")
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
        labels = km.fit_predict(data_tsne)
        sil_km = silhouette_score(data_tsne, labels)

        hac = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = hac.fit_predict(data_tsne)
        sil_hac = silhouette_score(data_tsne, labels)

        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               random_state=config.SEED, max_iter=config.GMM_MAX_ITER)
        gmm.fit(data_tsne)
        labels = gmm.predict(data_tsne)
        sil_gmm = silhouette_score(data_tsne, labels)

        records.append({'k': k, 'KMeans_sil': sil_km, 'HAC_sil': sil_hac, 'GMM_sil': sil_gmm})
        print(f"    k={k}: KMeans={sil_km:.3f}, HAC={sil_hac:.3f}, GMM={sil_gmm:.3f}")

    df = pd.DataFrame(records)

    # Voting: moi thuat toan chon k co silhouette cao nhat
    best_km = int(df.loc[df['KMeans_sil'].idxmax(), 'k'])
    best_hac = int(df.loc[df['HAC_sil'].idxmax(), 'k'])
    best_gmm = int(df.loc[df['GMM_sil'].idxmax(), 'k'])
    votes = [best_km, best_hac, best_gmm]

    from collections import Counter
    vote_counts = Counter(votes)
    best_k = vote_counts.most_common(1)[0][0]
    n_votes = vote_counts.most_common(1)[0][1]

    print(f"\n    Voting: KMeans->k={best_km}, HAC->k={best_hac}, GMM->k={best_gmm}")
    print(f"    => k toi uu = {best_k} ({n_votes}/3 phieu)")

    return best_k, df


# ============================================================================
#  SENSITIVITY ANALYSIS (from step3)
# ============================================================================

def sensitivity_k_sweep(data_tsne, k_range=range(2, 11)):
    """Quet k cho KMeans, HAC, GMM."""
    records = []
    for k in k_range:
        for algo_name, algo_fn in [
            ('KMeans', lambda: KMeans(n_clusters=k, random_state=config.SEED, n_init=10)),
            ('HAC', lambda: AgglomerativeClustering(n_clusters=k, linkage='ward')),
        ]:
            model = algo_fn()
            labels = model.fit_predict(data_tsne)
            records.append({
                'algorithm': algo_name, 'k': k,
                'silhouette': silhouette_score(data_tsne, labels),
                'calinski_harabasz': calinski_harabasz_score(data_tsne, labels),
                'davies_bouldin': davies_bouldin_score(data_tsne, labels),
            })

        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               random_state=config.SEED, max_iter=config.GMM_MAX_ITER)
        gmm.fit(data_tsne)
        labels = gmm.predict(data_tsne)
        records.append({
            'algorithm': 'GMM', 'k': k,
            'silhouette': silhouette_score(data_tsne, labels),
            'calinski_harabasz': calinski_harabasz_score(data_tsne, labels),
            'davies_bouldin': davies_bouldin_score(data_tsne, labels),
            'aic': gmm.aic(data_tsne), 'bic': gmm.bic(data_tsne),
        })
    return pd.DataFrame(records)


def sensitivity_gmm_cov(data_tsne, k_range=range(2, 11)):
    """Quet GMM covariance types."""
    records = []
    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        for k in k_range:
            try:
                gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                                       random_state=config.SEED, max_iter=config.GMM_MAX_ITER)
                gmm.fit(data_tsne)
                labels = gmm.predict(data_tsne)
                records.append({
                    'covariance_type': cov_type, 'k': k,
                    'silhouette': silhouette_score(data_tsne, labels),
                    'bic': gmm.bic(data_tsne), 'aic': gmm.aic(data_tsne),
                })
            except Exception:
                continue
    return pd.DataFrame(records)


def sensitivity_dbscan(data_tsne, min_pts_range=range(2, 10), n_eps=15):
    """Quet eps x MinPts cho DBSCAN."""
    records = []
    for min_pts in min_pts_range:
        nbrs = NearestNeighbors(n_neighbors=min_pts).fit(data_tsne)
        distances, _ = nbrs.kneighbors(data_tsne)
        k_distances = np.sort(distances[:, min_pts - 1])
        eps_values = np.unique(np.round(
            np.percentile(k_distances, np.linspace(50, 99, n_eps)), 6))

        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=min_pts)
            labels = db.fit_predict(data_tsne)
            n_cls = len(np.unique(labels[labels != -1]))
            n_noise = int(np.sum(labels == -1))
            sil = -1
            if n_cls >= 2:
                mask = labels != -1
                if np.sum(mask) > n_cls:
                    sil = silhouette_score(data_tsne[mask], labels[mask])
            records.append({
                'min_pts': min_pts, 'eps': eps, 'n_clusters': n_cls,
                'n_noise': n_noise, 'noise_ratio': n_noise / len(labels),
                'silhouette': sil,
            })
    return pd.DataFrame(records)


# ============================================================================
#  STABILITY ANALYSIS (from step3)
# ============================================================================

def stability_analysis(data_tsne, k, n_runs=30):
    """Chay stability: multi-init ARI cho KMeans, HAC, GMM, DBSCAN."""
    results = {}

    # KMeans
    km_labels = []
    km_sils = []
    for i in range(n_runs):
        km = KMeans(n_clusters=k, random_state=i, n_init=10)
        labels = km.fit_predict(data_tsne)
        km_labels.append(labels)
        km_sils.append(silhouette_score(data_tsne, labels))
    km_ari = _pairwise_ari(km_labels)
    results['KMeans'] = {'ari': km_ari, 'sils': km_sils,
                          'ari_mean': np.mean(km_ari), 'ari_std': np.std(km_ari)}

    # HAC (4 linkage variants)
    hac_labels = []
    hac_sils = []
    for linkage in ['ward', 'complete', 'average', 'single']:
        hac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = hac.fit_predict(data_tsne)
        hac_labels.append(labels)
        hac_sils.append(silhouette_score(data_tsne, labels))
    hac_ari = _pairwise_ari(hac_labels)
    results['HAC'] = {'ari': hac_ari, 'sils': hac_sils,
                       'ari_mean': np.mean(hac_ari), 'ari_std': np.std(hac_ari)}

    # GMM
    gmm_labels = []
    gmm_sils = []
    for i in range(n_runs):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               random_state=i, max_iter=config.GMM_MAX_ITER)
        gmm.fit(data_tsne)
        labels = gmm.predict(data_tsne)
        gmm_labels.append(labels)
        gmm_sils.append(silhouette_score(data_tsne, labels))
    gmm_ari = _pairwise_ari(gmm_labels)
    results['GMM'] = {'ari': gmm_ari, 'sils': gmm_sils,
                       'ari_mean': np.mean(gmm_ari), 'ari_std': np.std(gmm_ari)}

    # DBSCAN (sweep eps)
    min_pts = config.DBSCAN_MIN_SAMPLES
    nbrs = NearestNeighbors(n_neighbors=min_pts).fit(data_tsne)
    distances, _ = nbrs.kneighbors(data_tsne)
    k_dist = np.sort(distances[:, min_pts - 1])
    eps_values = np.unique(np.round(np.percentile(k_dist, np.linspace(70, 95, 15)), 6))

    db_labels = []
    db_sils = []
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_pts)
        labels = db.fit_predict(data_tsne)
        n_cls = len(np.unique(labels[labels != -1]))
        if n_cls >= 2:
            mask = labels != -1
            if np.sum(mask) > n_cls:
                db_labels.append(labels)
                db_sils.append(silhouette_score(data_tsne[mask], labels[mask]))

    db_ari = _pairwise_ari(db_labels) if len(db_labels) >= 2 else []
    results['DBSCAN'] = {'ari': db_ari, 'sils': db_sils,
                          'ari_mean': np.mean(db_ari) if db_ari else 0.0,
                          'ari_std': np.std(db_ari) if db_ari else 0.0}

    return results


# ============================================================================
#  INTRA-CLUSTER CORRELATION (from step4)
# ============================================================================

def intra_cluster_correlation(matrices_dict, labels, coord_names):
    """Tuong quan noi cum giua cac kenh (X, Y, h)."""
    n_ch = len(coord_names)
    corr_results = {}

    for cid in np.unique(labels):
        mask = labels == cid
        corr_matrix = np.zeros((n_ch, n_ch))
        for i, ci in enumerate(coord_names):
            for j, cj in enumerate(coord_names):
                corrs = []
                for idx in np.where(mask)[0]:
                    si = matrices_dict[ci][idx]
                    sj = matrices_dict[cj][idx]
                    valid = ~(np.isnan(si) | np.isnan(sj))
                    if np.sum(valid) > 10:
                        r, _ = pearsonr(si[valid], sj[valid])
                        corrs.append(r)
                corr_matrix[i, j] = np.mean(corrs) if corrs else 0
        corr_results[cid] = {'corr_matrix': corr_matrix, 'n_samples': int(np.sum(mask)),
                              'coord_names': coord_names}
    return corr_results


def cross_station_corr_by_cluster(proc_e, proc_w, labels_combined, n_e, coord_names):
    """Tuong quan giua 2 tram trong tung cum."""
    labels_e = labels_combined[:n_e]
    labels_w = labels_combined[n_e:]
    results = {}

    for cid in np.unique(labels_combined):
        mask_e = labels_e == cid
        mask_w = labels_w == cid
        cluster_corr = {}
        for coord in coord_names:
            if np.sum(mask_e) > 0 and np.sum(mask_w) > 0:
                mean_e = np.nanmean(proc_e[coord][mask_e], axis=0)
                mean_w = np.nanmean(proc_w[coord][mask_w], axis=0)
                valid = ~(np.isnan(mean_e) | np.isnan(mean_w))
                if np.sum(valid) > 10:
                    r, p = pearsonr(mean_e[valid], mean_w[valid])
                    cluster_corr[coord] = {'r': r, 'p': p}
                else:
                    cluster_corr[coord] = {'r': np.nan, 'p': np.nan}
            else:
                cluster_corr[coord] = {'r': np.nan, 'p': np.nan}
        results[cid] = {'n_e': int(np.sum(mask_e)), 'n_w': int(np.sum(mask_w)),
                          'correlations': cluster_corr}
    return results


# ============================================================================
#  VISUALIZATION
# ============================================================================

def plot_k_sensitivity(df_k, station, save=True):
    """Bieu do sensitivity k sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('silhouette', 'Silhouette (cao = tot)', True),
        ('calinski_harabasz', 'Calinski-Harabasz (cao = tot)', True),
        ('davies_bouldin', 'Davies-Bouldin (thap = tot)', False),
    ]
    for ax, (metric, title, higher) in zip(axes, metrics):
        for algo in ['KMeans', 'HAC', 'GMM']:
            sub = df_k[df_k['algorithm'] == algo]
            ax.plot(sub['k'], sub[metric], 'o-', label=algo, linewidth=2, markersize=5)
        ax.set_xlabel('k')
        ax.set_ylabel(metric)
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df_k['k'].unique()))
    fig.suptitle(f'{station} – Do nhay tham so k', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, f'C01_k_sensitivity_{station}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"    [saved] {path}")
    plt.close(fig)


def plot_scatter_clustering(data_tsne, results, station, save=True):
    """Bieu do scatter clustering."""
    methods = list(results.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i, (method, res) in enumerate(results.items()):
        ax = axes[i]
        labels = res['labels']
        unique_l = np.unique(labels)
        for j, label in enumerate(unique_l):
            mask = labels == label
            color = 'gray' if label == -1 else COLORS[j % len(COLORS)]
            name = 'Noise' if label == -1 else f'C{label}'
            ax.scatter(data_tsne[mask, 0], data_tsne[mask, 1],
                       c=color, s=40, alpha=0.7, label=name)
        ax.set_title(f'{method} (Sil={res["silhouette"]:.3f})', fontweight='bold', fontsize=10)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{station} – PP1 Clustering (t-SNE)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, f'C02_scatter_{station}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"    [saved] {path}")
    plt.close(fig)


def plot_stability_comparison(stab_e, stab_w, save=True):
    """So sanh stability giua 2 tram."""
    methods = list(stab_e.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    COLORS = ['steelblue', 'coral', 'mediumseagreen', 'orchid']

    for ax_idx, (stab, title) in enumerate([(stab_e, '4E'), (stab_w, '4W')]):
        ax = axes[ax_idx]
        data = [stab[m]['ari'] for m in methods if stab[m]['ari']]
        labels = [m for m in methods if stab[m]['ari']]
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(COLORS[i % len(COLORS)])
                patch.set_alpha(0.7)
        ax.axhline(0.75, color='green', linestyle='--', alpha=0.5, label='Tot (0.75)')
        ax.set_title(f'{title} – Stability (Pairwise ARI)', fontweight='bold')
        ax.set_ylabel('ARI')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RD, 'C03_stability_comparison.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"    [saved] {path}")
    plt.close(fig)


def plot_combined_scatter(tsne_e, tsne_w, labels_e, labels_w, k, method, save=True):
    """Scatter 2 tram canh nhau."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for ax_idx, (tsne, labels, name) in enumerate([
        (tsne_e, labels_e, '4E'), (tsne_w, labels_w, '4W')
    ]):
        ax = axes[ax_idx]
        for j in range(k):
            mask = labels == j
            ax.scatter(tsne[mask, 0], tsne[mask, 1],
                       c=COLORS[j % len(COLORS)], s=50, alpha=0.7, label=f'C{j}')
        ax.set_title(f'{name} – {method}', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # So sanh kich thuoc cum
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

    fig.suptitle(f'So sanh 4E vs 4W – {method}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, f'C04_comparison_{method}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"    [saved] {path}")
    plt.close(fig)


def plot_joint_clustering(tsne_combined, labels_combined, n_e, k, method, save=True):
    """Scatter clustering chung 2 tram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    ax = axes[0]
    for j in range(k):
        mask = labels_combined == j
        ax.scatter(tsne_combined[mask, 0], tsne_combined[mask, 1],
                   c=COLORS[j % len(COLORS)], s=40, alpha=0.6, label=f'C{j}')
    ax.set_title(f'Clustering chung – {method}', fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(tsne_combined[:n_e, 0], tsne_combined[:n_e, 1],
               c='steelblue', s=40, alpha=0.6, label='4E', marker='o')
    ax.scatter(tsne_combined[n_e:, 0], tsne_combined[n_e:, 1],
               c='coral', s=40, alpha=0.6, label='4W', marker='^')
    ax.set_title('Phan bo theo tram', fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Clustering chung 4E+4W – {method}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, f'C05_joint_{method}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"    [saved] {path}")
    plt.close(fig)


def plot_intra_cluster_corr(corr_results, station, save=True):
    """Ma tran tuong quan noi cum."""
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

    fig.suptitle(f'{station} – Tuong quan noi cum (X, Y, h)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, f'C06_intra_corr_{station}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"    [saved] {path}")
    plt.close(fig)


def plot_cross_station_corr(cross_results, save=True):
    """Tuong quan giua 2 tram theo cum."""
    unique_labels = sorted(cross_results.keys())
    n = len(unique_labels)
    coord_names = ['X_Coord', 'Y_Coord', 'h_Coord']
    short_names = ['X', 'Y', 'h']
    COLORS = ['steelblue', 'coral', 'mediumseagreen']

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, cid in enumerate(unique_labels):
        ax = axes[i]
        res = cross_results[cid]
        vals = [res['correlations'][c]['r'] for c in coord_names]
        bars = ax.bar(short_names, vals, color=COLORS, alpha=0.7, edgecolor='black')
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f'Cum {cid} (4E:{res["n_e"]}, 4W:{res["n_w"]})', fontweight='bold')
        ax.set_ylabel('Pearson r (4E vs 4W)')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Tuong quan 4E vs 4W trong tung cum', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, 'C07_cross_station_corr.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"    [saved] {path}")
    plt.close(fig)


def plot_metrics_comparison(res_e, res_w, res_combined, save=True):
    """So sanh metrics 3 cach."""
    methods = [m for m in res_e.keys() if m != 'DBSCAN']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric in enumerate(['silhouette', 'calinski_harabasz', 'davies_bouldin']):
        ax = axes[idx]
        x = np.arange(len(methods))
        width = 0.25
        vals_e = [res_e[m][metric] for m in methods]
        vals_w = [res_w[m][metric] for m in methods]
        vals_c = [res_combined[m][metric] for m in methods]

        ax.bar(x - width, vals_e, width, label='4E', color='steelblue', alpha=0.7)
        ax.bar(x, vals_w, width, label='4W', color='coral', alpha=0.7)
        ax.bar(x + width, vals_c, width, label='4E+4W', color='mediumseagreen', alpha=0.7)

        title_map = {'silhouette': 'Silhouette (cao=tot)',
                      'calinski_harabasz': 'Calinski-Harabasz (cao=tot)',
                      'davies_bouldin': 'Davies-Bouldin (thap=tot)'}
        ax.set_title(title_map[metric], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('So sanh metrics: 4E vs 4W vs 4E+4W chung', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(RD, 'C08_metrics_comparison.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"    [saved] {path}")
    plt.close(fig)


# ============================================================================
#  MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHAN TICH PP1 HOAN CHINH – 2 TRAM 4E VA 4W")
    print("=" * 70)

    missing_thresh = args.missing_thresh
    file_e = os.path.join(config.DATA_DIR, 'full_gnss_2e.csv')
    file_w = os.path.join(config.DATA_DIR, 'full_gnss_2w.csv')

    # ======================================================================
    #  BUOC 1: TAI DU LIEU + THONG KE CHI TIET
    # ======================================================================
    print("\n" + "=" * 70)
    print("[BUOC 1/6] TAI DU LIEU VA THONG KE CHI TIET")
    print("=" * 70)

    df_e = load_station_data(file_e)
    df_w = load_station_data(file_w)

    print_data_info(df_e, 'TRAM 4E')
    print_data_info(df_w, 'TRAM 4W')

    # ======================================================================
    #  BUOC 2: TAO MA TRAN THEO GIO (h_Coord) + MATCHED HOURS
    # ======================================================================
    print("\n" + "=" * 70)
    print("[BUOC 2/6] TAO MA TRAN THEO GIO (h_Coord) – MATCHED HOURS")
    print("=" * 70)

    print(f"\n  Missing threshold: {missing_thresh}%")

    matrix_e, info_e_all = create_hourly_matrix_from_df(df_e, 'h_Coord', missing_thresh)
    matrix_w, info_w_all = create_hourly_matrix_from_df(df_w, 'h_Coord', missing_thresh)
    print(f"  4E (truoc match): {matrix_e.shape[0]} gio")
    print(f"  4W (truoc match): {matrix_w.shape[0]} gio")

    # Matched hours – chi giu gio chung
    keys_e = set(info_e_all['datetime'])
    keys_w = set(info_w_all['datetime'])
    common = sorted(keys_e & keys_w)
    print(f"  Gio chung: {len(common)}")

    mask_e = info_e_all['datetime'].isin(common).values
    mask_w = info_w_all['datetime'].isin(common).values
    matrix_e = matrix_e[mask_e]
    matrix_w = matrix_w[mask_w]
    info_e = info_e_all[mask_e].reset_index(drop=True)
    info_w = info_w_all[mask_w].reset_index(drop=True)

    print(f"  4E (sau match): {matrix_e.shape}")
    print(f"  4W (sau match): {matrix_w.shape}")

    # Tien xu ly
    print("\n  Tien xu ly (Hampel -> reshape -> Kalman)...")
    filtered_e = preprocess_pipeline_simple(matrix_e)
    filtered_w = preprocess_pipeline_simple(matrix_w)
    print(f"  4E filtered: {filtered_e.shape}")
    print(f"  4W filtered: {filtered_w.shape}")

    # ======================================================================
    #  BUOC 3: TIM K TOI UU (step1)
    # ======================================================================
    print("\n" + "=" * 70)
    print("[BUOC 3/6] TIM K TOI UU (PP1)")
    print("=" * 70)

    print("\n  --- 4E ---")
    print("    Scale -> PCA -> t-SNE...")
    tsne_e, scaled_e = pp1_extract_features(filtered_e)

    if args.k is None:
        best_k_e, df_k_e = find_optimal_k(tsne_e, scaled_e)
        print(f"\n  --- 4W ---")
        print("    Scale -> PCA -> t-SNE...")
        tsne_w, scaled_w = pp1_extract_features(filtered_w)
        best_k_w, df_k_w = find_optimal_k(tsne_w, scaled_w)

        # Dung k chung = mode cua 2 tram
        from collections import Counter
        vote = Counter([best_k_e, best_k_w]).most_common(1)[0][0]
        k = vote
        print(f"\n  k toi uu: 4E={best_k_e}, 4W={best_k_w} => chon k={k}")
    else:
        k = args.k
        print(f"\n  Su dung k={k} (tu tham so)")
        print(f"\n  --- 4W ---")
        print("    Scale -> PCA -> t-SNE...")
        tsne_w, scaled_w = pp1_extract_features(filtered_w)

    # ======================================================================
    #  BUOC 4: PHAN CUM CHI TIET (step2)
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"[BUOC 4/6] PHAN CUM PP1 (k={k})")
    print("=" * 70)

    print("\n  --- 4E ---")
    results_e = run_clustering(tsne_e, k)
    _print_metrics(results_e, '4E – Metrics')

    print("\n  --- 4W ---")
    results_w = run_clustering(tsne_w, k)
    _print_metrics(results_w, '4W – Metrics')

    # Clustering chung 2 tram
    print("\n  --- 4E + 4W chung ---")
    n_e = len(filtered_e)
    combined = np.vstack([filtered_e, filtered_w])
    print(f"    Combined: {combined.shape}")
    print("    Scale -> PCA -> t-SNE...")
    tsne_combined, scaled_combined = pp1_extract_features(combined)
    results_combined = run_clustering(tsne_combined, k)
    _print_metrics(results_combined, '4E+4W chung – Metrics')

    # Chon best method
    best_e = max(results_e.keys(), key=lambda m: results_e[m]['silhouette'])
    best_w = max(results_w.keys(), key=lambda m: results_w[m]['silhouette'])
    best_c = max(results_combined.keys(), key=lambda m: results_combined[m]['silhouette'])

    # Visualization
    print("\n  Ve bieu do clustering...")
    plot_scatter_clustering(tsne_e, results_e, '4E')
    plot_scatter_clustering(tsne_w, results_w, '4W')
    plot_combined_scatter(tsne_e, tsne_w, results_e[best_e]['labels'],
                          results_w[best_w]['labels'], k, best_e)
    plot_joint_clustering(tsne_combined, results_combined[best_c]['labels'],
                          n_e, k, best_c)
    plot_metrics_comparison(results_e, results_w, results_combined)

    # ======================================================================
    #  BUOC 5: DO NHAY + DO ON DINH (step3)
    # ======================================================================
    if not args.skip_sensitivity:
        print("\n" + "=" * 70)
        print("[BUOC 5/6] PHAN TICH DO NHAY THAM SO & DO ON DINH")
        print("=" * 70)

        # Sensitivity k sweep
        print("\n  5a. Sensitivity – K sweep...")
        print("    4E...")
        df_sens_e = sensitivity_k_sweep(tsne_e)
        print("    4W...")
        df_sens_w = sensitivity_k_sweep(tsne_w)

        plot_k_sensitivity(df_sens_e, '4E')
        plot_k_sensitivity(df_sens_w, '4W')

        # In tom tat sensitivity
        print("\n  K tot nhat theo Silhouette:")
        for station, df_s in [('4E', df_sens_e), ('4W', df_sens_w)]:
            print(f"    {station}:")
            for algo in ['KMeans', 'HAC', 'GMM']:
                sub = df_s[df_s['algorithm'] == algo]
                best = sub.loc[sub['silhouette'].idxmax()]
                print(f"      {algo}: k={int(best['k'])} (Sil={best['silhouette']:.4f})")

        # Sensitivity GMM covariance
        print("\n  5b. GMM Covariance Types...")
        df_gmm_e = sensitivity_gmm_cov(tsne_e)
        df_gmm_w = sensitivity_gmm_cov(tsne_w)

        print("    4E – Best BIC:")
        for cov in df_gmm_e['covariance_type'].unique():
            sub = df_gmm_e[df_gmm_e['covariance_type'] == cov]
            best = sub.loc[sub['bic'].idxmin()]
            print(f"      {cov}: k={int(best['k'])}, BIC={best['bic']:.1f}")

        print("    4W – Best BIC:")
        for cov in df_gmm_w['covariance_type'].unique():
            sub = df_gmm_w[df_gmm_w['covariance_type'] == cov]
            best = sub.loc[sub['bic'].idxmin()]
            print(f"      {cov}: k={int(best['k'])}, BIC={best['bic']:.1f}")

        # Sensitivity DBSCAN
        print("\n  5c. DBSCAN Sensitivity...")
        df_db_e = sensitivity_dbscan(tsne_e)
        df_db_w = sensitivity_dbscan(tsne_w)

        for station, df_db in [('4E', df_db_e), ('4W', df_db_w)]:
            valid = df_db[df_db['silhouette'] > -1]
            if len(valid) > 0:
                best = valid.loc[valid['silhouette'].idxmax()]
                print(f"    {station} best DBSCAN: MinPts={int(best['min_pts'])}, "
                      f"eps={best['eps']:.4f}, k={int(best['n_clusters'])}, "
                      f"Sil={best['silhouette']:.4f}")

        # Stability
        print("\n  5d. Stability Analysis...")
        print(f"    4E (k={k})...")
        stab_e = stability_analysis(tsne_e, k)
        print(f"    4W (k={k})...")
        stab_w = stability_analysis(tsne_w, k)

        plot_stability_comparison(stab_e, stab_w)

        print("\n  Stability Summary:")
        hdr = f"    {'Tram':<6} {'Thuat toan':<12} {'ARI mean':>10} {'ARI std':>10} {'On dinh?':>10}"
        print(hdr)
        print(f"    {'-'*48}")
        for station, stab in [('4E', stab_e), ('4W', stab_w)]:
            for method, res in stab.items():
                stable = "Co" if res['ari_mean'] >= 0.75 else ("TB" if res['ari_mean'] >= 0.5 else "Khong")
                print(f"    {station:<6} {method:<12} {res['ari_mean']:>10.3f} "
                      f"{res['ari_std']:>10.3f} {stable:>10}")

        # Luu CSV
        df_sens_e.to_csv(os.path.join(RD, 'C_sensitivity_4E.csv'), index=False)
        df_sens_w.to_csv(os.path.join(RD, 'C_sensitivity_4W.csv'), index=False)
    else:
        print("\n  [skip] Phan tich do nhay (--skip-sensitivity)")

    # ======================================================================
    #  BUOC 6: PHAN TICH DA BIEN (step4)
    # ======================================================================
    if not args.skip_multivariate:
        print("\n" + "=" * 70)
        print("[BUOC 6/6] PHAN TICH DA BIEN (X, Y, h) + TUONG QUAN")
        print("=" * 70)

        coord_cols = ['X_Coord', 'Y_Coord', 'h_Coord']

        print(f"\n  Tao ma tran da bien (missing threshold={missing_thresh}%)...")
        mv_e_all, mv_info_e_all = create_multivariate_hourly_matrix(df_e, coord_cols, missing_thresh)
        mv_w_all, mv_info_w_all = create_multivariate_hourly_matrix(df_w, coord_cols, missing_thresh)

        # Matched hours
        keys_e_mv = set(mv_info_e_all['datetime'])
        keys_w_mv = set(mv_info_w_all['datetime'])
        common_mv = sorted(keys_e_mv & keys_w_mv)
        print(f"  4E: {len(mv_info_e_all)} gio, 4W: {len(mv_info_w_all)} gio, chung: {len(common_mv)}")

        mask_e_mv = mv_info_e_all['datetime'].isin(common_mv).values
        mask_w_mv = mv_info_w_all['datetime'].isin(common_mv).values

        mv_e = {col: mv_e_all[col][mask_e_mv] for col in coord_cols}
        mv_w = {col: mv_w_all[col][mask_w_mv] for col in coord_cols}

        print(f"  Matched: 4E={mv_e['h_Coord'].shape}, 4W={mv_w['h_Coord'].shape}")

        # Tien xu ly tung kenh
        print("  Tien xu ly tung kenh...")
        proc_e = {}
        proc_w = {}
        for col in coord_cols:
            proc_e[col] = preprocess_channel(mv_e[col])
            proc_w[col] = preprocess_channel(mv_w[col])

        # Clustering da bien rieng tung tram
        print(f"\n  Clustering da bien rieng (k={k})...")
        feat_e = np.hstack(list(proc_e.values()))
        feat_w = np.hstack(list(proc_w.values()))
        print(f"    4E features: {feat_e.shape}")
        print(f"    4W features: {feat_w.shape}")

        print("    4E: Scale -> PCA -> t-SNE...")
        mv_tsne_e, _ = pp1_extract_features(feat_e)
        mv_res_e = run_clustering(mv_tsne_e, k)
        _print_metrics(mv_res_e, '4E da bien')

        print("    4W: Scale -> PCA -> t-SNE...")
        mv_tsne_w, _ = pp1_extract_features(feat_w)
        mv_res_w = run_clustering(mv_tsne_w, k)
        _print_metrics(mv_res_w, '4W da bien')

        # Clustering chung
        print("\n  Clustering da bien chung 4E+4W...")
        n_e_mv = len(feat_e)
        feat_combined_mv = np.vstack([feat_e, feat_w])
        print(f"    Combined: {feat_combined_mv.shape}")
        print("    Scale -> PCA -> t-SNE...")
        mv_tsne_comb, _ = pp1_extract_features(feat_combined_mv)
        mv_res_comb = run_clustering(mv_tsne_comb, k)
        _print_metrics(mv_res_comb, '4E+4W da bien chung')

        best_mv_e = max(mv_res_e.keys(), key=lambda m: mv_res_e[m]['silhouette'])
        best_mv_w = max(mv_res_w.keys(), key=lambda m: mv_res_w[m]['silhouette'])
        best_mv_c = max(mv_res_comb.keys(), key=lambda m: mv_res_comb[m]['silhouette'])

        # Tuong quan noi cum
        print("\n  Tuong quan noi cum...")
        corr_e = intra_cluster_correlation(proc_e, mv_res_e[best_mv_e]['labels'], coord_cols)
        corr_w = intra_cluster_correlation(proc_w, mv_res_w[best_mv_w]['labels'], coord_cols)

        for station, corr in [('4E', corr_e), ('4W', corr_w)]:
            print(f"\n    {station}:")
            for cid, res in corr.items():
                names = [n.replace('_Coord', '') for n in res['coord_names']]
                print(f"      Cum {cid} (n={res['n_samples']}):")
                for i, ni in enumerate(names):
                    for j, nj in enumerate(names):
                        if j > i:
                            print(f"        {ni}-{nj}: r={res['corr_matrix'][i,j]:.3f}")

        # Cross-station correlation
        best_labels_c = mv_res_comb[best_mv_c]['labels']
        cross_corr = cross_station_corr_by_cluster(proc_e, proc_w, best_labels_c,
                                                    n_e_mv, coord_cols)
        print("\n  Tuong quan 4E vs 4W theo cum:")
        for cid, res in cross_corr.items():
            print(f"    Cum {cid} (4E:{res['n_e']}, 4W:{res['n_w']}):")
            for coord, cr in res['correlations'].items():
                short = coord.replace('_Coord', '')
                r_val = cr['r']
                print(f"      {short}: r={r_val:.3f}" if not np.isnan(r_val) else f"      {short}: N/A")

        # Visualization da bien
        print("\n  Ve bieu do da bien...")
        plot_intra_cluster_corr(corr_e, '4E')
        plot_intra_cluster_corr(corr_w, '4W')
        plot_cross_station_corr(cross_corr)
        plot_combined_scatter(mv_tsne_e, mv_tsne_w,
                              mv_res_e[best_mv_e]['labels'],
                              mv_res_w[best_mv_w]['labels'], k, f'MV_{best_mv_e}')
        plot_joint_clustering(mv_tsne_comb, best_labels_c, n_e_mv, k, f'MV_{best_mv_c}')
    else:
        print("\n  [skip] Phan tich da bien (--skip-multivariate)")

    # ======================================================================
    #  TOM TAT CUOI CUNG
    # ======================================================================
    print("\n" + "=" * 70)
    print("TOM TAT KET QUA CUOI CUNG")
    print("=" * 70)

    print(f"\n  So cum: k = {k}")
    print(f"  Missing threshold: {missing_thresh}%")
    print(f"  So mau matched: 4E={matrix_e.shape[0]}, 4W={matrix_w.shape[0]}")

    print(f"\n  --- PP1 don bien (h_Coord) ---")
    for station, results in [('4E', results_e), ('4W', results_w)]:
        best = max(results.keys(), key=lambda m: results[m]['silhouette'])
        print(f"  {station} (best: {best}):")
        for m, r in results.items():
            flag = ' <-- best' if m == best else ''
            print(f"    {m:<14} Sil={r['silhouette']:.4f} Cal={r['calinski_harabasz']:.1f} "
                  f"Dav={r['davies_bouldin']:.4f}{flag}")

    print(f"\n  4E+4W chung:")
    for m, r in results_combined.items():
        flag = ' <-- best' if m == best_c else ''
        print(f"    {m:<14} Sil={r['silhouette']:.4f} Cal={r['calinski_harabasz']:.1f} "
              f"Dav={r['davies_bouldin']:.4f}{flag}")

    if not args.skip_sensitivity:
        print(f"\n  --- Stability (k={k}) ---")
        for station, stab in [('4E', stab_e), ('4W', stab_w)]:
            print(f"  {station}:")
            for method, res in stab.items():
                print(f"    {method:<12} ARI={res['ari_mean']:.3f} +/- {res['ari_std']:.3f}")

    if not args.skip_multivariate:
        print(f"\n  --- PP1 da bien (X, Y, h) ---")
        for station, results in [('4E', mv_res_e), ('4W', mv_res_w)]:
            best = max(results.keys(), key=lambda m: results[m]['silhouette'])
            print(f"  {station} (best: {best}):")
            for m, r in results.items():
                print(f"    {m:<14} Sil={r['silhouette']:.4f}")

    print(f"\n  Tat ca hinh anh: {RD}")
    print("  HOAN THANH!")


if __name__ == '__main__':
    main()
