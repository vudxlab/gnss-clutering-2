"""
Module kiem tra tinh on dinh cua ket qua phan cum (Stability Analysis).

Gom 3 phan chinh:
  1. Bootstrap Stability  – Lay mau bootstrap nhieu lan, tinh ARI giua cac lan
  2. Temporal Coherence   – Kiem tra nhan cum co cau truc thoi gian (runs test)
  3. Visualization        – Ve bieu do tong hop ket qua stability

ARI (Adjusted Rand Index):
  > 0.8  : cum rat on dinh
  0.5-0.8: cum tuong doi on dinh
  < 0.5  : cum khong dang tin cay
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.utils import resample

from . import config


# ============================================================================
# 1. Bootstrap Stability Analysis
# ============================================================================

def bootstrap_stability(X, n_clusters, method='hac', n_iterations=100,
                        sample_ratio=0.8, random_state=None):
    """
    Danh gia tinh on dinh cua phan cum bang bootstrap resampling.

    Quy trinh:
      1. Phan cum toan bo du lieu -> labels_full
      2. Lap n_iterations lan:
         a. Lay mau ngau nhien sample_ratio% du lieu
         b. Phan cum tren mau -> labels_boot
         c. Phan cum toan bo du lieu tai cac chi so tuong ung -> labels_ref
         d. Tinh ARI giua labels_boot va labels_ref
      3. Tra ve phan phoi ARI

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Du lieu da chuan hoa.
    n_clusters : int
        So cum.
    method : str
        'hac' hoac 'gmm'.
    n_iterations : int
        So lan bootstrap (mac dinh 100).
    sample_ratio : float
        Ty le mau lay moi lan (mac dinh 0.8 = 80%).
    random_state : int or None
        Seed cho reproducibility.

    Returns
    -------
    result : dict
        ari_scores    : list of float – ARI moi lan bootstrap
        ari_mean      : float
        ari_std       : float
        ari_median    : float
        n_iterations  : int
        method        : str
        n_clusters    : int
        interpretation: str – Dien giai ket qua
    """
    if random_state is None:
        random_state = config.SEED

    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    n_boot = int(n_samples * sample_ratio)

    ari_scores = []

    for i in range(n_iterations):
        # Lay mau bootstrap (khong thay the de giu cac chi so duy nhat)
        boot_indices = rng.choice(n_samples, size=n_boot, replace=False)
        X_boot = X[boot_indices]

        try:
            # Phan cum tren mau bootstrap
            labels_boot = _fit_predict(X_boot, n_clusters, method, random_state + i)

            # Phan cum toan bo du lieu tai cac chi so tuong ung
            labels_ref = _fit_predict(X[boot_indices], n_clusters, method, random_state)

            # Tinh ARI
            ari = adjusted_rand_score(labels_ref, labels_boot)
            ari_scores.append(ari)
        except Exception:
            continue

    if not ari_scores:
        return {
            'ari_scores': [], 'ari_mean': 0.0, 'ari_std': 0.0,
            'ari_median': 0.0, 'n_iterations': 0, 'method': method,
            'n_clusters': n_clusters, 'interpretation': 'Khong the tinh',
        }

    ari_mean = float(np.mean(ari_scores))
    ari_std = float(np.std(ari_scores))
    ari_median = float(np.median(ari_scores))

    interpretation = _interpret_ari(ari_mean)

    return {
        'ari_scores': ari_scores,
        'ari_mean': ari_mean,
        'ari_std': ari_std,
        'ari_median': ari_median,
        'n_iterations': len(ari_scores),
        'method': method,
        'n_clusters': n_clusters,
        'interpretation': interpretation,
    }


def _fit_predict(X, n_clusters, method, random_state):
    """Phan cum voi method cho truoc."""
    if method == 'hac':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        return model.fit_predict(X)
    elif method == 'gmm':
        model = GaussianMixture(
            n_components=n_clusters, covariance_type='full',
            random_state=random_state, max_iter=config.GMM_MAX_ITER,
        )
        model.fit(X)
        return model.predict(X)
    else:
        raise ValueError(f"Method '{method}' khong duoc ho tro. Dung 'hac' hoac 'gmm'.")


def _interpret_ari(ari_mean):
    """Dien giai ket qua ARI."""
    if ari_mean >= 0.9:
        return "Rat on dinh – cum dang tin cay cao"
    elif ari_mean >= 0.75:
        return "On dinh tot – cum co the tin cay"
    elif ari_mean >= 0.5:
        return "Tuong doi on dinh – can xem xet them"
    elif ari_mean >= 0.25:
        return "Khong on dinh – cum it dang tin cay"
    else:
        return "Rat khong on dinh – cum khong dang tin cay"


# ============================================================================
# 2. Temporal Coherence Analysis
# ============================================================================

def temporal_coherence(labels):
    """
    Kiem tra tinh lien tuc thoi gian cua nhan cum.

    Neu cum co y nghia vat ly, cac gio cung cum co xu huong xuat hien
    theo chuoi lien tuc (khong ngau nhien). Kiem tra bang runs test.

    Parameters
    ----------
    labels : np.ndarray, shape (n_samples,)
        Nhan cum theo thu tu thoi gian.

    Returns
    -------
    result : dict
        n_runs         : int – So doan lien tuc (runs)
        expected_runs  : float – So runs ky vong neu ngau nhien
        z_statistic    : float – Z-score
        p_value        : float – p-value (2-sided)
        is_structured  : bool – True neu p < 0.05
        interpretation : str
        transition_matrix : np.ndarray – Ma tran chuyen trang thai
    """
    from scipy.stats import norm

    n = len(labels)
    if n < 3:
        return {
            'n_runs': 0, 'expected_runs': 0, 'z_statistic': 0,
            'p_value': 1.0, 'is_structured': False,
            'interpretation': 'Qua it du lieu',
            'transition_matrix': np.array([]),
        }

    unique_labels = np.unique(labels)
    k = len(unique_labels)

    # Dem so runs (doan lien tuc cung nhan)
    n_runs = 1
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            n_runs += 1

    # Dem so luong moi nhan
    counts = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
    ni_list = list(counts.values())

    # Ky vong va phuong sai cua so runs (Wald-Wolfowitz)
    sum_ni_sq = sum(ni ** 2 for ni in ni_list)
    expected_runs = 1 + (n ** 2 - sum_ni_sq) / n

    numerator = (sum_ni_sq ** 2 - sum(ni ** 3 for ni in ni_list))
    denominator = n ** 2 * (n - 1)

    if denominator > 0:
        var_runs = (numerator / denominator)
        # Them gia tri trung gian
        var_runs = max(var_runs, 1e-10)  # tranh chia 0
    else:
        var_runs = 1e-10

    std_runs = np.sqrt(var_runs)
    z_statistic = (n_runs - expected_runs) / std_runs if std_runs > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z_statistic)))

    # Ma tran chuyen trang thai (transition matrix)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    trans_matrix = np.zeros((k, k), dtype=int)
    for i in range(n - 1):
        from_idx = label_to_idx[labels[i]]
        to_idx = label_to_idx[labels[i + 1]]
        trans_matrix[from_idx, to_idx] += 1

    is_structured = p_value < 0.05

    if is_structured and z_statistic < 0:
        interpretation = (
            f"Co cau truc thoi gian (p={p_value:.4f}) – "
            f"cac gio cung cum co xu huong xuat hien lien tiep. "
            f"Cum co y nghia vat ly."
        )
    elif is_structured and z_statistic > 0:
        interpretation = (
            f"Co cau truc thoi gian (p={p_value:.4f}) – "
            f"cac nhan cum chuyen doi thuong xuyen hon ngau nhien. "
            f"Co the chi ra su dao dong nhanh."
        )
    else:
        interpretation = (
            f"Khong co cau truc thoi gian ro rang (p={p_value:.4f}) – "
            f"nhan cum phan bo gan nhu ngau nhien theo thoi gian."
        )

    return {
        'n_runs': n_runs,
        'expected_runs': float(expected_runs),
        'z_statistic': float(z_statistic),
        'p_value': float(p_value),
        'is_structured': is_structured,
        'interpretation': interpretation,
        'transition_matrix': trans_matrix,
        'unique_labels': unique_labels,
        'label_counts': counts,
    }


# ============================================================================
# 3. Run full stability analysis
# ============================================================================

def run_stability_analysis(X, labels_dict, n_clusters_dict,
                           n_iterations=100, sample_ratio=0.8,
                           save=True, result_dir=None):
    """
    Chay toan bo stability analysis cho nhieu phuong phap.

    Parameters
    ----------
    X : np.ndarray
        Du lieu da chuan hoa.
    labels_dict : dict
        {method_name: labels_array} – vd: {'HAC': labels_hac, 'GMM': labels_gmm}
    n_clusters_dict : dict
        {method_name: n_clusters} – vd: {'HAC': 4, 'GMM': 4}
    n_iterations : int
    sample_ratio : float
    save : bool
    result_dir : str

    Returns
    -------
    stability_results : dict
        {method_name: {bootstrap: ..., temporal: ...}}
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS")
    print("=" * 60)

    stability_results = {}

    for method_name, labels in labels_dict.items():
        n_clusters = n_clusters_dict.get(method_name, len(np.unique(labels)))
        method_key = method_name.lower().replace(' ', '_')

        # Xac dinh loai thuat toan
        if 'hac' in method_key or 'hierarch' in method_key:
            algo = 'hac'
        elif 'gmm' in method_key or 'gaussian' in method_key:
            algo = 'gmm'
        else:
            algo = 'hac'  # mac dinh

        print(f"\n--- {method_name} (k={n_clusters}, algo={algo}) ---")

        # Bootstrap stability
        print(f"  Bootstrap stability ({n_iterations} iterations)...")
        boot_result = bootstrap_stability(
            X, n_clusters=n_clusters, method=algo,
            n_iterations=n_iterations, sample_ratio=sample_ratio,
        )
        print(f"  ARI = {boot_result['ari_mean']:.3f} +/- {boot_result['ari_std']:.3f}")
        print(f"  >> {boot_result['interpretation']}")

        # Temporal coherence
        print(f"  Temporal coherence...")
        temp_result = temporal_coherence(labels)
        print(f"  Runs: {temp_result['n_runs']} "
              f"(ky vong: {temp_result['expected_runs']:.1f}), "
              f"p={temp_result['p_value']:.4f}")
        print(f"  >> {temp_result['interpretation']}")

        stability_results[method_name] = {
            'bootstrap': boot_result,
            'temporal': temp_result,
        }

    # Visualization
    plot_stability_results(stability_results, save=save, result_dir=result_dir)

    # In bang tom tat
    _print_summary(stability_results)

    return stability_results


def _print_summary(stability_results):
    """In bang tom tat stability analysis."""
    print("\n" + "=" * 70)
    print("TOM TAT STABILITY ANALYSIS")
    print("=" * 70)
    hdr = (f"{'Method':<22} {'ARI mean':>10} {'ARI std':>10} "
           f"{'Runs p':>10} {'On dinh?':>10}")
    print(hdr)
    print("-" * len(hdr))

    for method, res in stability_results.items():
        boot = res['bootstrap']
        temp = res['temporal']
        stable = "Co" if boot['ari_mean'] >= 0.5 else "Khong"
        print(f"{method:<22} {boot['ari_mean']:>10.3f} {boot['ari_std']:>10.3f} "
              f"{temp['p_value']:>10.4f} {stable:>10}")


# ============================================================================
# 4. Visualization
# ============================================================================

def plot_stability_results(stability_results, save=True, result_dir=None,
                           filename_prefix='S'):
    """
    Ve bieu do tong hop stability analysis.

    Tao 2 figure:
      S01_bootstrap_stability.png  – Histogram + boxplot ARI
      S02_temporal_coherence.png   – Transition matrix + runs bar
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    methods = list(stability_results.keys())

    # ======================================================================
    # Figure 1: Bootstrap Stability
    # ======================================================================
    n_methods = len(methods)
    fig1, axes1 = plt.subplots(1, n_methods + 1, figsize=(6 * (n_methods + 1), 5))
    if n_methods + 1 == 1:
        axes1 = [axes1]

    COLORS = ['steelblue', 'coral', 'mediumseagreen', 'orchid', 'gold']

    # Histograms cho tung method
    for i, method in enumerate(methods):
        ax = axes1[i]
        boot = stability_results[method]['bootstrap']
        scores = boot['ari_scores']

        if scores:
            ax.hist(scores, bins=20, color=COLORS[i % len(COLORS)],
                    alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(boot['ari_mean'], color='red', linestyle='--',
                       linewidth=2, label=f"Mean={boot['ari_mean']:.3f}")
            ax.axvline(boot['ari_median'], color='green', linestyle=':',
                       linewidth=2, label=f"Median={boot['ari_median']:.3f}")

            # Vung tham chieu
            ax.axvspan(0.75, 1.0, alpha=0.1, color='green', label='On dinh tot')
            ax.axvspan(0.0, 0.5, alpha=0.1, color='red', label='Khong on dinh')

        ax.set_title(f'{method}\nk={boot["n_clusters"]}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Adjusted Rand Index (ARI)')
        ax.set_ylabel('Tan suat')
        ax.set_xlim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Boxplot so sanh
    ax_box = axes1[-1]
    all_scores = []
    box_labels = []
    for method in methods:
        scores = stability_results[method]['bootstrap']['ari_scores']
        if scores:
            all_scores.append(scores)
            boot = stability_results[method]['bootstrap']
            box_labels.append(f"{method}\n(k={boot['n_clusters']})")

    if all_scores:
        bp = ax_box.boxplot(all_scores, labels=box_labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)

        ax_box.axhline(0.75, color='green', linestyle='--', alpha=0.5,
                       label='Nguong on dinh tot (0.75)')
        ax_box.axhline(0.5, color='orange', linestyle='--', alpha=0.5,
                       label='Nguong tuong doi (0.50)')
        ax_box.axhline(0.25, color='red', linestyle='--', alpha=0.5,
                       label='Nguong khong on dinh (0.25)')

    ax_box.set_title('So sanh Bootstrap Stability', fontsize=12, fontweight='bold')
    ax_box.set_ylabel('ARI')
    ax_box.set_ylim(-0.1, 1.1)
    ax_box.legend(fontsize=8)
    ax_box.grid(True, alpha=0.3)

    fig1.suptitle('Bootstrap Stability Analysis\n'
                  f'({stability_results[methods[0]]["bootstrap"]["n_iterations"]} iterations, '
                  f'80% sample)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, f'{filename_prefix}01_bootstrap_stability.png')
        fig1.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()

    # ======================================================================
    # Figure 2: Temporal Coherence
    # ======================================================================
    fig2, axes2 = plt.subplots(1, n_methods + 1, figsize=(6 * (n_methods + 1), 5))
    if n_methods + 1 == 1:
        axes2 = [axes2]

    # Transition matrix cho tung method
    for i, method in enumerate(methods):
        ax = axes2[i]
        temp = stability_results[method]['temporal']
        trans = temp['transition_matrix']
        unique_labels = temp.get('unique_labels', np.arange(trans.shape[0]))

        if trans.size > 0:
            # Normalize theo hang
            row_sums = trans.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            trans_norm = trans / row_sums

            im = ax.imshow(trans_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax, label='Xac suat chuyen')

            tick_labels = [f'C{lbl}' for lbl in unique_labels]
            ax.set_xticks(range(len(unique_labels)))
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(range(len(unique_labels)))
            ax.set_yticklabels(tick_labels)

            # Hien thi gia tri trong o
            for r in range(trans_norm.shape[0]):
                for c in range(trans_norm.shape[1]):
                    val = trans_norm[r, c]
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                            fontsize=10, fontweight='bold', color=color)

        p_str = f"p={temp['p_value']:.4f}"
        struct_str = "Co cau truc" if temp['is_structured'] else "Ngau nhien"
        ax.set_title(f'{method}\n{struct_str} ({p_str})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Cum tiep theo')
        ax.set_ylabel('Cum hien tai')

    # Runs bar chart
    ax_runs = axes2[-1]
    run_data = []
    expected_data = []
    method_labels = []
    p_values = []
    for method in methods:
        temp = stability_results[method]['temporal']
        run_data.append(temp['n_runs'])
        expected_data.append(temp['expected_runs'])
        p_values.append(temp['p_value'])
        method_labels.append(method)

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax_runs.bar(x - width / 2, run_data, width,
                        label='Runs thuc te', color='steelblue', alpha=0.8)
    bars2 = ax_runs.bar(x + width / 2, expected_data, width,
                        label='Runs ky vong\n(ngau nhien)', color='lightcoral', alpha=0.8)

    ax_runs.set_xticks(x)
    ax_runs.set_xticklabels(method_labels)
    ax_runs.set_ylabel('So runs')
    ax_runs.set_title('So sanh Runs: Thuc te vs Ky vong',
                      fontsize=12, fontweight='bold')
    ax_runs.legend()
    ax_runs.grid(True, alpha=0.3)

    # Ghi p-value len tren
    for i, (bar, p) in enumerate(zip(bars1, p_values)):
        sig = '*' if p < 0.05 else 'ns'
        ax_runs.text(bar.get_x() + width / 2, max(run_data[i], expected_data[i]) + 1,
                     f'p={p:.3f}\n{sig}', ha='center', va='bottom', fontsize=9)

    fig2.suptitle('Temporal Coherence Analysis\n'
                  '(Runs test – kiem tra cau truc thoi gian cua nhan cum)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, f'{filename_prefix}02_temporal_coherence.png')
        fig2.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()
