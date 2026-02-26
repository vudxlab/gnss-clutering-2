"""
Module tìm số cụm tối ưu bằng cách thử nhiều giá trị k
và tổng hợp kết quả bằng hệ thống bỏ phiếu (voting).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from . import config
from . import clustering as cl


def _save(fig, name, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"{name}.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"  [saved] {path}")


def find_optimal_clusters(data, data_scaled=None, k_range=None,
                          methods=None, save=True, result_dir=None):
    """
    Tìm số cụm tối ưu bằng cách chạy các thuật toán với nhiều giá trị k.

    Parameters
    ----------
    data : np.ndarray
        Dữ liệu đầu vào (thường là data_tsne).
    data_scaled : np.ndarray, optional
        Dữ liệu chuẩn hóa để TimeSeriesKMeans tính metrics.
    k_range : tuple (start, end)
        Khoảng k cần thử (end không bao gồm).
    methods : list of str
        Subset thuật toán: ['timeseries_kmeans', 'hac', 'dbscan', 'gmm'].

    Returns
    -------
    optimization_result : dict
    """
    if k_range is None:
        k_range = config.K_RANGE
    if methods is None:
        methods = ['timeseries_kmeans', 'hac', 'dbscan', 'gmm']
    if data_scaled is None:
        data_scaled = data

    print("TIM KIEM SO CUM TOI UU TOAN DIEN")
    print("=" * 70)

    k_values = list(range(k_range[0], k_range[1]))
    method_results = {m: {'k_values': [], 'silhouette': [], 'calinski': [], 'davies': []}
                      for m in ['timeseries_kmeans', 'hac', 'gmm', 'dbscan']}
    all_results = {}

    for k in tqdm(k_values, desc="Testing k values"):
        print(f"\nTesting k = {k}")
        print("-" * 40)
        k_results = {}

        if 'timeseries_kmeans' in methods:
            try:
                res, _ = cl.run_timeseries_kmeans(data, data_scaled=data_scaled, n_clusters=k)
                if res and res['silhouette'] > -1:
                    _append_method(method_results['timeseries_kmeans'], k, res)
                    k_results['timeseries_kmeans'] = res
                    print(f"  TimeSeriesKMeans: Sil={res['silhouette']:.3f}")
            except Exception as e:
                print(f"  TimeSeriesKMeans Error: {str(e)[:60]}")

        if 'hac' in methods:
            try:
                res = cl.run_hierarchical_clustering(data, n_clusters=k)
                if res and res['silhouette'] > -1:
                    _append_method(method_results['hac'], k, res)
                    k_results['hac'] = res
                    print(f"  HAC: Sil={res['silhouette']:.3f}")
            except Exception as e:
                print(f"  HAC Error: {str(e)[:60]}")

        if 'gmm' in methods:
            try:
                res = cl.run_gmm_clustering(data, n_clusters=k)
                if res and res['silhouette'] > -1:
                    _append_method(method_results['gmm'], k, res)
                    k_results['gmm'] = res
                    print(f"  GMM: Sil={res['silhouette']:.3f}")
            except Exception as e:
                print(f"  GMM Error: {str(e)[:60]}")

        if 'dbscan' in methods and k == k_values[0]:
            try:
                res = cl.run_dbscan_clustering(data)
                if res and res['silhouette'] > -1 and res['n_clusters'] > 1:
                    actual_k = res['n_clusters']
                    _append_method(method_results['dbscan'], actual_k, res)
                    k_results['dbscan'] = res
                    print(f"  DBSCAN: {actual_k} cum, Sil={res['silhouette']:.3f}")
            except Exception as e:
                print(f"  DBSCAN Error: {str(e)[:60]}")

        all_results[k] = k_results

    # Tìm k tốt nhất theo từng metric / method
    best_k_by_method = _find_best_k(method_results)

    # Voting
    k_votes = _vote(best_k_by_method)
    sorted_votes = sorted(k_votes.items(), key=lambda x: len(x[1]), reverse=True)

    print("\nVOTING RESULTS:")
    print("-" * 40)
    for k, votes in sorted_votes:
        print(f"k={k}: {len(votes)} votes - {votes}")

    recommended_k = sorted_votes[0][0] if sorted_votes else None
    vote_count = len(sorted_votes[0][1]) if sorted_votes else 0

    if recommended_k:
        print(f"\nK TOI UU DE XUAT: k = {recommended_k} ({vote_count} phieu)")

    create_optimization_plots(method_results, best_k_by_method, recommended_k, sorted_votes,
                              save=save, result_dir=result_dir)

    return {
        'recommended_k': recommended_k,
        'vote_count': vote_count,
        'voting_details': sorted_votes,
        'method_results': method_results,
        'best_k_by_method': best_k_by_method,
        'all_results': all_results,
        'tested_range': k_range,
        'methods_tested': methods,
    }


def _append_method(method_dict, k, res):
    method_dict['k_values'].append(k)
    method_dict['silhouette'].append(res['silhouette'])
    method_dict['calinski'].append(res['calinski_harabasz'])
    method_dict['davies'].append(res['davies_bouldin'])


def _find_best_k(method_results):
    best = {}
    for method, results in method_results.items():
        if len(results['k_values']) == 0:
            continue
        best[method] = {
            'best_silhouette_k': results['k_values'][int(np.argmax(results['silhouette']))],
            'best_calinski_k': results['k_values'][int(np.argmax(results['calinski']))],
            'best_davies_k': results['k_values'][int(np.argmin(results['davies']))],
            'silhouette_scores': results['silhouette'],
            'calinski_scores': results['calinski'],
            'davies_scores': results['davies'],
            'k_values': results['k_values'],
        }
    return best


def _vote(best_k_by_method):
    k_votes = {}
    for method, best_ks in best_k_by_method.items():
        for metric_key, k_val in best_ks.items():
            if metric_key.endswith('_k'):
                k_votes.setdefault(k_val, []).append(f"{method}_{metric_key}")
    return k_votes


def create_optimization_plots(method_results, best_k_by_method, recommended_k,
                              voting_details=None, save=True, result_dir=None):
    """Tạo biểu đồ tổng quan kết quả optimization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ['blue', 'red', 'green', 'orange']
    method_names = ['timeseries_kmeans', 'hac', 'gmm', 'dbscan']

    def _plot_metric(ax, metric_key, title, ylabel):
        for i, method in enumerate(method_names):
            mr = method_results[method]
            if len(mr['k_values']) > 0:
                ax.plot(mr['k_values'], mr[metric_key], 'o-',
                        color=colors[i], label=method.upper(), linewidth=2, markersize=6)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_metric(axes[0, 0], 'silhouette', 'Silhouette Scores vs K', 'Silhouette Score')
    _plot_metric(axes[0, 1], 'calinski', 'Calinski-Harabasz Scores vs K', 'Calinski-Harabasz Score')
    _plot_metric(axes[0, 2], 'davies', 'Davies-Bouldin Scores vs K\n(Lower is Better)', 'Davies-Bouldin Score')

    # Best K bar chart
    ax = axes[1, 0]
    metrics = ['silhouette', 'calinski', 'davies']
    x_pos = np.arange(len(method_names))
    width = 0.25
    for i, metric in enumerate(metrics):
        best_ks = [best_k_by_method[m][f'best_{metric}_k'] if m in best_k_by_method else 0
                   for m in method_names]
        ax.bar(x_pos + i * width, best_ks, width, label=f'Best {metric.capitalize()}', alpha=0.8)
    ax.set_title('Best K by Method and Metric', fontsize=14, fontweight='bold')
    ax.set_xlabel('Methods')
    ax.set_ylabel('Optimal K')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([m.upper() for m in method_names])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Performance heatmap
    ax = axes[1, 1]
    perf_matrix, method_labels = [], []
    for method in method_names:
        mr = method_results[method]
        if len(mr['k_values']) == 0:
            continue
        if recommended_k and recommended_k in mr['k_values']:
            idx = mr['k_values'].index(recommended_k)
            row = [mr['silhouette'][idx], mr['calinski'][idx] / 1000, 1 - mr['davies'][idx]]
        else:
            row = [max(mr['silhouette']), max(mr['calinski']) / 1000, 1 - min(mr['davies'])]
        perf_matrix.append(row)
        method_labels.append(method.upper())
    if perf_matrix:
        im = ax.imshow(perf_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Silhouette', 'Calinski/1000', '1-Davies'])
        ax.set_yticks(range(len(method_labels)))
        ax.set_yticklabels(method_labels)
        ax.set_title('Method Performance Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)

    # Voting results
    ax = axes[1, 2]
    if voting_details:
        top5 = voting_details[:5]
        k_vals = [item[0] for item in top5]
        vote_counts = [len(item[1]) for item in top5]
        bars = ax.bar(range(len(k_vals)), vote_counts, color='skyblue', alpha=0.8)
        ax.set_title('Top 5 K Values by Vote Count', fontsize=14, fontweight='bold')
        ax.set_xlabel('K Values')
        ax.set_ylabel('Number of Votes')
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels([f'k={k}' for k in k_vals])
        for bar, count in zip(bars, vote_counts):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save:
        if result_dir is None:
            result_dir = config.RESULT_DIR
        _save(fig, '14_optimal_k_analysis', result_dir)
    plt.show()
