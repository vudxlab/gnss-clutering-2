"""
Module visualization - bao gom tat ca bieu do trong notebook Clustering_GNSS_3e.

Ham luu hinh:  _save(fig, name, result_dir)
Moi ham ve deu nhan tham so:
    save       : bool, mac dinh True
    result_dir : str, mac dinh config.RESULT_DIR
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from . import config


# ============================================================================
# Tien ich luu hinh
# ============================================================================

def _save(fig, name, result_dir):
    """Luu fig vao result_dir/<name>.png."""
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"{name}.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"  [saved] {path}")


# ============================================================================
# Section 2 – Du lieu theo ngay
# ============================================================================

def plot_daily_heatmap(daily_matrix, unique_dates, sample_interval=60,
                       save=True, result_dir=None):
    """
    Cell 7 – Heatmap tong quan ma tran ngay x giay (lay mau moi sample_interval giay).
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    sampled = daily_matrix[:, ::sample_interval]
    # Nhan thoi gian (gio:phut)
    step_min = sample_interval // 60 if sample_interval >= 60 else 1
    time_labels = [f"{h:02d}:{m:02d}"
                   for h in range(24)
                   for m in range(0, 60, max(1, step_min))]

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(sampled, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Toa do thang dung (m)')
    ax.set_title(
        f'Du lieu GNSS theo thoi gian trong ngay (mau moi {sample_interval}s)\n'
        'Vung toi = du lieu thieu',
        fontsize=16, fontweight='bold'
    )
    ax.set_xlabel('Thoi gian trong ngay', fontsize=12)
    ax.set_ylabel('Ngay', fontsize=12)

    x_ticks = list(range(0, len(time_labels), 120))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([time_labels[i] for i in x_ticks], rotation=45)
    ax.set_yticks(range(len(unique_dates)))
    ax.set_yticklabels([str(d) for d in unique_dates])

    plt.tight_layout()
    if save:
        _save(fig, '01_daily_heatmap', result_dir)
    plt.show()


def plot_daily_timeseries(daily_matrix, unique_dates,
                          save=True, result_dir=None):
    """
    Cell 8 – Mot cot subplot, moi hang la 1 ngay, ve chuoi h_Coord theo gio.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    n = len(unique_dates)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n))
    if n == 1:
        axes = [axes]

    for i, date in enumerate(unique_dates):
        data = daily_matrix[i, :]
        valid = ~np.isnan(data)
        t = np.arange(len(data))[valid] / 3600
        v = data[valid]

        if len(v) > 0:
            axes[i].plot(t, v, 'b-', linewidth=0.8, alpha=0.8)
            axes[i].scatter(t, v, c='red', s=0.3, alpha=0.4)

        axes[i].set_title(str(date), fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Gio')
        axes[i].set_ylabel('h_Coord (m)')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 24)

        if len(v) > 0:
            axes[i].text(0.02, 0.98, f'So diem: {len(v):,}',
                         transform=axes[i].transAxes, fontsize=10,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    fig.suptitle('Toa do thang dung theo thoi gian cho tung ngay',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save:
        _save(fig, '02_daily_timeseries', result_dir)
    plt.show()


# ============================================================================
# Section 2 – Du lieu theo gio (Cells 12, 14, 15, 16, 17)
# ============================================================================

def plot_hourly_heatmap(hourly_matrix, valid_hours_info, hourly_info_df=None,
                        sample_interval=60, save=True, result_dir=None):
    """
    Cell 12 (phan 1) – Heatmap ma tran theo gio + histogram ty le thieu + bar so gio hop le.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    # --- Bieu do 1: Heatmap ---
    sampled = hourly_matrix[:, ::sample_interval]
    fig1, ax1 = plt.subplots(figsize=(18, 12))
    im = ax1.imshow(sampled, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax1, label='Toa do thang dung (m)')

    n_removed = (len(hourly_info_df) - len(valid_hours_info)) if hourly_info_df is not None else 0
    ax1.set_title(
        f'Ma tran du lieu theo gio ({len(hourly_matrix)} gio hop le)'
        + (f'\nDa loai bo {n_removed} gio co >0% du lieu thieu' if n_removed else ''),
        fontsize=16, fontweight='bold'
    )
    ax1.set_xlabel('Phut trong gio (mau moi phut)', fontsize=12)
    ax1.set_ylabel('Chi so gio (theo thu tu thoi gian)', fontsize=12)

    x_ticks = list(range(0, 60, 5))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{i:02d}' for i in x_ticks])

    y_step = max(1, len(hourly_matrix) // 20)
    y_ticks = list(range(0, len(hourly_matrix), y_step))
    y_labels = [valid_hours_info.iloc[i]['datetime'] for i in y_ticks]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=8)

    plt.tight_layout()
    if save:
        _save(fig1, '03_hourly_heatmap', result_dir)
    plt.show()

    # --- Bieu do 2: Histogram ty le thieu + bar so gio hop le theo ngay ---
    if hourly_info_df is not None:
        missing_threshold = config.MISSING_THRESHOLD
        fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].hist(hourly_info_df['missing_percentage'], bins=30, alpha=0.7,
                     color='lightcoral', label=f'Tat ca ({len(hourly_info_df)} gio)')
        axes[0].hist(valid_hours_info['missing_percentage'], bins=30, alpha=0.7,
                     color='lightblue', label=f'Hop le ({len(valid_hours_info)} gio)')
        axes[0].axvline(missing_threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Nguong {missing_threshold}%')
        axes[0].set_xlabel('Ty le du lieu thieu (%)')
        axes[0].set_ylabel('So gio')
        axes[0].set_title('Phan bo ty le du lieu thieu theo gio')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        daily_stats = valid_hours_info.groupby('date').size()
        axes[1].bar(range(len(daily_stats)), daily_stats.values, color='skyblue', alpha=0.7)
        axes[1].set_xlabel('Ngay')
        axes[1].set_ylabel('So gio hop le')
        axes[1].set_title('So gio hop le theo tung ngay')
        axes[1].set_xticks(range(len(daily_stats)))
        axes[1].set_xticklabels([str(d) for d in daily_stats.index], rotation=45)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            _save(fig2, '04_hourly_filter_stats', result_dir)
        plt.show()


def plot_hourly_overview(hourly_matrix, valid_hours_info,
                         save=True, result_dir=None):
    """
    Cell 14 – 4-subplot tong quan: 3D scatter, histogram gia tri, boxplot theo ngay,
    line plot 5 gio mau.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    fig = plt.figure(figsize=(16, 12))

    # --- Subplot 1: 3D scatter ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    sr = slice(0, len(hourly_matrix), max(1, len(hourly_matrix) // 20))
    sc = slice(0, 3600, 300)
    ri, ci = np.meshgrid(np.arange(len(hourly_matrix))[sr],
                         np.arange(3600)[sc], indexing='ij')
    sd = hourly_matrix[sr, sc]
    vm = ~np.isnan(sd)
    sc3 = ax1.scatter(ri[vm], ci[vm], sd[vm], c=sd[vm], cmap='viridis', alpha=0.6, s=1)
    ax1.set_xlabel('Chi so gio')
    ax1.set_ylabel('Giay trong gio')
    ax1.set_zlabel('h_Coord (m)')
    ax1.set_title('Bieu do 3D du lieu theo gio')

    # --- Subplot 2: Histogram gia tri ---
    ax2 = fig.add_subplot(2, 2, 2)
    valid_data = hourly_matrix[~np.isnan(hourly_matrix)]
    ax2.hist(valid_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Gia tri h_Coord (m)')
    ax2.set_ylabel('Tan suat')
    ax2.set_title(f'Phan bo gia tri toa do thang dung\n({len(valid_data):,} diem)')
    ax2.grid(True, alpha=0.3)

    # --- Subplot 3: Box plot theo ngay ---
    ax3 = fig.add_subplot(2, 2, 3)
    box_data, box_labels = [], []
    for date in sorted(valid_hours_info['date'].unique()):
        date_mask = valid_hours_info['date'] == date
        date_idxs = valid_hours_info[date_mask].index.tolist()
        vals = []
        vi_list = list(valid_hours_info.index)
        for idx in date_idxs:
            row_i = vi_list.index(idx)
            hv = hourly_matrix[row_i, :]
            vals.extend(hv[~np.isnan(hv)])
        if vals:
            box_data.append(vals)
            box_labels.append(str(date))
    if box_data:
        ax3.boxplot(box_data, labels=box_labels)
    ax3.set_xlabel('Ngay')
    ax3.set_ylabel('h_Coord (m)')
    ax3.set_title('Box plot theo tung ngay')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # --- Subplot 4: 5 gio mau ---
    ax4 = fig.add_subplot(2, 2, 4)
    np.random.seed(42)
    sample_hrs = np.random.choice(len(hourly_matrix),
                                  size=min(5, len(hourly_matrix)), replace=False)
    for hour_idx in sample_hrs:
        hd = hourly_matrix[hour_idx, :]
        vm2 = ~np.isnan(hd)
        if np.sum(vm2) > 0:
            t = np.arange(3600)[vm2] / 60
            info = valid_hours_info.iloc[hour_idx]
            ax4.plot(t, hd[vm2], label=info['datetime'], alpha=0.7, linewidth=1)
    ax4.set_xlabel('Phut trong gio')
    ax4.set_ylabel('h_Coord (m)')
    ax4.set_title('Du lieu mau cua 5 gio')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        _save(fig, '05_hourly_overview_4subplots', result_dir)
    plt.show()


def plot_hourly_analysis(hourly_matrix, valid_hours_info, hourly_info_df=None,
                         save=True, result_dir=None):
    """
    Cell 15 – 2x2: heatmap chi tiet (30s), so sanh truoc/sau loc, violin, weekday bar.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # --- [0,0] Heatmap chi tiet (30s) ---
    s30 = hourly_matrix[:, ::30]
    im1 = axes[0, 0].imshow(s30, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title(
        f'Heatmap chi tiet ma tran theo gio\n({len(hourly_matrix)} gio hop le, mau moi 30 giay)',
        fontsize=14, fontweight='bold'
    )
    axes[0, 0].set_xlabel('Thoi gian trong gio (mau moi 30 giay)')
    axes[0, 0].set_ylabel('Chi so gio')
    xt = list(range(0, 120, 10))
    axes[0, 0].set_xticks(xt)
    axes[0, 0].set_xticklabels([f'{i//2:02d}:{(i%2)*30:02d}' for i in xt], rotation=45)
    plt.colorbar(im1, ax=axes[0, 0], label='h_Coord (m)')

    # --- [0,1] So sanh truoc/sau loc ---
    ax01 = axes[0, 1]
    if hourly_info_df is not None:
        categories = ['Truoc loc', 'Sau loc']
        total_hrs = [len(hourly_info_df), len(valid_hours_info)]
        miss_rates = [hourly_info_df['missing_percentage'].mean(),
                      valid_hours_info['missing_percentage'].mean()]
        bars1 = ax01.bar(categories, total_hrs, color=['lightcoral', 'lightblue'], alpha=0.7)
        ax01.set_ylabel('So gio', color='blue')
        ax01.set_title('So sanh truoc va sau loc', fontsize=14, fontweight='bold')
        for bar, val in zip(bars1, total_hrs):
            ax01.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                      str(val), ha='center', va='bottom', fontweight='bold')
        ax01b = ax01.twinx()
        ax01b.plot(categories, miss_rates, 'ro-', linewidth=3, markersize=8)
        ax01b.set_ylabel('Ty le thieu du lieu (%)', color='red')
        ax01b.tick_params(axis='y', labelcolor='red')
        for i, rate in enumerate(miss_rates):
            ax01b.text(i, rate + 0.3, f'{rate:.1f}%', ha='center', va='bottom',
                       fontweight='bold', color='red')
    else:
        ax01.text(0.5, 0.5, 'Khong co hourly_info_df',
                  ha='center', va='center', transform=ax01.transAxes)
        ax01.set_title('So sanh truoc va sau loc', fontsize=14, fontweight='bold')

    # --- [1,0] Violin plot theo 8 khung thoi gian ---
    n_bins, bin_size = 8, 3600 // 8
    vdata, vlabels = [], []
    for b in range(n_bins):
        s0, s1 = b * bin_size, (b + 1) * bin_size
        vals = []
        for row in hourly_matrix:
            v = row[s0:s1]
            vals.extend(v[~np.isnan(v)])
        if vals:
            vdata.append(vals)
            vlabels.append(f'{s0//60:02d}-{s1//60:02d}')
    if vdata:
        axes[1, 0].violinplot(vdata, positions=range(len(vdata)), showmeans=True)
    axes[1, 0].set_xlabel('Khung thoi gian (phut)')
    axes[1, 0].set_ylabel('h_Coord (m)')
    axes[1, 0].set_title('Phan bo gia tri theo thoi gian trong gio\n(Violin Plot)',
                          fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(vlabels)))
    axes[1, 0].set_xticklabels(vlabels)
    axes[1, 0].grid(True, alpha=0.3)

    # --- [1,1] Chat luong du lieu theo ngay trong tuan ---
    vhi = valid_hours_info.copy()
    vhi['weekday'] = pd.to_datetime(vhi['date']).dt.day_name()
    wd_stats = vhi.groupby('weekday')['missing_percentage'].agg(['mean', 'std', 'count'])
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    present = [d for d in order if d in wd_stats.index]
    if present:
        means = [wd_stats.loc[d, 'mean'] for d in present]
        stds  = [wd_stats.loc[d, 'std'].fillna(0) if hasattr(wd_stats.loc[d, 'std'], 'fillna')
                 else (wd_stats.loc[d, 'std'] if not np.isnan(wd_stats.loc[d, 'std']) else 0)
                 for d in present]
        counts = [int(wd_stats.loc[d, 'count']) for d in present]
        bars2 = axes[1, 1].bar(present, means, yerr=stds, capsize=5,
                               color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        for bar, cnt in zip(bars2, counts):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'n={cnt}', ha='center', va='bottom', fontsize=9)
    axes[1, 1].set_xlabel('Ngay trong tuan')
    axes[1, 1].set_ylabel('Ty le thieu du lieu TB (%)')
    axes[1, 1].set_title('Chat luong du lieu theo ngay trong tuan',
                          fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        _save(fig, '06_hourly_analysis_2x2', result_dir)
    plt.show()


def plot_sample_hours(hourly_matrix, valid_hours_info, n_samples=6,
                      save=True, result_dir=None):
    """
    Cell 16 – 3x2 subplots: ve gio co du lieu tot nhat cho moi ngay (toi da n_samples).
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    sample_indices, selected_dt = [], []
    vi_list = list(valid_hours_info.index)
    for date in sorted(valid_hours_info['date'].unique())[:n_samples]:
        rows = valid_hours_info[valid_hours_info['date'] == date]
        if len(rows):
            best = rows.loc[rows['missing_percentage'].idxmin()]
            row_i = vi_list.index(best.name)
            sample_indices.append(row_i)
            selected_dt.append(best['datetime'])

    n_rows = math.ceil(len(sample_indices) / 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, (hidx, dtstr) in enumerate(zip(sample_indices, selected_dt)):
        hd = hourly_matrix[hidx, :]
        t = np.arange(3600) / 60
        vm = ~np.isnan(hd)

        axes[i].plot(t, hd, color='lightgray', alpha=0.5, linewidth=0.5, label='Tat ca')
        axes[i].plot(t[vm], hd[vm], color='blue', linewidth=1, label='Du lieu hop le')
        axes[i].scatter(t[vm], hd[vm], color='red', s=0.5, alpha=0.3)

        axes[i].set_title(dtstr, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Phut trong gio')
        axes[i].set_ylabel('h_Coord (m)')
        axes[i].grid(True, alpha=0.3)

        vc = int(np.sum(vm))
        mv = float(np.mean(hd[vm])) if vc > 0 else 0.0
        mp = (3600 - vc) / 3600 * 100
        axes[i].text(0.02, 0.98,
                     f'Diem hop le: {vc}/3600\nTrung binh: {mv:.3f}m\nThieu: {mp:.1f}%',
                     transform=axes[i].transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # An subplot thua
    for j in range(len(sample_indices), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Du lieu chi tiet cua {len(sample_indices)} gio mau (gio tot nhat moi ngay)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save:
        _save(fig, '07_sample_hours', result_dir)
    plt.show()


def plot_first_n_hours(hourly_matrix, valid_hours_info, n=20,
                       save=True, result_dir=None):
    """
    Cell 17 – Heatmap N gio dau tien + luoi subplot line plot cho moi gio.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    n = min(n, len(hourly_matrix))
    first_n = hourly_matrix[:n, :]
    first_info = valid_hours_info.head(n)

    # --- Bieu do 1: Heatmap ---
    sampled = first_n[:, ::60]
    fig1, ax1 = plt.subplots(figsize=(18, 10))
    im = ax1.imshow(sampled, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax1, label='h_Coord (m)')
    ax1.set_title(f'Heatmap {n} gio dau tien trong ma tran\n(Moi hang = 1 gio, mau moi phut)',
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Phut trong gio')
    ax1.set_ylabel(f'Chi so gio (0-{n-1})')
    xt = list(range(0, 60, 5))
    ax1.set_xticks(xt)
    ax1.set_xticklabels([f'{i:02d}' for i in xt])
    y_labels = [f"{i}: {row['datetime']}" for i, (_, row) in enumerate(first_info.iterrows())]
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(y_labels, fontsize=8)
    plt.tight_layout()
    if save:
        _save(fig1, f'08_first_{n}_hours_heatmap', result_dir)
    plt.show()

    # --- Bieu do 2: Luoi subplots ---
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    t = np.arange(3600) / 60

    for i in range(n):
        hd = first_n[i, :]
        info = first_info.iloc[i]
        vm = ~np.isnan(hd)
        ax = axes[i]

        if np.any(vm):
            diff = np.diff(np.concatenate(([False], vm, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends   = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                ax.plot(t[s:e], hd[s:e], 'b-', linewidth=0.8, alpha=0.8)
            ax.scatter(t[vm], hd[vm], c='red', s=0.3, alpha=0.6)

        ax.set_title(f"#{i}: {info['datetime']}", fontsize=9, fontweight='bold')
        ax.set_xlabel('Phut')
        ax.set_ylabel('h_Coord (m)')
        ax.grid(True, alpha=0.3)

        vc = int(np.sum(vm))
        mv = float(np.mean(hd[vm])) if vc > 0 else 0.0
        mp = (3600 - vc) / 3600 * 100
        ax.text(0.02, 0.98, f'Valid: {vc}/3600\nMean: {mv:.3f}m\nMiss: {mp:.1f}%',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig2.suptitle(f'Chi tiet {n} gio dau tien (moi subplot = 1 gio)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        _save(fig2, f'09_first_{n}_hours_grid', result_dir)
    plt.show()


# ============================================================================
# Section 3 – Tien xu ly (Cells 24, 25, 28)
# ============================================================================

def plot_z_comparison(original_z, filtered_z, row_index,
                      save=True, result_dir=None):
    """
    Cell 24 – So sanh mot hang du lieu truoc/sau Hampel filter.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(original_z, label='Truoc Hampel', color='blue', linewidth=1)
    ax.plot(filtered_z, label='Sau Hampel', color='red', linestyle='--', linewidth=1)
    ax.set_title(f'Du lieu do cao – hang {row_index}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Thoi gian (giay)')
    ax.set_ylabel('Do cao H (m)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save:
        _save(fig, f'10_hampel_compare_row{row_index:03d}', result_dir)
    plt.show()


def plot_z_comparison_batch(hourly_matrix, hampel_data, n=25,
                             save=True, result_dir=None):
    """
    Cell 24 – Ve n hang dau tien, moi hang 1 figure nho (so sanh truoc/sau Hampel).
    Luu thanh 1 figure luoi thay vi n figure rieng le.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    n = min(n, len(hourly_matrix))
    n_cols = 5
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n):
        axes[i].plot(hourly_matrix[i], color='blue', linewidth=0.6, alpha=0.7, label='Goc')
        axes[i].plot(hampel_data[i], color='red', linewidth=0.6, linestyle='--',
                     alpha=0.7, label='Hampel')
        axes[i].set_title(f'Hang {i}', fontsize=8)
        axes[i].grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    axes[0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'So sanh {n} hang dau: du lieu goc vs sau Hampel filter',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        _save(fig, '10_hampel_compare_batch', result_dir)
    plt.show()


def plot_multiple_series(data, n_cols=5, row_height=2, fig_width=20, title='',
                         save=True, result_dir=None, filename='multiple_series'):
    """
    Cells 25, 28 – Ve nhieu chuoi du lieu 2D tren luoi subplot.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    n = len(data)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, row_height * n_rows))
    axes = axes.flatten()

    for i in range(n):
        axes[i].plot(data[i], linewidth=0.7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save:
        _save(fig, filename, result_dir)
    plt.show()


# ============================================================================
# Section 4 – Ket qua phan cum (Cells 53, 56)
# ============================================================================

def plot_clustering_results(clustering_results, data_2d,
                             save=True, result_dir=None):
    """
    Cell 53/54 – Scatter plot 2D (t-SNE) + bar chart 3 metrics.

    Returns
    -------
    best_labels : np.ndarray
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # --- Scatter plots ---
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, (method, res) in enumerate(clustering_results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        for j, lbl in enumerate(np.unique(res['labels'])):
            mask = res['labels'] == lbl
            if lbl == -1:
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                           c=COLORS[j % len(COLORS)], s=60, alpha=0.7, label=f'Cluster {lbl}')
        ax.set_title(f'{method}\nSilhouette: {res["silhouette"]:.3f}',
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    for j in range(len(clustering_results), 4):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save:
        _save(fig1, '15_clustering_scatter', result_dir)
    plt.show()

    # --- Bar chart metrics ---
    methods = list(clustering_results.keys())
    sil  = [clustering_results[m]['silhouette'] for m in methods]
    cal  = [clustering_results[m]['calinski_harabasz'] for m in methods]
    dav  = [clustering_results[m]['davies_bouldin'] for m in methods]

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    for ax, scores, ttl, color in [
        (axes2[0], sil,  'Silhouette Score\n(Cang cao cang tot)', 'skyblue'),
        (axes2[1], cal,  'Calinski-Harabasz Score\n(Cang cao cang tot)', 'lightgreen'),
        (axes2[2], dav,  'Davies-Bouldin Score\n(Cang thap cang tot)', 'salmon'),
    ]:
        bars = ax.bar(methods, scores, color=color, alpha=0.8)
        ax.set_title(ttl, fontweight='bold')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        for bar, sc in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save:
        _save(fig2, '16_clustering_metrics', result_dir)
    plt.show()

    # Tinh diem tong hop
    norm = {}
    for m in methods:
        s = clustering_results[m]['silhouette']
        c = clustering_results[m]['calinski_harabasz'] / max(cal) if max(cal) > 0 else 0
        d = 1 - clustering_results[m]['davies_bouldin'] / max(dav) if max(dav) > 0 else 0
        norm[m] = s * 0.4 + c * 0.3 + d * 0.3

    best = max(norm, key=norm.get)
    print(f"\nTHUAT TOAN TOT NHAT TONG HOP: {best} (Score: {norm[best]:.4f})")
    return clustering_results[best]['labels']


def plot_clusters_lineplot_all_methods(clustering_results, data_scaled,
                                        save=True, result_dir=None):
    """
    Cell 56/57 – Line plot tung cum (mean +- std) cho tung phuong phap.
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR

    print(f"\nLINE PLOT TUNG CUM – {data_scaled.shape[0]} CHUOI DU LIEU")
    print("-" * 70)

    t = np.arange(data_scaled.shape[1])

    for method_name, res in clustering_results.items():
        if res['silhouette'] == -1:
            continue

        labels = res['labels']
        unique_lbl = np.unique(labels)
        n_cls = len(unique_lbl)

        fig, axes = plt.subplots(n_cls, 1, figsize=(16, 4 * n_cls))
        if n_cls == 1:
            axes = [axes]

        fig.suptitle(f'{method_name} – Phan Tich Tung Cum ({data_scaled.shape[0]} chuoi)',
                     fontsize=16, fontweight='bold', y=1.0)

        cmap = plt.cm.Set3(np.linspace(0, 1, n_cls))

        for i, cid in enumerate(unique_lbl):
            ax = axes[i]
            idxs = np.where(labels == cid)[0]
            if len(idxs) == 0:
                continue

            for idx in idxs:
                s = data_scaled[idx, :]
                vm = ~np.isnan(s)
                if np.sum(vm):
                    ax.plot(t[vm], s[vm], alpha=0.4, linewidth=0.5, color=cmap[i])

            cdata = data_scaled[idxs, :]
            mean  = np.nanmean(cdata, axis=0)
            std   = np.nanstd(cdata, axis=0)
            vm2   = ~np.isnan(mean)

            if np.sum(vm2):
                ax.plot(t[vm2], mean[vm2], color='black', linewidth=3,
                        alpha=0.9, label='Cluster Mean')
                ax.fill_between(t[vm2], (mean - std)[vm2], (mean + std)[vm2],
                                alpha=0.2, color='gray', label='±1 Std Dev')

            lbl_txt = f'Cluster {cid}' if cid != -1 else 'Noise Points'
            ax.set_title(f'{lbl_txt} ({len(idxs)} chuoi)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Points')
            ax.set_ylabel('Standardized Value')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            ax.text(0.02, 0.98,
                    f'Mean: {np.nanmean(mean):.3f}\nStd: {np.nanmean(std):.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

        plt.tight_layout()
        if save:
            safe_name = method_name.replace(' ', '_').lower()
            _save(fig, f'17_lineplot_{safe_name}', result_dir)
        plt.show()

        print(f"\n{method_name}:")
        for cid in unique_lbl:
            cnt = int(np.sum(labels == cid))
            lbl_txt = f'Cluster {cid}' if cid != -1 else 'Noise'
            print(f"  {lbl_txt}: {cnt} chuoi ({cnt/len(labels)*100:.1f}%)")

    print(f"\nHoan thanh visualization {len(clustering_results)} thuat toan!")
