"""
Deep Learning Clustering – Method 3

Method 3A: Conv1D Autoencoder
  - Encode chuoi thoi gian 360 diem → latent vector 16-32D
  - Cluster tren khong gian latent (HAC, GMM, HDBSCAN, Ensemble)

Method 3B: Moment Foundation Model (TODO)
  - Zero-shot embeddings tu pre-trained model
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from . import config


# ============================================================================
# Conv1D Autoencoder Architecture
# ============================================================================

class Conv1DAutoencoder(nn.Module):
    """
    Conv1D Autoencoder cho chuoi thoi gian GNSS.

    Encoder: Conv1D layers giam chieu dan → flatten → linear → latent
    Decoder: linear → unflatten → ConvTranspose1D layers tang chieu → output

    Input shape:  (batch, 1, seq_len)  e.g. (batch, 1, 360)
    Latent shape: (batch, latent_dim)  e.g. (batch, 32)
    Output shape: (batch, 1, seq_len)
    """

    def __init__(self, seq_len=360, latent_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),   # -> (16, 180)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # -> (32, 90)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # -> (64, 45)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 23)
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Calculate flattened size after conv
        self._conv_out_len = self._get_conv_output_len(seq_len)
        self._flat_size = 128 * self._conv_out_len

        self.encoder_fc = nn.Sequential(
            nn.Linear(self._flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self._flat_size),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def _get_conv_output_len(self, seq_len):
        """Tinh kich thuoc output sau encoder conv."""
        x = torch.zeros(1, 1, seq_len)
        x = self.encoder_conv(x)
        return x.shape[2]

    def encode(self, x):
        """Encode input → latent vector."""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.encoder_fc(h)
        return z

    def decode(self, z):
        """Decode latent → reconstructed signal."""
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 128, self._conv_out_len)
        x_recon = self.decoder_conv(h)
        # Trim or pad to match original seq_len
        if x_recon.shape[2] > self.seq_len:
            x_recon = x_recon[:, :, :self.seq_len]
        elif x_recon.shape[2] < self.seq_len:
            pad = self.seq_len - x_recon.shape[2]
            x_recon = nn.functional.pad(x_recon, (0, pad))
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ============================================================================
# Training
# ============================================================================

def train_autoencoder(data, seq_len=360, latent_dim=32, epochs=100,
                      batch_size=32, lr=1e-3, device=None):
    """
    Train Conv1D Autoencoder.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, seq_len)
        Du lieu da tien xu ly (khong NaN).
    seq_len : int
    latent_dim : int
    epochs : int
    batch_size : int
    lr : float
    device : str or None

    Returns
    -------
    model : Conv1DAutoencoder (trained)
    latent_vectors : np.ndarray, shape (n_samples, latent_dim)
    train_losses : list
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"  Device: {device}")

    # Chuan hoa du lieu
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Chuyen sang tensor
    X = torch.FloatTensor(data_scaled).unsqueeze(1)  # (n, 1, seq_len)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = Conv1DAutoencoder(seq_len=seq_len, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    train_losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_recon, _ = model(batch_x)
            loss = criterion(x_recon, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(dataset)
        train_losses.append(epoch_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss = {epoch_loss:.6f}")

    # Extract latent vectors
    model.eval()
    with torch.no_grad():
        X_all = X.to(device)
        latent_vectors = model.encode(X_all).cpu().numpy()

    print(f"  Latent vectors shape: {latent_vectors.shape}")
    return model, latent_vectors, train_losses, scaler


# ============================================================================
# Clustering on latent space
# ============================================================================

def cluster_latent_space(latent_vectors, n_clusters=2):
    """
    Phan cum tren khong gian latent.

    Neu so chieu > so mau, giam chieu bang PCA truoc khi cluster.

    Returns
    -------
    clustering_results : dict
    Z : np.ndarray (standardized, possibly PCA-reduced)
    """
    import hdbscan
    from sklearn.decomposition import PCA

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

    # Chuan hoa latent vectors
    scaler = StandardScaler()
    Z = scaler.fit_transform(latent_vectors)

    # Giam chieu bang PCA neu so chieu qua cao so voi so mau
    n_samples, n_dim = Z.shape
    if n_dim > n_samples // 2:
        n_comp = min(n_samples // 2, 50)
        print(f"    PCA reduction: {n_dim}D -> {n_comp}D (n_samples={n_samples})")
        pca = PCA(n_components=n_comp, random_state=config.SEED)
        Z = pca.fit_transform(Z)
        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"    PCA explained variance: {explained:.1f}%")

    clustering_results = {}

    # HAC
    lbl_hac = AgglomerativeClustering(
        n_clusters=n_clusters, linkage='ward'
    ).fit_predict(Z)
    sil, cal, dav = _metrics(Z, lbl_hac)
    clustering_results['HAC'] = {
        'labels': lbl_hac, 'silhouette': sil,
        'calinski_harabasz': cal, 'davies_bouldin': dav,
        'n_clusters': len(np.unique(lbl_hac)),
    }
    print(f"    HAC      : Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}")

    # GMM (fallback: full -> diag neu ill-defined)
    try:
        gmm = GaussianMixture(
            n_components=n_clusters, covariance_type='full',
            random_state=config.SEED,
        )
        gmm.fit(Z)
        lbl_gmm = gmm.predict(Z)
    except ValueError:
        print("    GMM full covariance failed, fallback to diag...")
        gmm = GaussianMixture(
            n_components=n_clusters, covariance_type='diag',
            random_state=config.SEED,
        )
        gmm.fit(Z)
        lbl_gmm = gmm.predict(Z)
    sil, cal, dav = _metrics(Z, lbl_gmm)
    clustering_results['GMM'] = {
        'labels': lbl_gmm, 'silhouette': sil,
        'calinski_harabasz': cal, 'davies_bouldin': dav,
        'n_clusters': len(np.unique(lbl_gmm)),
    }
    print(f"    GMM      : Sil={sil:.3f}, Cal={cal:.1f}, Dav={dav:.3f}")

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        metric='euclidean',
    )
    lbl_hdb = clusterer.fit_predict(Z)
    n_cls_hdb = len(np.unique(lbl_hdb[lbl_hdb != -1]))
    sil, cal, dav = _metrics(Z, lbl_hdb)
    clustering_results['HDBSCAN'] = {
        'labels': lbl_hdb, 'silhouette': sil,
        'calinski_harabasz': cal, 'davies_bouldin': dav,
        'n_clusters': n_cls_hdb,
        'n_noise': int((lbl_hdb == -1).sum()),
    }
    print(f"    HDBSCAN  : Sil={sil:.3f}, {n_cls_hdb} cum, "
          f"noise={int((lbl_hdb == -1).sum())}")

    return clustering_results, Z


# ============================================================================
# Visualization
# ============================================================================

def plot_training_loss(train_losses, result_dir=None, save=True):
    """Ve bieu do loss qua cac epoch."""
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Conv1D Autoencoder – Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'M3_01_training_loss.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_reconstruction(model, data, scaler, n_samples=6, device=None,
                        result_dir=None, save=True):
    """Ve so sanh tin hieu goc va tai tao."""
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    data_scaled = scaler.transform(data)

    # Chon ngau nhien n_samples
    rng = np.random.RandomState(config.SEED)
    idxs = rng.choice(len(data), min(n_samples, len(data)), replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for i, idx in enumerate(idxs):
            x = torch.FloatTensor(data_scaled[idx]).unsqueeze(0).unsqueeze(0).to(device)
            x_recon, _ = model(x)
            x_np = data_scaled[idx]
            x_recon_np = x_recon.cpu().numpy().squeeze()

            t = np.arange(len(x_np))
            axes[i].plot(t, x_np, 'b-', alpha=0.7, label='Original', linewidth=1)
            axes[i].plot(t, x_recon_np[:len(t)], 'r--', alpha=0.7,
                        label='Reconstructed', linewidth=1)
            mse = np.mean((x_np - x_recon_np[:len(t)]) ** 2)
            axes[i].set_title(f'Sample {idx} – MSE={mse:.6f}', fontsize=11)
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.3)

    fig.suptitle('Conv1D Autoencoder – Reconstruction Quality',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(result_dir, 'M3_02_reconstruction.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_latent_scatter(Z, labels, method_name, result_dir=None, save=True):
    """Scatter plot latent space (PCA 2D neu latent_dim > 2)."""
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    from sklearn.decomposition import PCA

    if Z.shape[1] > 2:
        pca = PCA(n_components=2, random_state=config.SEED)
        Z_2d = pca.fit_transform(Z)
        explained = pca.explained_variance_ratio_.sum() * 100
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
        subtitle = f'PCA 2D of latent space ({explained:.1f}% variance)'
    else:
        Z_2d = Z
        xlabel, ylabel = 'Latent 1', 'Latent 2'
        subtitle = 'Latent space'

    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    unique_lbls = np.unique(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, lbl in enumerate(unique_lbls):
        mask = labels == lbl
        lbl_txt = 'Noise' if lbl == -1 else f'Cluster {lbl}'
        marker = 'x' if lbl == -1 else 'o'
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                   c=COLORS[i % len(COLORS)], s=80, marker=marker,
                   alpha=0.7, edgecolors='k', linewidths=0.3,
                   label=f'{lbl_txt} (n={mask.sum()})')

    ax.set_title(f'{method_name} – Conv1D Autoencoder Clustering\n({subtitle})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        safe = method_name.replace(' ', '_').lower()
        path = os.path.join(result_dir, f'M3_03_latent_scatter_{safe}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


def plot_cluster_timeseries_deep(hourly_matrix, labels, method_name,
                                  result_dir=None, save=True):
    """Ve chuoi thoi gian trung binh ± std tung cum (Method 3)."""
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    unique_lbls = sorted(set(labels))
    if -1 in unique_lbls:
        unique_lbls.remove(-1)

    if len(unique_lbls) == 0:
        print(f"  [skip] {method_name}: khong co cum nao (tat ca la noise)")
        return

    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    t = np.arange(3600) / 60  # phut

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
                ax.plot(t[vm], s[vm], alpha=0.25, linewidth=0.5,
                       color=COLORS[i % len(COLORS)])

        cdata = hourly_matrix[idxs, :]
        mean = np.nanmean(cdata, axis=0)
        std = np.nanstd(cdata, axis=0)
        vm2 = ~np.isnan(mean)
        if vm2.sum() > 0:
            ax.plot(t[vm2], mean[vm2], color='black', linewidth=2.5,
                    label='Mean', zorder=5)
            ax.fill_between(t[vm2], (mean - std)[vm2], (mean + std)[vm2],
                            alpha=0.2, color='gray', label='±1 std')

        ax.set_title(f'Cluster {lbl}  –  {mask.sum()} gio', fontsize=12, fontweight='bold')
        ax.set_ylabel('h_Coord (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Phut trong gio')
    fig.suptitle(f'{method_name} – Chuoi thoi gian trung binh tung cum',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        safe = method_name.replace(' ', '_').lower()
        path = os.path.join(result_dir, f'M3_05_cluster_ts_{safe}.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.show()


# ============================================================================
# Pipeline Method 3A – Conv1D Autoencoder
# ============================================================================

def run_autoencoder_pipeline(hourly_matrix, hampel_data, valid_hours_info,
                              n_clusters=2, latent_dim=32, epochs=100,
                              result_dir=None):
    """
    Pipeline phan cum Method 3A – Conv1D Autoencoder:
      1. Tien xu ly du lieu (interpolate NaN)
      2. Train autoencoder
      3. Extract latent vectors
      4. Cluster tren latent space
      5. Visualize

    Parameters
    ----------
    hourly_matrix : np.ndarray, shape (n_hours, 3600)
    hampel_data   : np.ndarray, shape (n_hours, 3600)
    valid_hours_info : pd.DataFrame
    n_clusters : int
    latent_dim : int
    epochs : int
    result_dir : str

    Returns
    -------
    results : dict
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("METHOD 3A – CONV1D AUTOENCODER CLUSTERING")
    print("=" * 60)

    # --- Buoc 1: Tien xu ly ---
    print("\n[1] Tien xu ly du lieu...")
    # Dung hampel_data (da loc), interpolate NaN con lai
    data = hampel_data.copy()
    for i in range(len(data)):
        row = data[i]
        nans = np.isnan(row)
        if nans.any():
            if nans.all():
                data[i] = 0.0
            else:
                # Linear interpolation
                valid = ~nans
                xp = np.where(valid)[0]
                fp = row[valid]
                data[i] = np.interp(np.arange(len(row)), xp, fp)

    # Reshape: 3600 → 360 (trung binh cua so 10)
    seq_len = 360
    data_reshaped = data.reshape(data.shape[0], seq_len, -1).mean(axis=2)
    print(f"    Input shape: {data_reshaped.shape} (n_samples x seq_len)")

    # --- Buoc 2: Train autoencoder ---
    print(f"\n[2] Train Conv1D Autoencoder (latent_dim={latent_dim}, epochs={epochs})...")
    model, latent_vectors, train_losses, data_scaler = train_autoencoder(
        data_reshaped, seq_len=seq_len, latent_dim=latent_dim,
        epochs=epochs, batch_size=32, lr=1e-3,
    )

    # Ve loss
    plot_training_loss(train_losses, result_dir=result_dir)

    # Ve reconstruction
    plot_reconstruction(model, data_reshaped, data_scaler, n_samples=6,
                       result_dir=result_dir)

    # --- Buoc 3: Cluster tren latent space ---
    print(f"\n[3] Phan cum tren latent space ({latent_dim}D), k={n_clusters}...")
    clustering_results, Z_scaled = cluster_latent_space(
        latent_vectors, n_clusters=n_clusters,
    )

    # --- Buoc 4: Visualize ---
    print("\n[4] Visualize ket qua...")
    for method_name, res in clustering_results.items():
        plot_latent_scatter(Z_scaled, res['labels'], method_name,
                           result_dir=result_dir)
        plot_cluster_timeseries_deep(hourly_matrix, res['labels'],
                                     f'AE_{method_name}', result_dir=result_dir)

    # --- Bang so sanh ---
    print("\nBANG SO SANH (Method 3A – Conv1D Autoencoder):")
    hdr = f"{'Method':<12} {'k':>4} {'Silhouette':>12} {'Calinski':>10} {'Davies':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m, r in clustering_results.items():
        print(f"{m:<12} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
              f"{r['calinski_harabasz']:>10.2f} {r['davies_bouldin']:>8.4f}")

    return {
        'model': model,
        'latent_vectors': latent_vectors,
        'Z_scaled': Z_scaled,
        'train_losses': train_losses,
        'clustering_results': clustering_results,
    }


# ============================================================================
# Method 3B – Moment Foundation Model
# ============================================================================

def extract_moment_embeddings(data, seq_len=512, device=None):
    """
    Trich xuat embeddings tu Moment foundation model (pre-trained).

    Moment yeu cau input co do dai 512. Neu seq_len != 512, se pad/truncate.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, seq_len_orig)
        Du lieu da tien xu ly (khong NaN).
    seq_len : int
        Do dai input cho Moment (mac dinh 512).
    device : str or None

    Returns
    -------
    embeddings : np.ndarray, shape (n_samples, embed_dim)
    """
    from momentfm import MOMENTPipeline

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"  Loading Moment model (AutonLab/MOMENT-1-large)...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    model = model.to(device)
    model.eval()

    # Chuan hoa du lieu
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Pad/truncate to seq_len (Moment requires 512)
    n_samples, orig_len = data_scaled.shape
    if orig_len < seq_len:
        # Pad with zeros
        padded = np.zeros((n_samples, seq_len))
        padded[:, :orig_len] = data_scaled
        data_input = padded
        input_mask = np.zeros((n_samples, seq_len))
        input_mask[:, :orig_len] = 1.0
    elif orig_len > seq_len:
        # Truncate
        data_input = data_scaled[:, :seq_len]
        input_mask = np.ones((n_samples, seq_len))
    else:
        data_input = data_scaled
        input_mask = np.ones((n_samples, seq_len))

    # Convert to tensor: (n_samples, 1, seq_len)
    X = torch.FloatTensor(data_input).unsqueeze(1).to(device)
    mask = torch.FloatTensor(input_mask).to(device)

    # Extract embeddings in batches
    batch_size = 32
    all_embeddings = []

    print(f"  Extracting embeddings ({n_samples} samples, batch_size={batch_size})...")
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_x = X[i:i+batch_size]
            batch_mask = mask[i:i+batch_size]
            output = model.embed(x_enc=batch_x, input_mask=batch_mask)
            # output.embeddings: (batch, embed_dim)
            emb = output.embeddings.cpu().numpy()
            all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def run_moment_pipeline(hourly_matrix, hampel_data, valid_hours_info,
                         n_clusters=2, result_dir=None):
    """
    Pipeline phan cum Method 3B – Moment Foundation Model:
      1. Tien xu ly du lieu (interpolate NaN)
      2. Extract embeddings tu Moment (zero-shot, khong train)
      3. Cluster tren embedding space
      4. Visualize

    Parameters
    ----------
    hourly_matrix : np.ndarray, shape (n_hours, 3600)
    hampel_data   : np.ndarray, shape (n_hours, 3600)
    valid_hours_info : pd.DataFrame
    n_clusters : int
    result_dir : str

    Returns
    -------
    results : dict
    """
    if result_dir is None:
        result_dir = config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("METHOD 3B – MOMENT FOUNDATION MODEL CLUSTERING")
    print("=" * 60)

    # --- Buoc 1: Tien xu ly ---
    print("\n[1] Tien xu ly du lieu...")
    data = hampel_data.copy()
    for i in range(len(data)):
        row = data[i]
        nans = np.isnan(row)
        if nans.any():
            if nans.all():
                data[i] = 0.0
            else:
                valid = ~nans
                xp = np.where(valid)[0]
                fp = row[valid]
                data[i] = np.interp(np.arange(len(row)), xp, fp)

    # Reshape: 3600 → 360 (trung binh cua so 10)
    seq_len_input = 360
    data_reshaped = data.reshape(data.shape[0], seq_len_input, -1).mean(axis=2)
    print(f"    Input shape: {data_reshaped.shape}")

    # --- Buoc 2: Extract embeddings ---
    print(f"\n[2] Extract Moment embeddings (zero-shot)...")
    embeddings = extract_moment_embeddings(data_reshaped, seq_len=512)

    # --- Buoc 3: Cluster tren embedding space ---
    print(f"\n[3] Phan cum tren embedding space ({embeddings.shape[1]}D), k={n_clusters}...")
    clustering_results, Z_scaled = cluster_latent_space(
        embeddings, n_clusters=n_clusters,
    )

    # --- Buoc 4: Visualize ---
    print("\n[4] Visualize ket qua...")
    for method_name, res in clustering_results.items():
        # Reuse latent scatter with M3B prefix
        plot_latent_scatter(Z_scaled, res['labels'], f'Moment_{method_name}',
                           result_dir=result_dir)
        plot_cluster_timeseries_deep(hourly_matrix, res['labels'],
                                     f'Moment_{method_name}', result_dir=result_dir)

    # --- Bang so sanh ---
    print("\nBANG SO SANH (Method 3B – Moment Foundation Model):")
    hdr = f"{'Method':<12} {'k':>4} {'Silhouette':>12} {'Calinski':>10} {'Davies':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m, r in clustering_results.items():
        print(f"{m:<12} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
              f"{r['calinski_harabasz']:>10.2f} {r['davies_bouldin']:>8.4f}")

    return {
        'embeddings': embeddings,
        'Z_scaled': Z_scaled,
        'clustering_results': clustering_results,
    }
