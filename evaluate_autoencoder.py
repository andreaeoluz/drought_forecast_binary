#!/usr/bin/env python3
"""evaluate_autoencoder.py - Detailed evaluation of a trained autoencoder.

Computes reconstruction metrics (MAE, RMSE, SSIM) over valid pixels only,
extracts and analyzes the latent representation (rank, effective rank,
explained variance), and generates diagnostic plots.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig, get_paths, get_data_path
from data import load_region_timeseries, ClimateNormalizer
from models import ConvLSTMAutoencoder
from utils import set_reproducible_seeds
from utils.logger import Logger, Colors


# Same rcParams used across analyze_spi_distribution_shift.py,
# generate_false_alarm_maps.py, and generate_obs_pred_panel.py, so every
# figure in the paper -- including these diagnostic plots -- shares one
# visual identity instead of the seaborn dark-grid/husl look used before.
def apply_publication_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Colorblind-safe palette (Okabe-Ito), reused instead of seaborn's "husl" /
# ad hoc named colors (lightgreen, lightcoral, ...) so these diagnostic
# plots read consistently with the other paper figures.
PALETTE = {
    "primary": "#0072B2",    # original / reference series
    "secondary": "#D55E00",  # reconstruction / error / "attention wanted"
    "tertiary": "#009E73",   # secondary reference threshold
    "neutral": "#555555",
}


class AutoencoderEvaluator:
    """
    Full evaluator for a trained autoencoder.

    Provides:
    - Reconstruction metrics (MAE, RMSE, SSIM), computed over valid pixels only.
    - Latent representation extraction.
    - Visualizations (comparison, errors, time series).
    - Latent space analysis (with centering).
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = Logger()
        self.device = config.device
        self.paths = get_paths(config.region)

        set_reproducible_seeds(config.random_seed)

        self.ae_path = self.paths["autoencoder_dir"] / "model.pth"
        self.normalizer_path = self.paths["autoencoder_dir"] / "normalizer.json"

        self.results = {}
        self.model = None
        self.normalizer = None
        self.data = None
        self.valid_mask = None
        self.months = None
        self.years = None

    # ========================================================================
    # LOADING METHODS
    # ========================================================================

    def load_model(self) -> bool:
        """Load the trained model and its normalizer."""
        if not self.ae_path.exists():
            self.logger.error(f"❌ Autoencoder not found: {self.ae_path}")
            return False

        if not self.normalizer_path.exists():
            self.logger.error(f"❌ Normalizer not found: {self.normalizer_path}")
            return False

        self.normalizer = ClimateNormalizer.load(self.normalizer_path)
        self.logger.success(f"✅ Normalizer loaded: {self.normalizer_path}")

        checkpoint = torch.load(self.ae_path, map_location=self.device, weights_only=False)

        model_config = {
            "input_dim": self.config.num_bands,
            "hidden_dims": self.config.model.hidden_dims,
            "kernel_size": self.config.model.kernel_size,
            "use_attention": self.config.model.use_attention,
            "output_dim": self.config.num_bands,
            "attention_dropout": getattr(self.config.model, 'attention_dropout', 0.3),
            "reconstruct_full_sequence": getattr(
                self.config.autoencoder, 'reconstruct_full_sequence', True
            ),
            "sequence_length": getattr(
                self.config.autoencoder, 'sequence_length', 12
            ),
            "decay_rate": getattr(
                self.config.autoencoder, 'decay_rate', 0.3
            ),
            "skip_weight": getattr(
                self.config.autoencoder, 'skip_weight', 0.1
            ),
            "attention_weight": getattr(
                self.config.autoencoder, 'attention_weight', 0.3
            ),
            "diversity_weight": getattr(
                self.config.autoencoder, 'diversity_weight', 0.01
            ),
        }

        self.model = ConvLSTMAutoencoder(model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        reconstruct_full = getattr(self.model, 'reconstruct_full_sequence', False)
        skip_weight = getattr(self.model, 'skip_weight', 0.1)
        decay_rate = getattr(self.model, 'decay_rate', 0.3)
        attention_weight = getattr(self.model, 'attention_weight', 0.3)

        self.logger.success(f"✅ Autoencoder loaded: {self.ae_path}")
        self.logger.info(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"   Best loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        self.logger.info(f"   Reconstruct full sequence: {reconstruct_full}")
        self.logger.info(f"   Skip weight: {skip_weight}")
        self.logger.info(f"   Decay rate: {decay_rate}")
        self.logger.info(f"   Attention weight: {attention_weight}")

        return True

    def load_data(self) -> bool:
        """Load the evaluation data (validation period)."""
        base_data_path = get_data_path()
        out = load_region_timeseries(base_data_path, self.config)

        self.valid_mask = out["valid_mask"]
        data = out["data"]
        self.months = out["months"]
        self.years = out["years"]

        split_info_gs = self.config.split.get_end_indices_gs()
        time_idx = np.array([y * 12 + (m - 1) for y, m in zip(self.years, self.months)])

        val_mask = (time_idx > split_info_gs["train_end"]) & (time_idx <= split_info_gs["val_end"])
        data_val = data[val_mask]
        months_val = self.months[val_mask]

        data_val_norm = self.normalizer.transform(data_val, months_val, self.valid_mask)
        self.data = np.nan_to_num(data_val_norm, nan=0.0)

        self.logger.success(f"✅ Data loaded: {len(self.data)} months")
        self.logger.info(f"   Shape: {self.data.shape}")
        self.logger.info(f"   Valid pixels: {self.valid_mask.sum()} / {self.valid_mask.size}")
        self.logger.info(f"   Valid ratio: {self.valid_mask.sum() / self.valid_mask.size:.2%}")

        return True

    # ========================================================================
    # RECONSTRUCTION METHODS
    # ========================================================================

    def get_reconstructions(self, p: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reconstructions from the autoencoder over the evaluation data.

        Returns:
            originals: (N, H, W, C) - original data.
            recons: (N, H, W, C) - reconstructions.
            masks: (N, H, W) - validity masks.
        """
        self.model.eval()

        all_originals = []
        all_recons = []
        all_masks = []

        reconstruct_full = getattr(self.model, 'reconstruct_full_sequence', False)

        mask_tensor = None
        if self.valid_mask is not None:
            mask_tensor = torch.from_numpy(self.valid_mask.astype(np.float32)).to(self.device)

        with torch.no_grad():
            for t in range(p, len(self.data)):
                x = self.data[t-p:t]  # (T, H, W, C)
                x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
                x_tensor = x_tensor.permute(0, 1, 4, 2, 3)  # (1, T, C, H, W)

                recon, info = self.model(x_tensor, mask=mask_tensor, return_info=True)

                if reconstruct_full and "recon_seq" in info:
                    recon_seq = info["recon_seq"]
                    recon = recon_seq[0, -1]  # Last timestep
                elif isinstance(recon, tuple):
                    recon = recon[0]
                else:
                    recon = recon[0]

                original = x[-1]  # (H, W, C)
                recon_np = recon.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)

                all_originals.append(original)
                all_recons.append(recon_np)
                all_masks.append(self.valid_mask)

        return np.array(all_originals), np.array(all_recons), np.array(all_masks)

    # ========================================================================
    # LATENT EXTRACTION METHODS
    # ========================================================================

    def get_latent_representations(
        self,
        p: int = 12,
        use_attention: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract latent representations from the autoencoder.

        Args:
            p: Historical sequence length.
            use_attention: If True, use encode_with_attention (with attention).

        Returns:
            Dict with:
                - latents: (N, C_latent, H, W)
                - latents_flat: (N, C_latent * H * W)
                - indices: temporal indices
                - months: corresponding months
                - shape: dict with dimensions
        """
        self.model.eval()

        all_latents = []
        all_indices = []
        all_months = []

        with torch.no_grad():
            for t in range(p, len(self.data)):
                x = self.data[t-p:t]  # (T, H, W, C)
                x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
                x_tensor = x_tensor.permute(0, 1, 4, 2, 3)  # (1, T, C, H, W)

                if use_attention and hasattr(self.model, 'encode_with_attention'):
                    latent = self.model.encode_with_attention(x_tensor)
                else:
                    latent = self.model.encode(x_tensor)  # (1, C_latent, H, W)

                all_latents.append(latent.cpu().numpy())
                all_indices.append(t)
                all_months.append(self.months[t] if t < len(self.months) else None)

        latents = np.array(all_latents).squeeze(1)  # (N, C_latent, H, W)

        return {
            'latents': latents,
            'latents_flat': latents.reshape(latents.shape[0], -1),
            'indices': np.array(all_indices),
            'months': np.array(all_months) if all_months[0] is not None else None,
            'shape': {
                'n_samples': latents.shape[0],
                'latent_dim': latents.shape[1],
                'height': latents.shape[2],
                'width': latents.shape[3],
            }
        }

    # ========================================================================
    # METRICS COMPUTATION
    # ========================================================================

    def compute_metrics(
        self,
        originals: np.ndarray,
        recons: np.ndarray,
        masks: np.ndarray
    ) -> Dict:
        """
        Compute reconstruction metrics over valid pixels only.

        Metrics:
            - MAE: Mean absolute error over valid pixels.
            - RMSE: Root mean squared error over valid pixels.
            - MSE: Mean squared error over valid pixels.
            - SSIM: Structural similarity over the valid region.
            - Max Error: Maximum error over valid pixels.

        Returns:
            Dict of computed metrics.
        """
        metrics = {}

        # Assumes all samples share the same validity mask.
        valid_mask = masks[0] if len(masks) > 0 else None

        if valid_mask is None:
            self.logger.warning("⚠️  No valid mask provided - using all pixels")
            valid_mask = np.ones(originals.shape[1:3], dtype=bool)

        valid = valid_mask.astype(bool)  # (H, W)

        N, H, W, C = originals.shape

        errors = originals - recons  # (N, H, W, C)
        valid_errors = errors[:, valid, :]  # (N, Nv, C)

        mae = np.mean(np.abs(valid_errors))
        mse = np.mean(valid_errors ** 2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(valid_errors))

        metrics['mae'] = float(mae)
        metrics['mse'] = float(mse)
        metrics['rmse'] = float(rmse)
        metrics['max_error'] = float(max_error)

        # ================================================================
        # SKILL SCORE (Nash-Sutcliffe-style, relative to a naive baseline)
        # ================================================================
        # Absolute MAE/RMSE are only meaningful relative to the scale of
        # what is being reconstructed. Anomaly-mode targets are much
        # smaller-magnitude (and inherently noisier/harder) than raw
        # normalized values, so a fixed absolute threshold silently
        # changes meaning depending on use_anomalies. The skill score
        # instead compares the model's MSE to the MSE of the trivial
        # "predict zero" baseline (the naive predictor for an already
        # zero-centered target, whether that's a z-scored raw value or a
        # true monthly anomaly): 1.0 = perfect, 0.0 = no better than the
        # naive baseline, <0 = worse than it. This stays comparable across
        # anomaly vs. raw-value reconstruction and across regions.
        valid_originals = originals[:, valid, :]
        baseline_mse = np.mean(valid_originals ** 2)
        baseline_mae = np.mean(np.abs(valid_originals))

        metrics['baseline_mse'] = float(baseline_mse)
        metrics['skill_score'] = (
            float(1.0 - mse / baseline_mse) if baseline_mse > 0 else float('nan')
        )
        metrics['nmae'] = (
            float(mae / baseline_mae) if baseline_mae > 0 else float('nan')
        )

        ssim_scores = self._compute_ssim_valid_region(originals, recons, valid_mask)
        metrics['ssim_mean'] = float(np.nanmean(ssim_scores)) if ssim_scores else np.nan
        metrics['ssim_per_band'] = [float(s) if not np.isnan(s) else np.nan for s in ssim_scores]

        band_names = self.config.data.bands if hasattr(self.config.data, 'bands') else [f'band_{i}' for i in range(C)]
        metrics['per_band'] = {}

        for b, band_name in enumerate(band_names):
            valid_error_band = valid_errors[:, :, b]  # (N, Nv)
            valid_original_band = valid_originals[:, :, b]

            band_mae = float(np.mean(np.abs(valid_error_band)))
            band_mse = float(np.mean(valid_error_band ** 2))
            band_rmse = float(np.sqrt(band_mse))
            band_baseline_mse = float(np.mean(valid_original_band ** 2))
            band_skill = (
                1.0 - band_mse / band_baseline_mse if band_baseline_mse > 0 else float('nan')
            )

            metrics['per_band'][band_name] = {
                'mae': band_mae,
                'rmse': band_rmse,
                'mse': band_mse,
                'skill_score': band_skill,
            }

        metrics['n_valid_pixels'] = int(valid.sum())
        metrics['n_total_pixels'] = int(valid.size)
        metrics['valid_ratio'] = float(valid.sum() / valid.size)

        return metrics

    def _compute_ssim_valid_region(
        self,
        originals: np.ndarray,
        recons: np.ndarray,
        valid_mask: np.ndarray
    ) -> List[float]:
        """
        Compute SSIM restricted to spatial windows that are entirely valid.

        Args:
            originals: Original data (N, H, W, C).
            recons: Reconstructed data (N, H, W, C).
            valid_mask: Spatial validity mask (H, W).

        Returns:
            List with the mean SSIM per band.
        """
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            self.logger.warning("SSIM not available (install scikit-image)")
            return [np.nan] * originals.shape[-1]

        N, H, W, C = originals.shape
        valid = valid_mask.astype(bool)

        win_size = 7
        half = win_size // 2

        if H < win_size or W < win_size:
            self.logger.warning(f"Grid too small for SSIM with win_size={win_size}")
            return [np.nan] * C

        scores_by_band = [[] for _ in range(C)]

        for n in range(N):
            # Only fully-valid windows are considered.
            for r in range(half, H - half):
                for c in range(half, W - half):

                    window_mask = valid[
                        r - half:r + half + 1,
                        c - half:c + half + 1
                    ]

                    if not window_mask.all():
                        continue

                    for b in range(C):

                        orig_window = originals[
                            n,
                            r - half:r + half + 1,
                            c - half:c + half + 1,
                            b
                        ]

                        recon_window = recons[
                            n,
                            r - half:r + half + 1,
                            c - half:c + half + 1,
                            b
                        ]

                        if (
                            not np.isfinite(orig_window).all()
                            or not np.isfinite(recon_window).all()
                        ):
                            continue

                        # Data range derived from the original window itself.
                        data_range = (
                            orig_window.max() -
                            orig_window.min()
                        )

                        if data_range <= 1e-8:
                            continue

                        try:
                            score = ssim(
                                orig_window,
                                recon_window,
                                data_range=data_range,
                                win_size=win_size
                            )

                            if np.isfinite(score):
                                scores_by_band[b].append(score)

                        except Exception:
                            continue

        ssim_scores = []

        for b in range(C):
            if scores_by_band[b]:
                ssim_scores.append(
                    float(np.mean(scores_by_band[b]))
                )
            else:
                ssim_scores.append(np.nan)

        return ssim_scores

    # ========================================================================
    # LATENT SPACE ANALYSIS
    # ========================================================================

    def get_latent_stats(self, latents: Dict[str, np.ndarray]) -> Dict:
        """
        Compute latent representation statistics.

        Includes:
            - Basic statistics (mean, std, min, max, etc.)
            - True matrix rank
            - Effective rank via a centered SVD (not simply rank / min dim)
            - Number of components needed for 90% and 95% of the variance
        """
        latent_flat = latents['latents_flat']  # (N, D)
        N, D = latent_flat.shape

        stats = {
            'n_samples': N,
            'latent_dim': D,
            'mean': float(latent_flat.mean()),
            'std': float(latent_flat.std()),
            'min': float(latent_flat.min()),
            'max': float(latent_flat.max()),
            'median': float(np.median(latent_flat)),
            'q25': float(np.percentile(latent_flat, 25)),
            'q75': float(np.percentile(latent_flat, 75)),
            'skewness': float(pd.Series(latent_flat.flatten()).skew()),
            'kurtosis': float(pd.Series(latent_flat.flatten()).kurtosis()),
        }

        stats['rank'] = int(np.linalg.matrix_rank(latent_flat))

        if N > 1 and D > 1:
            latent_centered = latent_flat - latent_flat.mean(axis=0, keepdims=True)

            try:
                u, s, vh = np.linalg.svd(latent_centered, full_matrices=False)

                variance = s ** 2
                explained_variance = variance / variance.sum()

                # Effective rank via the participation ratio (normalized
                # Shannon entropy of the explained-variance spectrum).
                stats['effective_rank'] = float(1.0 / np.sum((explained_variance) ** 2))

                cumsum = np.cumsum(explained_variance)
                stats['rank_90_percent'] = int(np.searchsorted(cumsum, 0.90) + 1)
                stats['rank_95_percent'] = int(np.searchsorted(cumsum, 0.95) + 1)

                stats['top_components'] = explained_variance[:min(5, len(explained_variance))].tolist()
                stats['top_components_cumsum'] = cumsum[:min(5, len(cumsum))].tolist()

                stats['svd_values'] = s[:min(10, len(s))].tolist()

            except np.linalg.LinAlgError as e:
                self.logger.warning(f"⚠️  SVD failed: {e}")
                stats['effective_rank'] = np.nan
                stats['rank_90_percent'] = D
                stats['rank_95_percent'] = D
        else:
            stats['effective_rank'] = 1.0 if D > 0 else 0.0
            stats['rank_90_percent'] = 1 if D > 0 else 0
            stats['rank_95_percent'] = 1 if D > 0 else 0

        return stats

    # ========================================================================
    # SAVE METHODS
    # ========================================================================

    def save_results(
        self,
        originals: np.ndarray,
        recons: np.ndarray,
        masks: np.ndarray,
        metrics: Dict,
        latents: Optional[Dict] = None
    ):
        """Save evaluation results to disk."""
        output_dir = self.paths["autoencoder_dir"] / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = pd.DataFrame({
            'Metric': [
                'MAE', 'RMSE', 'MSE', 'SSIM', 'Max Error',
                'Skill Score', 'Normalized MAE', 'Baseline MSE',
                'Valid Pixels', 'Total Pixels', 'Valid Ratio'
            ],
            'Value': [
                metrics['mae'],
                metrics['rmse'],
                metrics['mse'],
                metrics.get('ssim_mean', np.nan),
                metrics['max_error'],
                metrics.get('skill_score', np.nan),
                metrics.get('nmae', np.nan),
                metrics.get('baseline_mse', np.nan),
                metrics.get('n_valid_pixels', 0),
                metrics.get('n_total_pixels', 0),
                metrics.get('valid_ratio', 0.0),
            ]
        })
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        band_metrics = []
        for band_name, band_vals in metrics['per_band'].items():
            band_metrics.append({
                'Band': band_name,
                'MAE': band_vals['mae'],
                'RMSE': band_vals['rmse'],
                'MSE': band_vals['mse'],
                'Skill Score': band_vals.get('skill_score', np.nan),
            })
        band_df = pd.DataFrame(band_metrics)
        band_df.to_csv(output_dir / "metrics_per_band.csv", index=False)

        n_samples = min(10, len(originals))
        sample_indices = np.random.choice(len(originals), n_samples, replace=False)

        np.savez(
            output_dir / "reconstructions.npz",
            originals=originals[sample_indices],
            recons=recons[sample_indices],
            masks=masks[sample_indices] if len(masks) > 0 else None,
            indices=sample_indices
        )

        if latents is not None:
            np.save(output_dir / "latent_representations.npy", latents['latents'])
            np.save(output_dir / "latent_representations_flat.npy", latents['latents_flat'])

            latent_stats = self.get_latent_stats(latents)

            stats_df = pd.DataFrame([latent_stats])
            stats_df.to_csv(output_dir / "latent_stats.csv", index=False)

            self.logger.info(f"   Latent shape: {latents['shape']}")
            self.logger.info(f"   Latent stats saved to: {output_dir / 'latent_stats.csv'}")

        self.logger.success(f"Results saved to: {output_dir}")

    # ========================================================================
    # PLOT METHODS
    # ========================================================================

    def plot_results(
        self,
        originals: np.ndarray,
        recons: np.ndarray,
        masks: np.ndarray,
        metrics: Dict,
        latents: Optional[Dict] = None
    ):
        """Generate evaluation plots."""
        output_dir = self.paths["autoencoder_dir"] / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use('default')
        apply_publication_style()

        self._plot_reconstruction_comparison(originals, recons, masks, output_dir)
        self._plot_error_analysis(originals, recons, masks, output_dir)
        self._plot_band_metrics(metrics, output_dir)
        self._plot_spatial_error(originals, recons, masks, output_dir)
        self._plot_temporal_series(originals, recons, masks, output_dir)

        if latents is not None:
            self._plot_latent_space(latents, output_dir)

        self.logger.success(f"Plots saved to: {output_dir}")

    def _plot_reconstruction_comparison(
        self,
        originals: np.ndarray,
        recons: np.ndarray,
        masks: np.ndarray,
        output_dir: Path
    ):
        """Plot original vs. reconstruction comparisons."""
        n_samples = min(4, len(originals))
        band_names = self.config.data.bands if hasattr(self.config.data, 'bands') else [f'band_{i}' for i in range(originals.shape[-1])]
        n_bands = len(band_names)

        fig = plt.figure(figsize=(16, 4 * n_samples))
        gs = GridSpec(n_samples, n_bands * 2 + 1, figure=fig)

        valid_mask = masks[0] if len(masks) > 0 else np.ones(originals.shape[1:3], dtype=bool)
        mask_3d = valid_mask[:, :, np.newaxis]

        for i in range(n_samples):
            idx = np.random.randint(len(originals))

            for b, band_name in enumerate(band_names):
                ax_orig = fig.add_subplot(gs[i, b * 2])
                orig_band = originals[idx, :, :, b] * mask_3d[:, :, 0]
                orig_band = np.ma.masked_where(~valid_mask, orig_band)
                im = ax_orig.imshow(orig_band, cmap='RdBu_r', aspect='auto')
                ax_orig.set_title(f'Original - {band_name}', fontsize=10)
                ax_orig.axis('off')

                ax_recon = fig.add_subplot(gs[i, b * 2 + 1])
                recon_band = recons[idx, :, :, b] * mask_3d[:, :, 0]
                recon_band = np.ma.masked_where(~valid_mask, recon_band)
                im_recon = ax_recon.imshow(recon_band, cmap='RdBu_r', aspect='auto')
                ax_recon.set_title(f'Reconstruction - {band_name}', fontsize=10)
                ax_recon.axis('off')

            ax_error = fig.add_subplot(gs[i, -1])
            error = (originals[idx] - recons[idx]) ** 2
            error = error * mask_3d
            error_mean = error.mean(axis=-1)
            error_mean = np.ma.masked_where(~valid_mask, error_mean)
            im_error = ax_error.imshow(error_mean, cmap='hot', aspect='auto')
            ax_error.set_title('MSE Error', fontsize=10)
            ax_error.axis('off')
            cb = plt.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8)

        fig.suptitle('Original vs Reconstruction Comparison', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = output_dir / 'reconstruction_comparison'
        plt.savefig(f'{out}.png')
        plt.savefig(f'{out}.pdf')
        plt.close()

    def _plot_error_analysis(
        self,
        originals: np.ndarray,
        recons: np.ndarray,
        masks: np.ndarray,
        output_dir: Path
    ):
        """Plot error distribution and correlation diagnostics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        valid_mask = masks[0] if len(masks) > 0 else np.ones(originals.shape[1:3], dtype=bool)
        valid = valid_mask.astype(bool)
        N, H, W, C = originals.shape

        errors = originals - recons  # (N, H, W, C)
        valid_errors = errors[:, valid, :]  # (N, Nv, C)
        errors_flat = valid_errors.flatten()
        errors_flat = errors_flat[~np.isnan(errors_flat)]

        ax = axes[0, 0]
        ax.hist(errors_flat, bins=50, alpha=0.85, density=True, color=PALETTE["primary"], edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=PALETTE["secondary"], linestyle='--', linewidth=1.8, label='Zero Error')
        ax.set_title('Error Distribution')
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend()

        ax = axes[0, 1]
        values = originals[:, valid, :].flatten()
        errors_abs = np.abs(valid_errors).flatten()
        valid_vals = ~(np.isnan(values) | np.isnan(errors_abs))
        ax.scatter(values[valid_vals], errors_abs[valid_vals], alpha=0.12, s=1, color=PALETTE["primary"])
        ax.set_title('Absolute Error vs Original Value (Valid Pixels)')
        ax.set_xlabel('Original Value')
        ax.set_ylabel('Absolute Error')
        ax.set_yscale('log')

        ax = axes[0, 2]
        band_errors = []
        band_names = self.config.data.bands if hasattr(self.config.data, 'bands') else [f'band_{i}' for i in range(C)]
        for b in range(C):
            band_err = valid_errors[:, :, b].flatten()
            band_err = band_err[~np.isnan(band_err)]
            band_errors.append(band_err)
        bp = ax.boxplot(band_errors, labels=band_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(PALETTE["primary"])
            patch.set_alpha(0.5)
        ax.set_title('Errors by Band')
        ax.set_ylabel('Error')
        ax.axhline(0, color=PALETTE["secondary"], linestyle='--', linewidth=1.2)

        ax = axes[1, 0]
        from scipy import stats
        stats.probplot(errors_flat, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor(PALETTE["primary"])
        ax.get_lines()[0].set_markeredgecolor(PALETTE["primary"])
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[1].set_color(PALETTE["secondary"])
        ax.set_title('Q-Q Plot')

        ax = axes[1, 1]
        rmse_per_sample = np.sqrt(np.mean(valid_errors ** 2, axis=(1, 2)))
        ax.plot(rmse_per_sample, 'o-', alpha=0.8, markersize=3, color=PALETTE["primary"])
        ax.axhline(np.mean(rmse_per_sample), color=PALETTE["secondary"], linestyle='--',
                   label=f'Mean: {np.mean(rmse_per_sample):.4f}')
        ax.set_title('RMSE per Sample')
        ax.set_xlabel('Sample')
        ax.set_ylabel('RMSE')
        ax.legend()

        ax = axes[1, 2]
        orig_flat = originals[:, valid, :].flatten()
        recon_flat = recons[:, valid, :].flatten()
        valid_pair = ~(np.isnan(orig_flat) | np.isnan(recon_flat))
        ax.scatter(orig_flat[valid_pair], recon_flat[valid_pair], alpha=0.12, s=1, color=PALETTE["primary"])
        ax.plot([-3, 3], [-3, 3], color=PALETTE["secondary"], linestyle='--', label='Perfect')
        ax.set_title('Original vs Reconstruction')
        ax.set_xlabel('Original')
        ax.set_ylabel('Reconstruction')
        ax.legend()

        fig.tight_layout()
        out = output_dir / 'error_analysis'
        plt.savefig(f'{out}.png')
        plt.savefig(f'{out}.pdf')
        plt.close()

    def _plot_band_metrics(self, metrics: Dict, output_dir: Path):
        """Plot per-band MAE and RMSE."""
        band_names = list(metrics['per_band'].keys())
        mae_vals = [metrics['per_band'][b]['mae'] for b in band_names]
        rmse_vals = [metrics['per_band'][b]['rmse'] for b in band_names]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        bars = ax.bar(band_names, mae_vals, color=PALETTE["primary"], edgecolor="white", alpha=0.9)
        ax.set_title('MAE by Band')
        ax.set_ylabel('MAE')
        ax.set_xlabel('Band')
        ax.set_xticklabels(band_names, rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.6f}', ha='center', va='bottom', fontsize=8)

        ax = axes[1]
        bars = ax.bar(band_names, rmse_vals, color=PALETTE["secondary"], edgecolor="white", alpha=0.9)
        ax.set_title('RMSE by Band')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Band')
        ax.set_xticklabels(band_names, rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.6f}', ha='center', va='bottom', fontsize=8)

        fig.suptitle('Reconstruction Metrics by Band', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = output_dir / 'band_metrics'
        plt.savefig(f'{out}.png')
        plt.savefig(f'{out}.pdf')
        plt.close()

    def _plot_spatial_error(
        self,
        originals: np.ndarray,
        recons: np.ndarray,
        masks: np.ndarray,
        output_dir: Path
    ):
        """Plot the spatial distribution of the reconstruction error."""
        valid_mask = masks[0] if len(masks) > 0 else np.ones(originals.shape[1:3], dtype=bool)
        mask_3d = valid_mask[:, :, np.newaxis]

        spatial_rmse = np.sqrt(np.mean((originals - recons) ** 2, axis=(0, 3)))
        spatial_rmse = spatial_rmse * valid_mask
        spatial_rmse = np.ma.masked_where(~valid_mask, spatial_rmse)

        spatial_mae = np.mean(np.abs(originals - recons), axis=(0, 3))
        spatial_mae = spatial_mae * valid_mask
        spatial_mae = np.ma.masked_where(~valid_mask, spatial_mae)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        im = ax.imshow(spatial_rmse, cmap='hot', aspect='auto')
        ax.set_title('Spatial RMSE')
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9)

        ax = axes[1]
        im = ax.imshow(spatial_mae, cmap='hot', aspect='auto')
        ax.set_title('Spatial MAE')
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9)

        fig.suptitle('Spatial Error Distribution', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = output_dir / 'spatial_error'
        plt.savefig(f'{out}.png')
        plt.savefig(f'{out}.pdf')
        plt.close()

    def _plot_temporal_series(
        self,
        originals: np.ndarray,
        recons: np.ndarray,
        masks: np.ndarray,
        output_dir: Path
    ):
        """Plot original vs. reconstructed time series for a random valid pixel."""
        valid_mask = masks[0] if len(masks) > 0 else np.ones(originals.shape[1:3], dtype=bool)
        valid_pixels = np.where(valid_mask)

        if len(valid_pixels[0]) == 0:
            self.logger.warning("⚠️  No valid pixels for temporal series")
            return

        idx = np.random.randint(len(valid_pixels[0]))
        pixel_h, pixel_w = valid_pixels[0][idx], valid_pixels[1][idx]

        band_names = self.config.data.bands if hasattr(self.config.data, 'bands') else [f'band_{i}' for i in range(originals.shape[-1])]
        n_bands = len(band_names)

        fig, axes = plt.subplots(n_bands, 1, figsize=(12, 3 * n_bands))
        if n_bands == 1:
            axes = [axes]

        for b, band_name in enumerate(band_names):
            ax = axes[b]

            orig_series = originals[:, pixel_h, pixel_w, b]
            recon_series = recons[:, pixel_h, pixel_w, b]

            ax.plot(orig_series, color=PALETTE["primary"], linewidth=1.8, label='Original', alpha=0.9)
            ax.plot(recon_series, color=PALETTE["secondary"], linestyle='--', linewidth=1.8,
                    label='Reconstruction', alpha=0.9)

            error = orig_series - recon_series
            ax.fill_between(range(len(orig_series)), orig_series - error, orig_series,
                          alpha=0.25, color=PALETTE["secondary"], label='Error')

            ax.set_title(f'{band_name} \u2014 Pixel ({pixel_h}, {pixel_w})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()

        fig.suptitle('Temporal Series: Original vs Reconstruction', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = output_dir / 'temporal_series'
        plt.savefig(f'{out}.png')
        plt.savefig(f'{out}.pdf')
        plt.close()

    def _plot_latent_space(self, latents: Dict[str, np.ndarray], output_dir: Path):
        """Plot the latent space with PCA/t-SNE, with centering applied for PCA."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
        except ImportError:
            self.logger.warning("⚠️  sklearn not available for latent space visualization")
            return

        latent_flat = latents['latents_flat']
        N, D = latent_flat.shape

        latent_centered = latent_flat - latent_flat.mean(axis=0, keepdims=True)

        n_components = min(2, D)

        pca = PCA(n_components=n_components)
        latent_pca = pca.fit_transform(latent_centered)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.65, s=12, color=PALETTE["primary"])
        if n_components == 2:
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('Latent Space \u2014 PCA (Centered)')

        if N > 20:
            try:
                perplexity = min(30, N // 2)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                latent_tsne = tsne.fit_transform(latent_flat)

                ax = axes[1]
                ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.65, s=12, color=PALETTE["primary"])
                ax.set_title('Latent Space \u2014 t-SNE')
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
            except Exception as e:
                self.logger.warning(f"⚠️  t-SNE failed: {e}")
                axes[1].text(0.5, 0.5, 't-SNE failed', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Latent Space - t-SNE (failed)')

        fig.suptitle('Latent Space Visualization', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = output_dir / 'latent_space'
        plt.savefig(f'{out}.png')
        plt.savefig(f'{out}.pdf')
        plt.close()

        fig, ax = plt.subplots(figsize=(9, 5.5))
        pca_full = PCA()
        pca_full.fit(latent_centered)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        ax.plot(cum_var, color=PALETTE["primary"], linewidth=2)
        ax.axhline(0.9, color=PALETTE["secondary"], linestyle='--', linewidth=1.2, label='90% variance')
        ax.axhline(0.95, color=PALETTE["tertiary"], linestyle='--', linewidth=1.2, label='95% variance')
        ax.set_title('Explained Variance by PCA Components (Centered)')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.legend()

        n_90 = np.argmax(cum_var >= 0.9) + 1
        n_95 = np.argmax(cum_var >= 0.95) + 1
        ax.axvline(n_90, color=PALETTE["secondary"], linestyle=':', alpha=0.7)
        ax.axvline(n_95, color=PALETTE["tertiary"], linestyle=':', alpha=0.7)
        ax.text(n_90, 0.1, f'n={n_90}', color=PALETTE["secondary"], fontsize=9)
        ax.text(n_95, 0.05, f'n={n_95}', color=PALETTE["tertiary"], fontsize=9)

        fig.tight_layout()
        out = output_dir / 'latent_explained_variance'
        plt.savefig(f'{out}.png')
        plt.savefig(f'{out}.pdf')
        plt.close()

    # ========================================================================
    # SUMMARY METHODS
    # ========================================================================

    def print_summary(self, metrics: Dict, latent_stats: Optional[Dict] = None):
        """Print an evaluation summary."""
        self.logger.header("EVALUATION SUMMARY")

        print(f"\n{Colors.BOLD}Valid Region Info:{Colors.RESET}")
        print(f"  {'Valid Pixels':<20} {metrics.get('n_valid_pixels', 'N/A'):,}")
        print(f"  {'Total Pixels':<20} {metrics.get('n_total_pixels', 'N/A'):,}")
        print(f"  {'Valid Ratio':<20} {metrics.get('valid_ratio', 0.0):.2%}")

        print(f"\n{Colors.BOLD}Main Metrics:{Colors.RESET}")
        print(f"  {'MAE':<20} {metrics['mae']:.6f}")
        print(f"  {'RMSE':<20} {metrics['rmse']:.6f}")
        print(f"  {'MSE':<20} {metrics['mse']:.6f}")
        print(f"  {'Max Error':<20} {metrics['max_error']:.6f}")

        skill = metrics.get('skill_score', np.nan)
        if not np.isnan(skill):
            print(f"  {'Skill Score':<20} {skill:.4f}  "
                  f"(vs. naive zero-baseline MSE={metrics.get('baseline_mse', float('nan')):.6f})")

        ssim_val = metrics.get('ssim_mean', np.nan)
        if not np.isnan(ssim_val):
            print(f"  {'SSIM':<20} {ssim_val:.4f}")

        print(f"\n{Colors.BOLD}Metrics by Band:{Colors.RESET}")
        print(f"  {'Band':<12} {'MAE':<12} {'RMSE':<12} {'Skill':<12}")
        print(f"  {'─' * 50}")
        for band_name, band_vals in metrics['per_band'].items():
            band_skill = band_vals.get('skill_score', np.nan)
            skill_str = f"{band_skill:.4f}" if not np.isnan(band_skill) else "N/A"
            print(f"  {band_name:<12} {band_vals['mae']:<12.6f} {band_vals['rmse']:<12.6f} {skill_str:<12}")

        if latent_stats is not None:
            print(f"\n{Colors.BOLD}Latent Statistics:{Colors.RESET}")
            print(f"  {'N Samples':<20} {latent_stats['n_samples']}")
            print(f"  {'Latent Dim':<20} {latent_stats['latent_dim']}")
            print(f"  {'Mean':<20} {latent_stats['mean']:.6f}")
            print(f"  {'Std':<20} {latent_stats['std']:.6f}")
            print(f"  {'Min':<20} {latent_stats['min']:.6f}")
            print(f"  {'Max':<20} {latent_stats['max']:.6f}")
            print(f"  {'Rank':<20} {latent_stats['rank']}")

            if 'effective_rank' in latent_stats and not np.isnan(latent_stats['effective_rank']):
                print(f"  {'Effective Rank':<20} {latent_stats['effective_rank']:.4f}")

            if 'rank_90_percent' in latent_stats:
                print(f"  {'Rank 90% Variance':<20} {latent_stats['rank_90_percent']}")
            if 'rank_95_percent' in latent_stats:
                print(f"  {'Rank 95% Variance':<20} {latent_stats['rank_95_percent']}")

        print(f"\n{Colors.BOLD}Qualitative Assessment:{Colors.RESET}")

        # Relative to a naive zero-baseline (see compute_metrics), not an
        # absolute MAE cutoff: absolute error only means something once you
        # know the scale of what's being reconstructed, and that scale
        # changes a lot between anomaly-mode (small, noisy deviations) and
        # raw-value mode (larger, seasonally-structured values). Bands follow
        # the same convention as Nash-Sutcliffe model-skill guidelines
        # (Moriasi et al., 2007): >0.75 very good, >0.5 good, >0.2 acceptable.
        skill = metrics.get('skill_score', float('nan'))
        if np.isnan(skill):
            print("  ⚠️  Skill score unavailable (baseline MSE was zero)")
        elif skill > 0.75:
            print(f"  ✅ Excellent: Reconstruction explains {skill:.1%} of the "
                  "target's variance vs. a naive zero-baseline")
        elif skill > 0.5:
            print(f"  ✅ Good: Reconstruction explains {skill:.1%} of the "
                  "target's variance vs. a naive zero-baseline")
        elif skill > 0.2:
            print(f"  ⚠️  Moderate: Reconstruction explains {skill:.1%} of the "
                  "target's variance vs. a naive zero-baseline")
        elif skill > 0:
            print(f"  ❌ Poor: Reconstruction only explains {skill:.1%} of the "
                  "target's variance vs. a naive zero-baseline")
        else:
            print(f"  ❌ Poor: Reconstruction is no better than a naive "
                  f"zero-baseline (skill score = {skill:.4f})")

        if 'ssim_mean' in metrics and not np.isnan(metrics['ssim_mean']):
            ssim_val = metrics['ssim_mean']
            if ssim_val > 0.9:
                print(f"  ✅ Excellent: High structural similarity (SSIM = {ssim_val:.4f})")
            elif ssim_val > 0.8:
                print(f"  ✅ Good: Good structural similarity (SSIM = {ssim_val:.4f})")
            else:
                print(f"  ⚠️  Moderate: Moderate structural similarity (SSIM = {ssim_val:.4f})")

    # ========================================================================
    # MAIN EVALUATION
    # ========================================================================

    def evaluate(self, extract_latents: bool = True):
        """
        Run the complete evaluation.

        Args:
            extract_latents: If True, extract and analyze latent representations.
        """
        self.logger.header("AUTOENCODER EVALUATION")

        if not self.load_model():
            return False
        if not self.load_data():
            return False

        self.logger.info("Computing reconstructions...")
        p = self.config.autoencoder.p if hasattr(self.config.autoencoder, 'p') else 12
        originals, recons, masks = self.get_reconstructions(p)

        self.logger.info(f"   Reconstructed {len(originals)} samples")
        self.logger.info(f"   Original shape: {originals.shape}")
        self.logger.info(f"   Reconstruction shape: {recons.shape}")

        self.logger.info("Computing metrics...")
        metrics = self.compute_metrics(originals, recons, masks)
        self.results['metrics'] = metrics

        latents = None
        latent_stats = None
        if extract_latents:
            self.logger.info("Extracting latent representations...")

            latents_no_attn = self.get_latent_representations(p, use_attention=False)
            self.logger.info(f"   Latent (no attention): {latents_no_attn['shape']}")

            if hasattr(self.model, 'encode_with_attention'):
                latents_attn = self.get_latent_representations(p, use_attention=True)
                self.logger.info(f"   Latent (with attention): {latents_attn['shape']}")
                latents = latents_attn
            else:
                latents = latents_no_attn

            latent_stats = self.get_latent_stats(latents)
            self.results['latent_stats'] = latent_stats

        self.save_results(originals, recons, masks, metrics, latents)
        self.plot_results(originals, recons, masks, metrics, latents)
        self.print_summary(metrics, latent_stats)

        return True


# ========================================================================
# MAIN
# ========================================================================

def main():
    """Main entry point."""
    config = ExperimentConfig()
    logger = Logger()

    logger.header("AUTOENCODER EVALUATOR")
    print(f"  Region: {config.region}")
    print(f"  SPI scale: {config.spi.scale}")
    print(f"  Using anomalies: {config.autoencoder.use_anomalies}")
    print(f"  Reconstruct full sequence: {config.autoencoder.reconstruct_full_sequence}")
    print(f"  Decay rate: {config.autoencoder.decay_rate}")
    print(f"  Skip weight: {config.autoencoder.skip_weight}")
    print()

    evaluator = AutoencoderEvaluator(config)
    success = evaluator.evaluate(extract_latents=True)

    if success:
        logger.success("Evaluation completed successfully!")
    else:
        logger.error("Evaluation failed!")


if __name__ == "__main__":
    main()