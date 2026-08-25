#!/usr/bin/env python3
"""
analyze_variable_importance.py - Variable importance analysis for climate variables.

Analyzes which climate variables are most important for:
1. Autoencoder reconstruction (based on variability and data quality)
2. Drought prediction (based on correlation with SPI)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig, get_paths, get_data_path
from data import load_region_timeseries
from data.spi import load_spi_cache, compute_spi
from utils import set_reproducible_seeds
from utils.logger import Logger, Colors


# Region-specific downsampling configurations
REGION_DOWNSAMPLE = {
    "Norte": (6, 6),
    "Nordeste": (5, 5),
    "Centro-Oeste": (5, 5),
    "Sudeste": (4, 4),
    "Sul": (3, 3),
}


# Band labels for display
BAND_LABELS = {
    "pr": "Precipitation",
    "pet": "PET",
    "soil": "Soil Moisture",
    "srad": "Solar Radiation",
    "vap": "Vapor Pressure",
    "vs": "Wind Speed",
    "tavg": "Temperature",
}


class VariableImportanceAnalyzer:
    """
    Analyzes importance of climate variables for autoencoder and predictor.

    For autoencoder: Based on variability (std) and data quality (outlier ratio)
    For predictor: Based on correlation with SPI (Pearson + Spearman)
    """

    def __init__(self, region: str = "Sul", threshold_spi: float = -2.0):
        """
        Initialize the analyzer.

        Args:
            region: Region name (Sul, Norte, etc.)
            threshold_spi: SPI threshold for drought classification
        """
        self.config = ExperimentConfig()
        self.config.region = region
        self.config.spi.threshold = threshold_spi
        self.config.spi.threshold_name = "extreme"

        # Apply region-specific downsampling
        if region in REGION_DOWNSAMPLE:
            self.config.data.downsample_h, self.config.data.downsample_w = REGION_DOWNSAMPLE[region]

        self.logger = Logger()
        set_reproducible_seeds(self.config.random_seed)

        self.paths = get_paths(region, threshold=threshold_spi)
        self.base_data_path = get_data_path()

        self.bands = self.config.data.bands
        self.band_labels = BAND_LABELS

    # =========================================================================
    # MASK VALIDATION
    # =========================================================================

    def _validate_mask(self, valid_mask: np.ndarray, data: np.ndarray) -> None:
        """
        Validate the validity mask.

        Checks if the mask includes non-drought pixels (value 0).

        Args:
            valid_mask: Validity mask (H, W)
            data: Climate data (T, H, W, C)
        """
        if valid_mask is None:
            self.logger.warning("⚠️  No validity mask provided")
            return

        total_pixels = valid_mask.size
        valid_pixels = valid_mask.sum()
        invalid_pixels = total_pixels - valid_pixels

        self.logger.info(f"\n📊 Validity Mask:")
        self.logger.info(f"   Total pixels: {total_pixels:,}")
        self.logger.info(f"   Valid: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
        self.logger.info(f"   Invalid: {invalid_pixels:,} ({invalid_pixels/total_pixels:.1%})")

        # Check for non-drought pixels (value 0) in the mask
        if data is not None and len(data) > 0:
            first_band = data[0, :, :, 0] if data.ndim == 4 else data[0]
            zero_pixels = (first_band == 0)
            zero_valid = zero_pixels & valid_mask

            self.logger.info(f"\n📊 Zero-value (non-drought) pixels:")
            self.logger.info(f"   Total: {zero_pixels.sum():,}")
            self.logger.info(f"   Valid in mask: {zero_valid.sum():,}")

            if zero_valid.sum() == 0 and zero_pixels.sum() > 0:
                self.logger.warning("\n⚠️  WARNING: No non-drought pixels are considered valid!")
                self.logger.warning("   Importance analysis will be based ONLY on drought pixels.")
                self.logger.warning("   This is a SERIOUS issue - check build_valid_mask function!")
            else:
                self.logger.success(f"   ✅ {zero_valid.sum():,} non-drought pixels are valid")

        # Check for NaN/Inf pixels
        if data is not None:
            nan_pixels = np.isnan(data).any(axis=-1).any(axis=0) if data.ndim == 4 else np.isnan(data).any(axis=0)
            nan_invalid = nan_pixels & ~valid_mask

            if nan_invalid.sum() > 0:
                self.logger.info(f"\nℹ️  NaN/Inf pixels marked as invalid: {nan_invalid.sum():,}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_data(self) -> Dict:
        """
        Load climate data and SPI with dimension alignment.

        Returns:
            Dictionary with SPI, variables, bands, metadata, and mask
        """
        self.logger.info(
            f"📂 Loading data with downsampling "
            f"{self.config.data.downsample_h}x{self.config.data.downsample_w}"
        )

        # Load region data
        out = load_region_timeseries(self.base_data_path, self.config)

        data = out["data"]
        valid_mask = out["valid_mask"]

        # Validate mask before using
        self._validate_mask(valid_mask, data)

        # Load SPI from cache
        try:
            spi, _ = load_spi_cache(self.config.spi.scale, self.paths["spi_cache_dir"])
            self.logger.info(f"✅ SPI loaded from cache: {spi.shape}")
        except FileNotFoundError:
            self.logger.warning("⚠️  SPI cache not found. Recalculating...")
            pr_index = self.config.data.bands.index("pr")
            precipitation = out["data"][..., pr_index]
            spi, _ = compute_spi(
                precipitation=precipitation,
                months=out["months"],
                scale=self.config.spi.scale,
                min_samples=self.config.spi.min_samples,
            )
            self.logger.info(f"✅ SPI recomputed: {spi.shape}")
        except Exception as e:
            self.logger.error(f"❌ Error loading SPI: {e}")
            raise

        self.logger.info(f"   Data shape: {data.shape}")
        self.logger.info(f"   Mask shape: {valid_mask.shape if valid_mask is not None else 'None'}")

        # Resize SPI if needed
        if spi.shape[1:] != data.shape[1:3]:
            self.logger.info(f"   Resizing SPI from {spi.shape} to match data {data.shape}")
            from skimage.transform import resize
            spi_resized = np.zeros((spi.shape[0], data.shape[1], data.shape[2]), dtype=spi.dtype)
            for t in range(spi.shape[0]):
                spi_resized[t] = resize(
                    spi[t],
                    (data.shape[1], data.shape[2]),
                    order=0,
                    preserve_range=True
                )
            spi = spi_resized
            self.logger.info(f"   SPI resized: {spi.shape}")

        # Align mask
        if valid_mask is not None and valid_mask.shape != data.shape[1:3]:
            self.logger.info(f"   Resizing valid_mask from {valid_mask.shape} to {data.shape[1:3]}")
            from skimage.transform import resize
            valid_mask = resize(
                valid_mask.astype(np.float32),
                (data.shape[1], data.shape[2]),
                order=0,
                preserve_range=True
            ).astype(bool)
            self.logger.info(f"   Mask resized: {valid_mask.shape}")

        # Flatten data for analysis (mask is used for FILTERING, not modification)
        T, H, W, C = data.shape
        data_flat = data.reshape(T, -1, C)

        if valid_mask is not None:
            mask_flat = valid_mask.flatten()
            data_flat = data_flat[:, mask_flat, :]
            spi_flat = spi.reshape(T, -1)[:, mask_flat]
        else:
            spi_flat = spi.reshape(T, -1)

        # Check if there is enough data after filtering
        if spi_flat.size == 0:
            self.logger.error("❌ No valid pixels after applying mask!")
            self.logger.error("   Check if the validity mask is correct.")
            raise ValueError("No valid pixels available for analysis")

        # Remove NaNs from SPI
        valid_pixels = ~np.isnan(spi_flat)
        spi_clean = spi_flat[valid_pixels]

        # Check distribution
        non_drought_pixels = spi_clean > self.config.spi.threshold
        drought_pixels = spi_clean <= self.config.spi.threshold

        self.logger.info(f"\n📊 SPI Distribution:")
        self.logger.info(f"   Total valid pixels: {len(spi_clean):,}")
        self.logger.info(
            f"   Non-drought (SPI > {self.config.spi.threshold}): "
            f"{non_drought_pixels.sum():,} ({100*non_drought_pixels.sum()/len(spi_clean):.1f}%)"
        )
        self.logger.info(
            f"   Drought (SPI ≤ {self.config.spi.threshold}): "
            f"{drought_pixels.sum():,} ({100*drought_pixels.sum()/len(spi_clean):.1f}%)"
        )

        if non_drought_pixels.sum() == 0:
            self.logger.warning("\n⚠️  WARNING: No non-drought pixels found!")
            self.logger.warning("   Correlation analysis will be based ONLY on drought pixels.")
            self.logger.warning("   This invalidates predictor importance analysis!")

        # Extract variables
        data_clean = {}
        for i, band in enumerate(self.bands):
            band_data = data_flat[:, :, i]
            data_clean[band] = band_data[valid_pixels]

        self.logger.info(f"\n✅ Data loaded: {len(spi_clean):,} valid pixels")
        self.logger.info(f"   Dimensions: T={T}, H={H}, W={W}, C={C}")

        return {
            "spi": spi_clean,
            "variables": data_clean,
            "bands": self.bands,
            "metadata": out["metadata"],
            "valid_mask": valid_mask,
        }

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def analyze_autoencoder_weights(self, data: Dict) -> Tuple[Dict, Dict]:
        """
        Analyze weights for AUTOENCODER based on variability and quality.

        Args:
            data: Data dictionary from load_data()

        Returns:
            Tuple of (weights, results)
        """
        self.logger.section("📊 AUTOENCODER ANALYSIS")

        variables = data["variables"]
        results = {}

        for band in self.bands:
            var_data = variables[band]

            # Calculate variability
            std_dev = np.std(var_data)
            range_val = np.percentile(var_data, 95) - np.percentile(var_data, 5)

            # Calculate quality (outlier ratio)
            q1 = np.percentile(var_data, 25)
            q3 = np.percentile(var_data, 75)
            iqr = q3 - q1
            outlier_ratio = np.sum((var_data < q1 - 1.5*iqr) | (var_data > q3 + 1.5*iqr)) / len(var_data)
            quality_score = 1 - outlier_ratio

            # Combined score (variability + quality)
            std_norm = std_dev / (std_dev + 1e-6)
            score = std_norm * 0.6 + quality_score * 0.4

            results[band] = {
                "std_dev": std_dev,
                "range": range_val,
                "quality": quality_score,
                "score": score
            }

            self.logger.info(
                f"  {self.band_labels.get(band, band):<15} "
                f"Score: {score:.3f} | Std: {std_dev:.3f} | Quality: {quality_score:.2%}"
            )

        # Normalize weights to values between 0.5 and 3.0
        scores = np.array([r["score"] for r in results.values()])
        max_score = scores.max()

        if max_score > 0:
            normalized = scores / max_score
            weights = {
                band: round(max(0.5, min(3.0, norm * 3.0)), 2)
                for band, norm in zip(self.bands, normalized)
            }
        else:
            weights = {band: 1.0 for band in self.bands}

        return weights, results

    def analyze_predictor_weights(self, data: Dict) -> Tuple[Dict, Dict]:
        """
        Analyze weights for PREDICTOR based on correlation with SPI.

        Args:
            data: Data dictionary from load_data()

        Returns:
            Tuple of (weights, results)
        """
        self.logger.section("📊 PREDICTOR ANALYSIS")

        spi = data["spi"]
        variables = data["variables"]

        # Check if there is enough data
        if len(spi) < 100:
            self.logger.warning("⚠️  Insufficient data for correlation analysis")
            return {band: 1.0 for band in self.bands}, {}

        results = {}

        # Sample to avoid memory issues
        n_samples = min(50000, len(spi))
        indices = np.random.choice(len(spi), n_samples, replace=False)

        for band in self.bands:
            var_data = variables[band][indices]
            spi_sample = spi[indices]

            # Calculate correlations
            pearson_r, p_pearson = pearsonr(var_data, spi_sample)
            spearman_r, p_spearman = spearmanr(var_data, spi_sample)

            # Score based on absolute correlation
            score = (abs(pearson_r) + abs(spearman_r)) / 2

            results[band] = {
                "pearson_r": pearson_r,
                "pearson_p": p_pearson,
                "spearman_r": spearman_r,
                "spearman_p": p_spearman,
                "score": score
            }

            # Color based on correlation strength
            color = Colors.GREEN if score > 0.3 else Colors.YELLOW if score > 0.15 else Colors.RESET
            self.logger.info(
                f"  {color}{self.band_labels.get(band, band):<15}{Colors.RESET} "
                f"Score: {score:.3f} | Pearson: {pearson_r:>6.3f} | Spearman: {spearman_r:>6.3f}"
            )

        # Normalize weights to values between 0.5 and 3.0
        scores = np.array([r["score"] for r in results.values()])
        max_score = scores.max()

        if max_score > 0:
            normalized = scores / max_score
            weights = {
                band: round(max(0.5, min(3.0, norm * 3.0)), 2)
                for band, norm in zip(self.bands, normalized)
            }
        else:
            weights = {band: 1.0 for band in self.bands}

        return weights, results

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def plot_analysis(
        self,
        autoencoder_weights: Dict,
        predictor_weights: Dict,
        out_dir: Path
    ) -> None:
        """
        Generate comparison plots.

        Args:
            autoencoder_weights: Autoencoder weights
            predictor_weights: Predictor weights
            out_dir: Output directory
        """
        bands = list(autoencoder_weights.keys())
        labels = [self.band_labels.get(b, b) for b in bands]

        # Create figure with both bar chart and table
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        x = np.arange(len(bands))
        width = 0.35

        ae_weights = [autoencoder_weights[b] for b in bands]
        pred_weights = [predictor_weights[b] for b in bands]

        bars1 = ax1.bar(x - width/2, ae_weights, width, label='Autoencoder',
                       color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, pred_weights, width, label='Predictor',
                       color='#e74c3c', alpha=0.8)

        # Add value labels
        for bar, val in zip(bars1, ae_weights):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        for bar, val in zip(bars2, pred_weights):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        ax1.set_xlabel('Variable', fontsize=11)
        ax1.set_ylabel('Weight', fontsize=11)
        ax1.set_title(f'Recommended Weights - {self.config.region} (SPI={self.config.spi.threshold})',
                     fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3, axis='y')
        ax1.set_ylim(0, 3.5)

        # Table
        table_data = []
        for band in bands:
            table_data.append([
                self.band_labels.get(band, band),
                f"{autoencoder_weights.get(band, 1.0):.2f}",
                f"{predictor_weights.get(band, 1.0):.2f}"
            ])

        table = ax2.table(
            cellText=table_data,
            colLabels=['Variable', 'Autoencoder', 'Predictor'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.3, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Color table cells based on values
        for i, row in enumerate(table_data):
            ae_val = float(row[1])
            pred_val = float(row[2])

            # Color for autoencoder
            if ae_val >= 2.0:
                color = '#d4edda'  # Green
            elif ae_val >= 1.5:
                color = '#fff3cd'  # Yellow
            else:
                color = '#f8d7da'  # Red

            table[(i+1, 1)].set_facecolor(color)

            # Color for predictor
            if pred_val >= 2.0:
                color = '#d4edda'
            elif pred_val >= 1.5:
                color = '#fff3cd'
            else:
                color = '#f8d7da'

            table[(i+1, 2)].set_facecolor(color)

        # Header color
        for j, col in enumerate(['Variable', 'Autoencoder', 'Predictor']):
            table[(0, j)].set_facecolor('#343a40')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax2.axis('off')
        ax2.set_title('Weight Summary', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_dir / "weights_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.success("✅ Comparison plot saved")

    # =========================================================================
    # RESULTS SAVING
    # =========================================================================

    def save_results(
        self,
        autoencoder_weights: Dict,
        predictor_weights: Dict,
        autoencoder_results: Dict,
        predictor_results: Dict,
        out_dir: Path
    ) -> None:
        """
        Save results to CSV.

        Args:
            autoencoder_weights: Autoencoder weights
            predictor_weights: Predictor weights
            autoencoder_results: Autoencoder analysis results
            predictor_results: Predictor analysis results
            out_dir: Output directory
        """
        df_data = []
        for band in self.bands:
            ae = autoencoder_results.get(band, {})
            pred = predictor_results.get(band, {})
            df_data.append({
                "variable": band,
                "label": self.band_labels.get(band, band),
                "ae_weight": autoencoder_weights.get(band, 1.0),
                "pred_weight": predictor_weights.get(band, 1.0),
                "ae_score": ae.get("score", 0.0),
                "ae_std": ae.get("std_dev", 0.0),
                "ae_quality": ae.get("quality", 0.0),
                "pred_corr": pred.get("score", 0.0),
                "pred_pearson": pred.get("pearson_r", 0.0),
                "pred_spearman": pred.get("spearman_r", 0.0),
                "pred_pearson_p": pred.get("pearson_p", 1.0),
                "pred_spearman_p": pred.get("spearman_p", 1.0),
            })

        df = pd.DataFrame(df_data)
        df.to_csv(out_dir / "variable_analysis.csv", index=False)
        self.logger.success(f"✅ Results saved to {out_dir / 'variable_analysis.csv'}")

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def run(self) -> None:
        """Execute complete analysis."""
        self.logger.header("🔍 VARIABLE IMPORTANCE ANALYSIS")

        self.logger.info(f"   Region: {self.config.region}")
        self.logger.info(f"   SPI Threshold: {self.config.spi.threshold}")
        self.logger.info(f"   Downsample: {self.config.data.downsample_h}x{self.config.data.downsample_w}")

        out_dir = self.paths["analysis_dir"] / "variable_importance"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            data = self.load_data()
        except ValueError as e:
            self.logger.error(f"❌ Error loading data: {e}")
            return

        # Run analyses
        ae_weights, ae_results = self.analyze_autoencoder_weights(data)
        pred_weights, pred_results = self.analyze_predictor_weights(data)

        # Generate outputs
        self.plot_analysis(ae_weights, pred_weights, out_dir)
        self.save_results(ae_weights, pred_weights, ae_results, pred_results, out_dir)

        # Final summary
        self.logger.header("📊 FINAL SUMMARY")

        print(f"\n📍 {self.config.region} (SPI={self.config.spi.threshold})")
        print(f"   Downsample: {self.config.data.downsample_h}x{self.config.data.downsample_w}")
        print(f"{'─' * 55}")

        print("\n🏷️  AUTOENCODER WEIGHTS (data variability + quality):")
        for band in self.bands:
            label = self.band_labels.get(band, band)
            weight = ae_weights.get(band, 1.0)
            color = Colors.GREEN if weight >= 2.0 else Colors.YELLOW if weight >= 1.5 else Colors.RESET
            print(f"  {color}{label:<15}: {weight:.2f}{Colors.RESET}")

        print("\n🎯 PREDICTOR WEIGHTS (correlation with SPI):")
        for band in self.bands:
            label = self.band_labels.get(band, band)
            weight = pred_weights.get(band, 1.0)
            color = Colors.GREEN if weight >= 2.0 else Colors.YELLOW if weight >= 1.5 else Colors.RESET
            print(f"  {color}{label:<15}: {weight:.2f}{Colors.RESET}")

        print("\n💡 RECOMMENDATION:")
        print(f"  Use AUTOENCODER weights in experiment_config.py")

        print("\n📝 To update autoencoder variable_weights:")
        print("variable_weights: List[float] = field(default_factory=lambda: [")
        for band in self.bands:
            weight = ae_weights.get(band, 1.0)
            label = self.band_labels.get(band, band)
            print(f"    {weight:.2f},   # {label}")
        print("])")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze variable importance for drought forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Sul region with SPI threshold -2.0
  python analyze_variable_importance.py --region Sul --threshold -2.0

  # Analyze Nordeste region
  python analyze_variable_importance.py --region Nordeste --threshold -2.0
        """
    )
    parser.add_argument("--region", type=str, default="Sul",
                       help="Region to analyze")
    parser.add_argument("--threshold", type=float, default=-2.0,
                       help="SPI threshold")

    args = parser.parse_args()

    analyzer = VariableImportanceAnalyzer(region=args.region, threshold_spi=args.threshold)
    analyzer.run()


if __name__ == "__main__":
    main()