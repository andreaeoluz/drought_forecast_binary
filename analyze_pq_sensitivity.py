#!/usr/bin/env python3
"""
analyze_pq_sensitivity.py - Analysis of p and q parameter sensitivity.

This script analyzes grid search results to evaluate model sensitivity
to p (history length) and q (forecast horizon) parameters,
comparing scenarios with and without transfer learning.

Generates:
1. MCC/CSI heatmaps for each region
2. p/q sensitivity plots
3. Statistical analysis by region
4. TL vs Scratch comparison
5. LaTeX tables for the paper
"""

import sys
from pathlib import Path

# Must happen before the local `utils` import below can resolve.
sys.path.insert(0, str(Path(__file__).parent))

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

from utils.logger import Colors, Logger

warnings.filterwarnings('ignore')


class PQSensitivityAnalyzer:
    """
    Analyzes p and q parameter sensitivity from grid search results.
    """

    # Regions and thresholds - English display names
    REGIONS = ["South", "Southeast", "Northeast", "Center-West", "North"]

    # Mapping back to the original Portuguese names (used to load files)
    REGION_MAPPING = {
        "South": "Sul",
        "Southeast": "Sudeste",
        "Northeast": "Nordeste",
        "Center-West": "Centro-Oeste",
        "North": "Norte"
    }

    THRESHOLD = -2.0
    THRESHOLD_NAME = "extreme"

    def __init__(self, base_dir: Path):
        """
        Initialize analyzer.

        Args:
            base_dir: Base directory with grid search results
        """
        self.base_dir = base_dir
        self.logger = Logger()
        self.results = {}
        self.metrics = ["mcc", "csi", "f1", "precision", "recall"]

    # ========================================================================
    # LOADING METHODS
    # ========================================================================

    def load_all_results(self) -> None:
        """Loads all grid search results for all regions."""
        self.logger.header("📊 LOADING GRID SEARCH RESULTS")

        for region_en in self.REGIONS:
            # Original Portuguese name, used to load files
            region_pt = self.REGION_MAPPING[region_en]
            self.logger.info(f"\n  Loading {region_en} ({region_pt})...")
            region_loaded = False

            for tl in [False, True]:
                suffix = f"thr_{abs(self.THRESHOLD):.1f}_tl_{tl}"
                results_dir = self.base_dir / region_pt / "grid_search" / "results" / suffix

                excel_file = results_dir / f"grid_search_results_{suffix}.xlsx"

                if excel_file.exists():
                    df = pd.read_excel(excel_file, sheet_name="All Results")
                    df['region'] = region_en  # store the English region name
                    df['region_pt'] = region_pt  # keep the original for reference
                    df['transfer_learning'] = tl
                    df['threshold_name'] = self.THRESHOLD_NAME

                    self.results[(region_en, tl)] = df
                    self.logger.success(f"    ✅ TL={tl}: {len(df)} combinations")
                    region_loaded = True
                else:
                    # Try alternative pattern (without TL in name)
                    alt_dir = self.base_dir / region_pt / "grid_search" / "results" / f"thr_{abs(self.THRESHOLD):.1f}"
                    alt_file = alt_dir / f"grid_search_results_thr_{abs(self.THRESHOLD):.1f}.xlsx"

                    if alt_file.exists():
                        df = pd.read_excel(alt_file, sheet_name="All Results")
                        df['region'] = region_en
                        df['region_pt'] = region_pt
                        df['transfer_learning'] = tl
                        df['threshold_name'] = self.THRESHOLD_NAME

                        self.results[(region_en, tl)] = df
                        self.logger.success(f"    ✅ TL={tl} (legacy): {len(df)} combinations")
                        region_loaded = True
                    else:
                        self.logger.warning(f"    ⚠️  TL={tl}: file not found")
                        self.results[(region_en, tl)] = None

            if not region_loaded:
                self.logger.warning(f"  ⚠️  No files found for {region_en}")

        self.logger.success("\n✅ All results loaded")

    # ========================================================================
    # AGGREGATION METHODS
    # ========================================================================

    def get_aggregated_results(self, metric: str = "mcc") -> pd.DataFrame:
        """
        Aggregates results by region, p, q and TL.

        Args:
            metric: Metric to aggregate ('mcc', 'csi', etc.)

        Returns:
            DataFrame with aggregated results
        """
        rows = []

        for (region_en, tl), df in self.results.items():
            if df is None:
                continue

            col = f"best_{metric}"
            if col not in df.columns:
                # Try without "best_"
                if metric in df.columns:
                    col = metric
                else:
                    continue

            for _, row in df.iterrows():
                rows.append({
                    "region": row.get("region", region_en),
                    "region_pt": row.get("region_pt", self.REGION_MAPPING.get(region_en, region_en)),
                    "p": row["p"],
                    "q": row["q"],
                    "transfer_learning": row.get("transfer_learning", tl),
                    metric: row[col],
                    "csi": row.get("best_csi", row.get("csi", np.nan)),
                    "mcc": row.get("best_mcc", row.get("mcc", np.nan)),
                    "threshold": row.get("threshold", row.get("best_threshold", np.nan)),
                    "precision": row.get("precision", np.nan),
                    "recall": row.get("recall", np.nan),
                    "f1": row.get("f1", np.nan),
                })

        return pd.DataFrame(rows)

    def get_best_parameters(self, metric: str = "mcc") -> pd.DataFrame:
        """
        Finds best parameters by region and TL.

        Args:
            metric: Metric to optimize

        Returns:
            DataFrame with best parameters
        """
        df = self.get_aggregated_results(metric)
        if df.empty:
            return df
        best = df.loc[df.groupby(["region", "transfer_learning"])[metric].idxmax()]
        return best.reset_index(drop=True)

    # ========================================================================
    # LATEX TABLE GENERATION
    # ========================================================================

    def generate_latex_tables(self, metric: str = "mcc", output_dir: Path = None) -> None:
        """
        Generates LaTeX tables with results.

        Args:
            metric: Metric to use
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity" / "latex"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)
        best = self.get_best_parameters(metric)

        if df.empty:
            self.logger.warning("⚠️  No data for LaTeX tables")
            return

        # Table 1: Best parameters by region
        # When metric == "csi" the generic second "CSI" column would just
        # repeat the first one (same value, same header) -- only add it
        # when it carries distinct information (i.e. metric != "csi").
        show_csi_col = metric != "csi"
        n_cols = 7 if show_csi_col else 6

        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Best p and q parameters by region}")
        latex.append("\\label{tab:best_params}")
        latex.append("\\small")
        latex.append("\\begin{tabular}{l" + "c" * (n_cols - 1) + "}")
        latex.append("\\toprule")
        header = ["\\textbf{Region}", "\\textbf{TL}", "\\textbf{Best p}", "\\textbf{Best q}",
                   f"\\textbf{{{metric.upper()}}}"]
        if show_csi_col:
            header.append("\\textbf{CSI}")
        header.append("\\textbf{Threshold}")
        latex.append(" & ".join(header) + " \\\\")
        latex.append("\\midrule")

        for _, row in best.iterrows():
            region = row["region"]
            tl = "Yes" if row["transfer_learning"] else "No"
            p = int(row["p"])
            q = int(row["q"])
            mcc = row[metric]
            csi = row.get("csi", np.nan)
            threshold = row.get("threshold", np.nan)

            fields = [region, tl, str(p), str(q), f"{mcc:.4f}"]
            if show_csi_col:
                fields.append(f"{csi:.4f}")
            fields.append(f"{threshold:.3f}")
            latex.append(" & ".join(fields) + " \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        with open(output_dir / f"best_params_{metric}.tex", "w") as f:
            f.write("\n".join(latex))

        # Table 2: Sensitivity matrix by region
        for region_en in self.REGIONS:
            subset = df[(df["region"] == region_en) & (df["transfer_learning"] == True)]
            if subset.empty:
                continue

            pivot = subset.pivot_table(
                values=metric,
                index="p",
                columns="q",
                aggfunc="mean"
            ).round(4)

            if pivot.empty:
                continue

            latex = []
            latex.append("\\begin{table}[htbp]")
            latex.append("\\centering")
            latex.append(f"\\caption{{p/q sensitivity - {region_en} (with TL)}}")
            latex.append(f"\\label{{tab:sensitivity_{region_en.lower()}}}")
            latex.append("\\small")
            latex.append("\\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
            latex.append("\\toprule")

            # Header
            header = ["p $\\backslash$ q"] + [str(q) for q in pivot.columns]
            latex.append(" & ".join(header) + " \\\\")
            latex.append("\\midrule")

            # Rows
            for p in pivot.index:
                row = [str(p)] + [f"{pivot.loc[p, q]:.4f}" for q in pivot.columns]
                # Highlight best value
                best_val = pivot.loc[p].max()
                for i, q in enumerate(pivot.columns):
                    if pivot.loc[p, q] == best_val:
                        row[i+1] = f"\\textbf{{{row[i+1]}}}"
                latex.append(" & ".join(row) + " \\\\")

            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
            latex.append("\\end{table}")

            with open(output_dir / f"sensitivity_{region_en.lower()}_{metric}.tex", "w") as f:
                f.write("\n".join(latex))

        # Table 3: TL vs Scratch comparison
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Transfer Learning vs Scratch comparison}")
        latex.append("\\label{tab:tl_comparison}")
        latex.append("\\small")
        latex.append("\\begin{tabular}{lcccccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Region} & \\textbf{Best (Scratch)} & \\textbf{Best (TL)} & "
                      "\\textbf{Gain} & \\textbf{Improvements (\\%)} & \\textbf{Best p/q (TL)} \\\\")
        latex.append("\\midrule")

        for region_en in self.REGIONS:
            subset = df[df["region"] == region_en]
            if subset.empty:
                continue

            scratch = subset[subset["transfer_learning"] == False]
            tl = subset[subset["transfer_learning"] == True]

            if scratch.empty or tl.empty:
                continue

            best_scratch = scratch.loc[scratch[metric].idxmax()]
            best_tl = tl.loc[tl[metric].idxmax()]

            gain = best_tl[metric] - best_scratch[metric]

            # Merge scratch and tl by (p, q) to compare only common combinations
            merged = pd.merge(
                scratch[["p", "q", metric]],
                tl[["p", "q", metric]],
                on=["p", "q"],
                suffixes=("_scratch", "_tl")
            )

            if not merged.empty:
                improvements = (merged[f"{metric}_tl"] > merged[f"{metric}_scratch"]).mean() * 100
            else:
                improvements = 0.0

            latex.append(
                f"{region_en} & {best_scratch[metric]:.4f} & "
                f"{best_tl[metric]:.4f} & {gain:+.4f} & "
                f"{improvements:.1f}\\% & "
                f"{int(best_tl['p'])}/{int(best_tl['q'])} \\\\"
            )

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        with open(output_dir / f"tl_comparison_{metric}.tex", "w") as f:
            f.write("\n".join(latex))

        self.logger.success(f"✅ LaTeX tables saved to: {output_dir}")

    # ========================================================================
    # SUMMARY METHODS
    # ========================================================================

    def print_summary(self, metric: str = "mcc") -> None:
        """
        Prints analysis summary.

        Args:
            metric: Metric to analyze
        """
        self.logger.header(f"📊 SENSITIVITY ANALYSIS SUMMARY - {metric.upper()}")

        df = self.get_aggregated_results(metric)
        best = self.get_best_parameters(metric)

        if df.empty:
            self.logger.warning("⚠️  No data available")
            return

        print(f"\n{Colors.BOLD}Best Parameters by Region:{Colors.RESET}")
        print(f"  {'Region':<15} {'TL':<6} {'p':<4} {'q':<4} {metric.upper():<8} {'CSI':<8} {'Threshold':<10}")
        print(f"  {'─' * 60}")

        for _, row in best.iterrows():
            tl = "Yes" if row["transfer_learning"] else "No"
            print(
                f"  {row['region']:<15} {tl:<6} {int(row['p']):<4} {int(row['q']):<4} "
                f"{row[metric]:<8.4f} {row.get('csi', 0):<8.4f} "
                f"{row.get('threshold', 0):<10.3f}"
            )

        print(f"\n{Colors.BOLD}Transfer Learning Impact:{Colors.RESET}")

        for region_en in self.REGIONS:
            subset = df[df["region"] == region_en]
            if subset.empty:
                continue

            scratch = subset[subset["transfer_learning"] == False]
            tl = subset[subset["transfer_learning"] == True]

            if scratch.empty or tl.empty:
                print(f"  {region_en:<15} Incomplete data")
                continue

            merged = pd.merge(
                scratch[["p", "q", metric]],
                tl[["p", "q", metric]],
                on=["p", "q"],
                suffixes=("_scratch", "_tl")
            )

            if merged.empty:
                print(f"  {region_en:<15} No common combinations")
                continue

            mean_gain = (merged[f"{metric}_tl"] - merged[f"{metric}_scratch"]).mean()
            median_gain = (merged[f"{metric}_tl"] - merged[f"{metric}_scratch"]).median()
            better_count = (merged[f"{metric}_tl"] > merged[f"{metric}_scratch"]).sum()
            total_count = len(merged)

            color = Colors.GREEN if mean_gain > 0 else Colors.RED
            print(
                f"  {region_en:<15} {color}Δ={mean_gain:+.4f}{Colors.RESET} "
                f"(median: {median_gain:+.4f}) | "
                f"Better: {better_count}/{total_count} ({100*better_count/total_count:.1f}%)"
            )

        print(f"\n{Colors.BOLD}Best Overall Configuration:{Colors.RESET}")
        if not best.empty:
            best_overall = best.loc[best[metric].idxmax()]
            print(f"  Region: {best_overall['region']}")
            print(f"  TL: {'Yes' if best_overall['transfer_learning'] else 'No'}")
            print(f"  p: {int(best_overall['p'])}")
            print(f"  q: {int(best_overall['q'])}")
            print(f"  {metric.upper()}: {best_overall[metric]:.4f}")
            print(f"  CSI: {best_overall.get('csi', 0):.4f}")
            print(f"  Threshold: {best_overall.get('threshold', 0):.3f}")

    # ========================================================================
    # PLOTTING METHODS - PUBLICATION READY
    # ========================================================================

    # Publication style configuration -- same family/sizing as the other
    # paper figures (analyze_spi_distribution_shift.py, generate_false_alarm_maps.py,
    # generate_obs_pred_panel.py, evaluate_autoencoder.py), plus pdf.fonttype
    # so PDF exports keep editable/searchable text.
    PUBLICATION_STYLE = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 1.0,
        'grid.alpha': 0.25,
        'grid.linestyle': '-',
        'grid.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }

    # Colorblind-safe (Okabe-Ito) region colors, matching the palette used
    # throughout the rest of the paper's figures instead of the previous
    # ad hoc hex set (which put a green and an orange-red close enough in
    # hue to be hard to tell apart under deuteranopia/protanopia).
    PUBLICATION_COLORS = {
        "South": "#0072B2",       # blue
        "Southeast": "#CC79A7",   # reddish purple
        "Northeast": "#E69F00",   # orange
        "Center-West": "#009E73",  # bluish green
        "North": "#D55E00",       # vermillion
    }

    # Sequential, colorblind-safe colormap for the p/q heatmaps. RdYlGn
    # (the previous choice) is a red-green diverging map -- exactly the
    # pair that's hardest to distinguish under the most common forms of
    # color vision deficiency, and it also loses all contrast when the
    # figure is printed or photocopied in grayscale.
    HEATMAP_CMAP = "cividis"

    def apply_publication_style(self):
        """Applies publication-ready style."""
        plt.rcParams.update(self.PUBLICATION_STYLE)
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.0)

    def plot_heatmaps(self, metric: str = "mcc", output_dir: Path = None,
                       fmt: str = "png", filename: str = None) -> None:
        """
        Generate p/q sensitivity heatmaps.

        Args:
            metric: Metric to visualize
            output_dir: Output directory
            fmt: Output file format (e.g. 'png', 'pdf'), used only when
                `filename` is not given
            filename: Explicit output filename (overrides the default
                `heatmaps_{metric}.{fmt}` pattern)
        """
        self.apply_publication_style()

        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)

        if df.empty:
            self.logger.warning(f"⚠️  No data for {metric} heatmaps")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(5, 2, figsize=(14.8, 18))

        # Color scale shared between Scratch and TL *within the same
        # region* (so that comparison is honest), but NOT across regions:
        # a single figure-wide vmax was tried first and it backfired --
        # the p=*,q=1 configurations are far ahead of every other (p,q) in
        # every region, so a global scale crushed all the q>1 cells into
        # near-identical dark colors and lost exactly the contrast that
        # matters for reading the sensitivity pattern. Per-row scaling
        # keeps the Scratch vs. TL comparison valid (the main thing this
        # figure is for) while restoring contrast within each region.
        # On top of that, a plain linear scale still buries every q>1 cell
        # near the dark end of the colormap, because q=1 configurations
        # consistently score far above every other (p,q) combination in
        # every region -- so the linear scale spends most of its dynamic
        # range on distinguishing values that are all "close to the q=1
        # value" and almost none on distinguishing the q>1 cells from each
        # other, which is exactly the pattern this figure needs to show.
        # A gamma<1 power-law norm (kept on a colorblind-safe cmap, unlike
        # switching to a diverging red-green map) spreads the low end back
        # out without touching color-safety.
        for idx, region_en in enumerate(self.REGIONS):
            row_values = df[df["region"] == region_en][metric]
            row_vmax = max(row_values.max() * 0.95, 0.05) if len(row_values) else 0.1

            for tl_idx, tl in enumerate([False, True]):
                ax = axes[idx, tl_idx]

                subset = df[(df["region"] == region_en) & (df["transfer_learning"] == tl)]
                if subset.empty:
                    ax.text(0.5, 0.5, 'No data',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, style='italic')
                    continue

                # Create pivot table
                pivot = subset.pivot_table(
                    values=metric,
                    index="p",
                    columns="q",
                    aggfunc="mean"
                )
                pivot = pivot.fillna(0)

                im = ax.imshow(pivot.values, cmap=self.HEATMAP_CMAP, aspect='auto',
                              norm=PowerNorm(gamma=0.5, vmin=0, vmax=row_vmax),
                              interpolation='nearest')

                # Configure axes
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)

                ax.set_xlabel('q (forecast horizon)')
                ax.set_ylabel('p (history length)')

                # Add region label in the corner (English name)
                tl_label = 'TL' if tl else 'Scratch'
                ax.text(0.02, 0.98, f'{region_en} - {tl_label}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # One colorbar per row (valid now that Scratch/TL share
                # vmin/vmax within that row).
                if tl_idx == 1:
                    cbar = plt.colorbar(im, ax=list(axes[idx, :]), fraction=0.023, pad=0.02)
                    cbar.set_label(metric.upper(), fontsize=9)
                    cbar.ax.tick_params(labelsize=8)

        # Remove extra spaces
        plt.subplots_adjust(wspace=0.35, hspace=0.35)

        # Save (PNG + PDF, matching the rest of the paper's figures, unless
        # an explicit filename was given -- then save exactly that file).
        if filename:
            out = output_dir / filename
            plt.savefig(out, pad_inches=0.05)
            plt.close()
            self.logger.success(f"✅ Heatmaps saved to: {out}")
        else:
            stem = f'heatmaps_{metric}'
            for ext in ("png", "pdf"):
                plt.savefig(output_dir / f"{stem}.{ext}", pad_inches=0.05)
            plt.close()
            self.logger.success(f"✅ Heatmaps saved to: {output_dir / stem}.[png/pdf]")

    def plot_sensitivity_analysis(self, metric: str = "mcc", output_dir: Path = None) -> None:
        """
        Generate sensitivity analysis plots.

        Args:
            metric: Metric to visualize
            output_dir: Output directory
        """
        self.apply_publication_style()

        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)

        if df.empty:
            self.logger.warning(f"⚠️  No data for {metric} sensitivity analysis")
            return

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Sensitivity to p
        ax = axes[0, 0]
        for region_en in self.REGIONS:
            for tl, style in [(False, '--'), (True, '-')]:
                subset = df[(df["region"] == region_en) & (df["transfer_learning"] == tl)]
                if subset.empty:
                    continue
                grouped = subset.groupby("p")[metric].mean()
                if not grouped.empty:
                    label = f'{region_en} {"(TL)" if tl else "(Scratch)"}'
                    ax.plot(grouped.index, grouped.values, style,
                           label=label, color=self.PUBLICATION_COLORS[region_en],
                           linewidth=1.5 if tl else 1.0,
                           markersize=4)

        ax.set_xlabel('p (history length)')
        ax.set_ylabel(f'{metric.upper()}')
        ax.legend(loc='best', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3)

        # 2. Sensitivity to q
        ax = axes[0, 1]
        for region_en in self.REGIONS:
            for tl, style in [(False, '--'), (True, '-')]:
                subset = df[(df["region"] == region_en) & (df["transfer_learning"] == tl)]
                if subset.empty:
                    continue
                grouped = subset.groupby("q")[metric].mean()
                if not grouped.empty:
                    label = f'{region_en} {"(TL)" if tl else "(Scratch)"}'
                    ax.plot(grouped.index, grouped.values, style,
                           label=label, color=self.PUBLICATION_COLORS[region_en],
                           linewidth=1.5 if tl else 1.0,
                           markersize=4)

        ax.set_xlabel('q (forecast horizon)')
        ax.set_ylabel(f'{metric.upper()}')
        ax.legend(loc='best', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3)

        # 3. Transfer Learning gain
        ax = axes[1, 0]
        gains = []
        for region_en in self.REGIONS:
            df_scratch = df[(df["region"] == region_en) & (df["transfer_learning"] == False)]
            df_tl = df[(df["region"] == region_en) & (df["transfer_learning"] == True)]

            if df_scratch.empty or df_tl.empty:
                continue

            for p in df["p"].unique():
                scratch_p = df_scratch[df_scratch["p"] == p][metric].mean()
                tl_p = df_tl[df_tl["p"] == p][metric].mean()
                if not np.isnan(scratch_p) and not np.isnan(tl_p):
                    gains.append({
                        "region": region_en,
                        "p": p,
                        "gain": tl_p - scratch_p
                    })

        gains_df = pd.DataFrame(gains)
        if not gains_df.empty:
            for region_en in self.REGIONS:
                subset = gains_df[gains_df["region"] == region_en]
                if not subset.empty:
                    ax.plot(subset["p"], subset["gain"], 'o-',
                           label=region_en, color=self.PUBLICATION_COLORS[region_en],
                           linewidth=1.5, markersize=6)

        ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('p (history length)')
        ax.set_ylabel(f'TL gain ({metric.upper()})')
        ax.legend(loc='best', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3)

        # 4. Distribution by region
        ax = axes[1, 1]
        data_by_region = []
        labels = []
        for region_en in self.REGIONS:
            subset = df[(df["region"] == region_en) & (df["transfer_learning"] == True)]
            if not subset.empty:
                data_by_region.append(subset[metric].values)
                labels.append(region_en)

        if data_by_region:
            bp = ax.boxplot(data_by_region, labels=labels, patch_artist=True,
                            widths=0.6, medianprops=dict(color='black', linewidth=1.5))
            for patch, region_en in zip(bp['boxes'], self.REGIONS[:len(data_by_region)]):
                patch.set_facecolor(self.PUBLICATION_COLORS[region_en])
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(0.5)

        ax.set_xlabel('Region')
        ax.set_ylabel(f'{metric.upper()}')
        ax.grid(True, alpha=0.3, axis='y')

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.3)

        # Save (PNG + PDF)
        for ext in ("png", "pdf"):
            plt.savefig(output_dir / f'sensitivity_analysis_{metric}.{ext}', pad_inches=0.05)
        plt.close()

        self.logger.success(f"✅ Sensitivity analysis saved to: {output_dir / f'sensitivity_analysis_{metric}'}.[png/pdf]")

    def plot_tl_comparison(self, metric: str = "mcc", output_dir: Path = None) -> None:
        """
        Generate TL vs Scratch comparison plots.

        Args:
            metric: Metric to visualize
            output_dir: Output directory
        """
        self.apply_publication_style()

        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)

        if df.empty:
            self.logger.warning(f"⚠️  No data for {metric} TL comparison")
            return

        # Determine layout
        n_regions = len(self.REGIONS)
        n_cols = 3
        n_rows = (n_regions + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 5.8 * n_rows))
        axes = axes.flatten()

        for idx, region_en in enumerate(self.REGIONS):
            if idx >= len(axes):
                break

            ax = axes[idx]

            subset = df[df["region"] == region_en]
            if subset.empty:
                ax.text(0.5, 0.5, 'No data',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, style='italic')
                continue

            # Data for Scratch and TL
            scratch = subset[subset["transfer_learning"] == False]
            tl = subset[subset["transfer_learning"] == True]

            if scratch.empty or tl.empty:
                ax.text(0.5, 0.5, 'Incomplete data',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, style='italic')
                continue

            # Merge on (p, q) combinations
            merged = pd.merge(
                scratch[["p", "q", metric]],
                tl[["p", "q", metric]],
                on=["p", "q"],
                suffixes=("_scratch", "_tl")
            )

            if merged.empty:
                ax.text(0.5, 0.5, 'No common combinations',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, style='italic')
                continue

            color = self.PUBLICATION_COLORS[region_en]
            x_vals = merged[f"{metric}_scratch"]
            y_vals = merged[f"{metric}_tl"]

            # Bigger, higher-contrast markers -- the previous s=30/alpha=0.6
            # made clusters of near-identical (p,q) points merge into a
            # single pale blob (worst in regions like Northeast/North where
            # most configurations score close to zero and only one or two
            # stand out). A larger marker with a solid dark edge stays
            # legible individually and still visibly darkens where points
            # overlap.
            ax.scatter(x_vals, y_vals, alpha=0.75, s=70, color=color,
                      edgecolors='black', linewidth=0.7, zorder=3)

            # Diagonal line (y=x)
            min_val = min(x_vals.min(), y_vals.min()) - 0.05
            max_val = max(x_vals.max(), y_vals.max()) + 0.05
            min_val = max(0, min_val)
            max_val = min(1, max_val)
            ax.plot([min_val, max_val], [min_val, max_val], color='#555555',
                    linestyle='--', alpha=0.6, linewidth=1.0, zorder=1)

            # Add region label (English name)
            ax.text(0.02, 0.98, region_en,
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            # Statistics
            mean_gain = (merged[f"{metric}_tl"] - merged[f"{metric}_scratch"]).mean()
            better_count = (merged[f"{metric}_tl"] > merged[f"{metric}_scratch"]).sum()
            total_count = len(merged)
            gain_color = "#009E73" if mean_gain > 0 else "#D55E00"

            # Add statistics (colored by sign so the headline number is
            # readable without parsing the +/- sign first)
            stats_text = f"$\\Delta$ = {mean_gain:+.4f}\n{better_count}/{total_count} better"
            ax.text(0.98, 0.02, stats_text,
                   transform=ax.transAxes, fontsize=9.5, color=gain_color, fontweight='bold',
                   horizontalalignment='right', verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor=gain_color, alpha=0.9))

            ax.set_xlabel('Scratch')
            ax.set_ylabel('TL')

            # Set consistent limits
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # Remove unused subplots
        for idx in range(len(self.REGIONS), len(axes)):
            fig.delaxes(axes[idx])

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        # Save (PNG + PDF)
        for ext in ("png", "pdf"):
            plt.savefig(output_dir / f'tl_comparison_{metric}.{ext}', pad_inches=0.05)
        plt.close()

        self.logger.success(f"✅ TL comparison saved to: {output_dir / f'tl_comparison_{metric}'}.[png/pdf]")

    def plot_best_configurations(self, metric: str = "mcc", output_dir: Path = None) -> None:
        """
        Generate best configurations summary plot.

        Args:
            metric: Metric to visualize
            output_dir: Output directory
        """
        self.apply_publication_style()

        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)

        if df.empty:
            self.logger.warning("⚠️  No data for best configurations")
            return

        # Find best configurations
        best_configs = []
        for region_en in self.REGIONS:
            for tl in [False, True]:
                subset = df[(df["region"] == region_en) & (df["transfer_learning"] == tl)]
                if subset.empty:
                    continue

                # Best combination
                best_idx = subset[metric].idxmax()
                best_row = subset.loc[best_idx]

                best_configs.append({
                    "region": region_en,
                    "tl": "TL" if tl else "Scratch",
                    "p": int(best_row["p"]),
                    "q": int(best_row["q"]),
                    metric: best_row[metric]
                })

        if not best_configs:
            self.logger.warning("⚠️  No best configurations found")
            return

        best_df = pd.DataFrame(best_configs)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data
        regions = best_df["region"].unique()
        x = np.arange(len(regions))
        width = 0.35

        # Separate TL and Scratch
        tl_data = best_df[best_df["tl"] == "TL"]
        scratch_data = best_df[best_df["tl"] == "Scratch"]

        # Bars
        tl_values = [tl_data[tl_data["region"] == r][metric].values[0] if r in tl_data["region"].values else 0 for r in regions]
        scratch_values = [scratch_data[scratch_data["region"] == r][metric].values[0] if r in scratch_data["region"].values else 0 for r in regions]

        bars1 = ax.bar(x - width/2, scratch_values, width, label='Scratch',
                       color='gray', alpha=0.6, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, tl_values, width, label='TL',
                       color=[self.PUBLICATION_COLORS[r] for r in regions],
                       alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add value + configuration label as one stacked annotation (they
        # previously sat at almost the same height and overlapped into
        # unreadable text).
        y_span = max(max(scratch_values, default=0), max(tl_values, default=0)) or 1.0
        label_gap = 0.03 * y_span

        for i, region_en in enumerate(regions):
            for bar, values, data in [
                (x[i] - width/2, scratch_values, scratch_data),
                (x[i] + width/2, tl_values, tl_data),
            ]:
                row = data[data["region"] == region_en]
                if row.empty:
                    continue
                val = values[i]
                p, q = int(row.iloc[0]["p"]), int(row.iloc[0]["q"])
                ax.text(bar, val + label_gap, f'p={p}, q={q}',
                       ha='center', va='bottom', fontsize=7)
                ax.text(bar, val + label_gap * 0.35, f'{val:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Configure axes
        ax.set_xlabel('Region')
        ax.set_ylabel(f'Best {metric.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(top=y_span * 1.18)

        plt.tight_layout()

        # Save (PNG + PDF)
        for ext in ("png", "pdf"):
            plt.savefig(output_dir / f'best_configurations_{metric}.{ext}', pad_inches=0.05)
        plt.close()

        self.logger.success(f"✅ Best configurations saved to: {output_dir / f'best_configurations_{metric}'}.[png/pdf]")

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    def run(self, output_dir: Path = None) -> None:
        """
        Executes complete analysis.

        Args:
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.header("🔍 P/Q SENSITIVITY ANALYSIS")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Regions: {', '.join(self.REGIONS)}")
        print(f"  Threshold: {self.THRESHOLD_NAME} (SPI ≤ {self.THRESHOLD})")

        # Load results
        self.load_all_results()

        # Check if there is data
        has_data = any(df is not None for df in self.results.values())
        if not has_data:
            self.logger.error("❌ No data loaded. Please check the base directory.")
            return

        # Analysis for each metric
        for metric in ["mcc", "csi"]:
            self.logger.section(f"📊 Analyzing {metric.upper()}")

            # Heatmaps
            self.plot_heatmaps(metric, output_dir)

            # Sensitivity analysis
            self.plot_sensitivity_analysis(metric, output_dir)

            # TL comparison
            self.plot_tl_comparison(metric, output_dir)

            # LaTeX tables
            self.generate_latex_tables(metric, output_dir)

        # Final summary
        self.print_summary("mcc")

        self.logger.header("✅ ANALYSIS COMPLETED")
        self.logger.info(f"  Results saved to: {output_dir}")


# ========================================================================
# MAIN
# ========================================================================

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze p/q sensitivity from grid search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all regions
  python analyze_pq_sensitivity.py --base-dir ./outputs

  # Analyze specific regions
  python analyze_pq_sensitivity.py --base-dir ./outputs --regions Sul Sudeste

  # Custom output directory
  python analyze_pq_sensitivity.py --base-dir ./outputs --output-dir ./analysis
        """
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="./outputs",
        help="Base directory with grid search results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: ./outputs/analysis/pq_sensitivity)"
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=None,
        help="Regions to analyze (default: all)"
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    analyzer = PQSensitivityAnalyzer(base_dir)

    if args.regions:
        analyzer.REGIONS = args.regions

    analyzer.run(output_dir)


if __name__ == "__main__":
    main()