"""rolling_origin.py - Rolling-origin (multi-split) temporal evaluation.

Improvement doc, items 5-6:

    "nenhuma modificação arquitetural elimina completamente o problema de
    distribution shift... utilizar múltiplas divisões temporais históricas
    (rolling-origin evaluation) ... garantir que os períodos de validação
    incluam períodos relativamente normais, secas moderadas e secas severas."

Instead of a single Train -> Validation split (which risks selecting a
configuration that only works well for the specific 2020-2022 climate
regime and then fails on a historically extreme test period), this module
builds `n_splits` chronologically-advancing splits from the available
history and reports both per-split and aggregated metrics, so model
selection is not overfit to a single climate regime.

This module is intentionally decoupled from GridSearch/PredictorTrainer:
it only computes boolean time masks and aggregates externally-produced
per-split metric dicts, so it can be driven by GridSearch.train_and_evaluate
(or any other trainer entry point) without duplicating model/training code.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class RollingSplit:
    """One rolling-origin split."""
    index: int
    label: str
    train_mask: np.ndarray
    val_mask: np.ndarray
    train_period: str
    val_period: str


def build_rolling_origin_splits(
    years: np.ndarray,
    months: np.ndarray,
    n_splits: int = 3,
    step_months: int = 24,
    val_window_months: int = 36,
    min_train_months: int = 120,
) -> List[RollingSplit]:
    """
    Build `n_splits` chronologically-advancing (train, validation) time
    masks over the full available history.

    Split `i` (0-indexed) trains on everything up to a growing origin point
    and validates on the `val_window_months` immediately after it; the
    origin advances by `step_months` between splits, and splits are ordered
    from oldest (earliest origin) to newest (latest origin, closest to the
    real test period) so later splits naturally tend to include more
    severe/atypical regimes if the climate has been drifting - exactly the
    condition that a single fixed split misses.

    Args:
        years: Array of year values, one per month in the time series.
        months: Array of month values (1-12), one per timestep.
        n_splits: Number of rolling splits to generate.
        step_months: Months between consecutive splits' origin points.
        val_window_months: Length of each split's validation window.
        min_train_months: Minimum months of training data before the
            first split's origin.

    Returns:
        List of RollingSplit, oldest first. Splits that would need more
        data than is available are silently dropped (with fewer than
        `n_splits` returned) rather than raising, so this degrades
        gracefully on shorter regional time series (e.g. Norte).
    """
    time_idx = np.array([y * 12 + (m - 1) for y, m in zip(years, months)])
    t_min, t_max = time_idx.min(), time_idx.max()
    total_months = t_max - t_min + 1

    splits: List[RollingSplit] = []

    # Last split's origin should still leave room for a full validation
    # window; earlier splits step backward from there by step_months.
    last_origin = t_max - val_window_months
    first_origin = t_min + min_train_months

    if last_origin < first_origin:
        return splits

    # Evenly (or step_months-)spaced origins, oldest to newest.
    if n_splits <= 1:
        origins = [last_origin]
    else:
        span = last_origin - first_origin
        computed_step = span / (n_splits - 1) if span > 0 else 0
        step = max(step_months, 1) if step_months > 0 else computed_step
        origins = [first_origin + i * step for i in range(n_splits)]
        origins = [min(o, last_origin) for o in origins]

    for i, origin in enumerate(origins):
        origin = int(round(origin))
        train_mask = time_idx <= origin
        val_mask = (time_idx > origin) & (time_idx <= origin + val_window_months)

        if train_mask.sum() < min_train_months or val_mask.sum() == 0:
            continue

        train_years, train_months_arr = years[train_mask], months[train_mask]
        val_years, val_months_arr = years[val_mask], months[val_mask]

        train_period = f"{train_years[0]}-{train_months_arr[0]:02d} to {train_years[-1]}-{train_months_arr[-1]:02d}"
        val_period = f"{val_years[0]}-{val_months_arr[0]:02d} to {val_years[-1]}-{val_months_arr[-1]:02d}"

        splits.append(RollingSplit(
            index=i,
            label=f"split_{i}",
            train_mask=train_mask,
            val_mask=val_mask,
            train_period=train_period,
            val_period=val_period,
        ))

    return splits


def aggregate_rolling_metrics(
    per_split_metrics: List[Dict[str, float]],
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across rolling-origin splits.

    Args:
        per_split_metrics: One metrics dict per split (e.g. each containing
            "csi", "mcc", ...).
        metric_names: Which keys to aggregate; defaults to the keys of the
            first split's dict.

    Returns:
        Dict mapping metric name -> {"mean", "std", "min", "max",
        "worst_split"}. `min`/`worst_split` matter most here: a
        configuration that looks great on average but collapses on one
        severe-regime split is exactly the failure mode rolling-origin
        evaluation is meant to catch (see the improvement doc's point 6 -
        validation across normal/moderate/severe regimes).
    """
    if not per_split_metrics:
        return {}

    if metric_names is None:
        metric_names = list(per_split_metrics[0].keys())

    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in per_split_metrics if name in m]
        if not values:
            continue
        values_arr = np.array(values, dtype=float)
        worst_idx = int(np.argmin(values_arr))
        aggregated[name] = {
            "mean": float(values_arr.mean()),
            "std": float(values_arr.std()),
            "min": float(values_arr.min()),
            "max": float(values_arr.max()),
            "worst_split": worst_idx,
        }

    return aggregated


def run_rolling_origin_evaluation(
    years: np.ndarray,
    months: np.ndarray,
    train_eval_fn: Callable[[RollingSplit], Dict[str, float]],
    config,
    logger=None,
) -> Dict:
    """
    Drive a full rolling-origin evaluation.

    Args:
        years, months: Full time series' year/month arrays.
        train_eval_fn: Callable that receives a RollingSplit and returns a
            metrics dict (e.g. {"csi": ..., "mcc": ...}). This is where the
            caller (typically GridSearch) actually trains/evaluates a model
            on that split's train_mask/val_mask - kept as an injected
            callback so this module has no dependency on the model/trainer
            classes.
        config: ExperimentConfig (uses `config.rolling_origin`).
        logger: Optional logger with .info/.warning.

    Returns:
        Dict with "splits" (per-split period + metrics) and "aggregated"
        (see aggregate_rolling_metrics).
    """
    ro_cfg = config.rolling_origin

    splits = build_rolling_origin_splits(
        years, months,
        n_splits=ro_cfg.n_splits,
        step_months=ro_cfg.step_months,
        val_window_months=ro_cfg.val_window_months,
        min_train_months=ro_cfg.min_train_months,
    )

    if not splits:
        if logger is not None:
            logger.warning(
                "⚠️ Rolling-origin evaluation: not enough history to build "
                "any split (check rolling_origin config vs. available data length)"
            )
        return {"splits": [], "aggregated": {}}

    if logger is not None:
        logger.info(f"🔄 Rolling-origin evaluation: {len(splits)} split(s)")

    results = []
    for split in splits:
        if logger is not None:
            logger.info(
                f"   [{split.index + 1}/{len(splits)}] train={split.train_period} "
                f"| val={split.val_period}"
            )
        metrics = train_eval_fn(split)
        results.append({
            "split": split.label,
            "train_period": split.train_period,
            "val_period": split.val_period,
            "metrics": metrics,
        })

    aggregated = aggregate_rolling_metrics([r["metrics"] for r in results])

    if logger is not None and aggregated:
        for name, stats in aggregated.items():
            logger.info(
                f"   {name.upper()}: mean={stats['mean']:.4f} "
                f"min={stats['min']:.4f} (split {stats['worst_split']}) "
                f"std={stats['std']:.4f}"
            )

    return {"splits": results, "aggregated": aggregated}
