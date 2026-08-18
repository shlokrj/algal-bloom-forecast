"""Combine baseline results into deterministic horizon-performance summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_summary_rows(
    regression_sources: Mapping[str, Sequence[Mapping[str, Any]]],
    event_sources: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Normalize regression and event result tables into long-form rows."""
    rows: list[dict[str, Any]] = []
    for source_name, source_rows in regression_sources.items():
        for record in source_rows:
            model = str(record["model"])
            if source_name == "feature_ablation":
                model = f"linear:{record['feature_group']}"
            for metric in ("mae", "rmse"):
                rows.append(
                    {
                        "metric_family": "continuous",
                        "source": source_name,
                        "split": record["split"],
                        "forecast_horizon_days": int(float(record["forecast_horizon_days"])),
                        "model": model,
                        "metric": metric,
                        "value": _number(record.get(metric)),
                        "n": int(record["n"]),
                    }
                )
    for source_name, source_rows in event_sources.items():
        for record in source_rows:
            for metric in ("pr_auc", "precision", "recall", "f1", "brier", "calibration_abs_error"):
                rows.append(
                    {
                        "metric_family": "event",
                        "source": source_name,
                        "split": record["split"],
                        "forecast_horizon_days": int(float(record["forecast_horizon_days"])),
                        "model": str(record["model"]),
                        "metric": metric,
                        "value": _number(record.get(metric)),
                        "n": int(record["n"]),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["metric_family"],
            row["split"],
            row["forecast_horizon_days"],
            row["metric"],
            row["model"],
        ),
    )


def best_by_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_family: str,
    metric: str,
    lower_is_better: bool,
) -> dict[str, dict[str, Any]]:
    """Select the best non-null result for every split/horizon pair."""
    candidates = [
        row
        for row in rows
        if row["metric_family"] == metric_family
        and row["metric"] == metric
        and row.get("value") is not None
    ]
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault((str(row["split"]), int(row["forecast_horizon_days"])), []).append(row)
    best: dict[str, dict[str, Any]] = {}
    for (split, horizon), group in sorted(grouped.items()):
        selected = min(
            group,
            key=lambda row: (
                (float(row["value"]), str(row["model"]))
                if lower_is_better
                else (-float(row["value"]), str(row["model"]))
            ),
        )
        best[f"{split}/{horizon}"] = {
            "split": split,
            "forecast_horizon_days": horizon,
            "model": selected["model"],
            "value": selected["value"],
            "n": selected["n"],
        }
    return best


def build_best_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Return best continuous and event metrics without tuning on test results."""
    return {
        "continuous_best_mae": best_by_metric(
            rows,
            metric_family="continuous",
            metric="mae",
            lower_is_better=True,
        ),
        "continuous_best_rmse": best_by_metric(
            rows,
            metric_family="continuous",
            metric="rmse",
            lower_is_better=True,
        ),
        "event_best_pr_auc": best_by_metric(
            rows,
            metric_family="event",
            metric="pr_auc",
            lower_is_better=False,
        ),
        "event_best_f1": best_by_metric(
            rows,
            metric_family="event",
            metric="f1",
            lower_is_better=False,
        ),
    }
