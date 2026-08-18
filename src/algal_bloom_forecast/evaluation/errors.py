"""Error and event-case diagnostics for held-out forecasts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def classify_event_error(actual: Any, prediction: Any, threshold: float) -> str:
    """Classify a forecast as a correct event, false alarm, or miss."""
    if actual in (None, "") or prediction in (None, ""):
        return "missing_score"
    observed = float(actual) >= threshold
    predicted = float(prediction) >= threshold
    if observed and predicted:
        return "true_positive"
    if not observed and predicted:
        return "false_alarm"
    if observed and not predicted:
        return "missed_event"
    return "true_negative"


def build_error_records(
    predictions: Sequence[Mapping[str, Any]],
    thresholds: Mapping[int, float],
) -> list[dict[str, Any]]:
    """Add signed, absolute, and event-case diagnostics to predictions."""
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        horizon = int(prediction["forecast_horizon_days"])
        actual = prediction.get("actual")
        score = prediction.get("prediction")
        signed_error = None
        absolute_error = None
        if actual not in (None, "") and score not in (None, ""):
            signed_error = float(score) - float(actual)
            absolute_error = abs(signed_error)
        rows.append(
            {
                **dict(prediction),
                "signed_error": signed_error,
                "absolute_error": absolute_error,
                "event_case": classify_event_error(actual, score, thresholds[horizon]),
                "event_threshold": thresholds[horizon],
            }
        )
    return rows


def build_error_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize errors and event cases by horizon."""
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["forecast_horizon_days"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for horizon, horizon_rows in sorted(grouped.items()):
        absolute_errors = [
            float(row["absolute_error"])
            for row in horizon_rows
            if row.get("absolute_error") not in (None, "")
        ]
        cases = Counter(str(row["event_case"]) for row in horizon_rows)
        worst = max(
            horizon_rows,
            key=lambda row: float(row.get("absolute_error") or -1.0),
        )
        summaries.append(
            {
                "forecast_horizon_days": horizon,
                "n": len(horizon_rows),
                "mae": sum(absolute_errors) / len(absolute_errors) if absolute_errors else None,
                "max_absolute_error": max(absolute_errors) if absolute_errors else None,
                "worst_observation_date": worst.get("observation_date"),
                "true_positive": cases["true_positive"],
                "true_negative": cases["true_negative"],
                "false_alarm": cases["false_alarm"],
                "missed_event": cases["missed_event"],
            }
        )
    return summaries
