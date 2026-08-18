"""Provisional high-intensity event labels and classification metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _horizon(value: Any) -> int:
    horizon = int(value)
    if horizon <= 0:
        raise ValueError("forecast_horizon_days must be positive")
    return horizon


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile from no values")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def fit_event_thresholds(
    train_records: Sequence[Mapping[str, Any]],
    *,
    target_field: str = "ci_sum",
    quantile: float = 0.75,
) -> dict[int, float]:
    """Fit one provisional high-intensity threshold per horizon from train only."""
    grouped: dict[int, list[float]] = {}
    for record in train_records:
        value = _number(record.get(target_field))
        if value is not None:
            grouped.setdefault(_horizon(record.get("forecast_horizon_days")), []).append(value)
    return {horizon: _quantile(values, quantile) for horizon, values in sorted(grouped.items())}


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        true_positives += label
        if label:
            precision_sum += true_positives / rank
    return precision_sum / positives


def event_metrics(
    pairs: Sequence[tuple[Any, Any]],
    *,
    event_threshold: float,
) -> dict[str, float | int | None]:
    """Evaluate intensity scores as thresholded high-intensity events.

    The Brier and calibration fields use hard probabilities (0 or 1) because
    these baselines produce intensity scores, not calibrated event probabilities.
    """
    valid = [
        (float(actual), float(score))
        for actual, score in pairs
        if _number(actual) is not None and _number(score) is not None
    ]
    if not valid:
        return {
            "n": 0,
            "positive_events": 0,
            "predicted_events": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "pr_auc": None,
            "brier": None,
            "observed_event_rate": None,
            "predicted_event_rate": None,
            "calibration_abs_error": None,
        }
    labels = [int(actual >= event_threshold) for actual, _ in valid]
    scores = [score for _, score in valid]
    predictions = [int(score >= event_threshold) for score in scores]
    true_positives = sum(label and prediction for label, prediction in zip(labels, predictions))
    false_positives = sum(
        not label and prediction for label, prediction in zip(labels, predictions)
    )
    false_negatives = sum(
        label and not prediction for label, prediction in zip(labels, predictions)
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else None
    )
    observed_rate = sum(labels) / len(labels)
    predicted_rate = sum(predictions) / len(predictions)
    return {
        "n": len(valid),
        "positive_events": sum(labels),
        "predicted_events": sum(predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": _average_precision(labels, scores),
        "brier": sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels))
        / len(labels),
        "observed_event_rate": observed_rate,
        "predicted_event_rate": predicted_rate,
        "calibration_abs_error": abs(predicted_rate - observed_rate),
    }


def evaluate_event_predictions(
    predictions: Sequence[Mapping[str, Any]],
    thresholds: Mapping[int, float],
    *,
    model_names: Sequence[str] = ("climatology", "persistence", "trend", "linear"),
) -> list[dict[str, Any]]:
    """Evaluate baseline intensity scores as events by split and horizon."""
    grouped: dict[tuple[str, int, str], list[tuple[Any, Any]]] = {}
    for prediction in predictions:
        split = str(prediction["split"])
        horizon = _horizon(prediction["forecast_horizon_days"])
        if horizon not in thresholds:
            raise ValueError(f"no event threshold for horizon {horizon}")
        for model_name in model_names:
            grouped.setdefault((split, horizon, model_name), []).append(
                (prediction.get("actual"), prediction.get(model_name))
            )
    results: list[dict[str, Any]] = []
    for (split, horizon, model_name), pairs in sorted(grouped.items()):
        results.append(
            {
                "split": split,
                "forecast_horizon_days": horizon,
                "model": model_name,
                "event_threshold": thresholds[horizon],
                **event_metrics(pairs, event_threshold=thresholds[horizon]),
            }
        )
    return results
