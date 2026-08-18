"""Simple leakage-safe continuous forecasting baselines."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np

from algal_bloom_forecast.evaluation.metrics import regression_metrics

LINEAR_FEATURES = (
    "usgs_maumee_discharge_m3s",
    "seasonal_day_of_year_sin",
    "seasonal_day_of_year_cos",
)


def _date(value: Any) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError("observation_date must be an ISO date string")
    return date.fromisoformat(value)


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


def _target_history(
    records: Sequence[Mapping[str, Any]], target_field: str
) -> list[tuple[date, float]]:
    by_date: dict[date, float] = {}
    for record in records:
        value = _number(record.get(target_field))
        if value is not None:
            by_date[_date(record.get("observation_date"))] = value
    return sorted(by_date.items())


def _history_before(
    history: Sequence[tuple[date, float]],
    *,
    target_date: date,
    horizon: int,
) -> list[tuple[date, float]]:
    cutoff = target_date - timedelta(days=horizon)
    dates = [item[0] for item in history]
    return list(history[: bisect_right(dates, cutoff)])


def _climatology(
    train_records: Sequence[Mapping[str, Any]], target_field: str
) -> dict[int, float | None]:
    grouped: dict[int, list[float]] = {}
    for record in train_records:
        value = _number(record.get(target_field))
        if value is not None:
            grouped.setdefault(_horizon(record.get("forecast_horizon_days")), []).append(value)
    return {
        horizon: sum(values) / len(values) if values else None
        for horizon, values in grouped.items()
    }


def _fit_linear(
    train_records: Sequence[Mapping[str, Any]],
    *,
    target_field: str,
    feature_names: Sequence[str],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    fitted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    horizons = sorted({_horizon(record.get("forecast_horizon_days")) for record in train_records})
    for horizon in horizons:
        rows = [
            record
            for record in train_records
            if _horizon(record.get("forecast_horizon_days")) == horizon
            and _number(record.get(target_field)) is not None
        ]
        if not rows:
            continue
        matrix = np.array(
            [
                [
                    value if (value := _number(record.get(feature))) is not None else np.nan
                    for feature in feature_names
                ]
                for record in rows
            ],
            dtype=float,
        )
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        matrix = np.where(np.isnan(matrix), medians, matrix)
        design = np.column_stack([np.ones(len(rows)), matrix])
        target = np.array([_number(record.get(target_field)) for record in rows], dtype=float)
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        fitted[horizon] = (medians, coefficients)
    return fitted


def _linear_prediction(
    record: Mapping[str, Any],
    *,
    fitted: tuple[np.ndarray, np.ndarray] | None,
    feature_names: Sequence[str],
) -> float | None:
    if fitted is None:
        return None
    medians, coefficients = fitted
    values = np.array(
        [
            _number(record.get(feature)) if _number(record.get(feature)) is not None else np.nan
            for feature in feature_names
        ],
        dtype=float,
    )
    values = np.where(np.isnan(values), medians, values)
    prediction = float(np.r_[1.0, values] @ coefficients)
    return max(0.0, prediction)


def build_baseline_predictions(
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    history_records: Sequence[Mapping[str, Any]],
    *,
    target_field: str = "ci_sum",
    linear_features: Sequence[str] = LINEAR_FEATURES,
) -> list[dict[str, Any]]:
    """Generate four baseline predictions for validation and test rows."""
    climatology = _climatology(train_records, target_field)
    linear = _fit_linear(
        train_records,
        target_field=target_field,
        feature_names=linear_features,
    )
    history = _target_history(history_records, target_field)
    predictions: list[dict[str, Any]] = []
    for record in evaluation_records:
        target_date = _date(record.get("observation_date"))
        horizon = _horizon(record.get("forecast_horizon_days"))
        previous = _history_before(history, target_date=target_date, horizon=horizon)
        persistence = previous[-1][1] if previous else None
        trend = persistence
        if len(previous) >= 2:
            first_date, first_value = previous[-2]
            last_date, last_value = previous[-1]
            elapsed_days = (last_date - first_date).days
            if elapsed_days > 0:
                slope = (last_value - first_value) / elapsed_days
                trend = max(0.0, last_value + slope * (target_date - last_date).days)
        predictions.append(
            {
                "split": record.get("split"),
                "forecast_horizon_days": horizon,
                "observation_date": record.get("observation_date"),
                "actual": _number(record.get(target_field)),
                "climatology": climatology.get(horizon),
                "persistence": persistence,
                "trend": trend,
                "linear": _linear_prediction(
                    record,
                    fitted=linear.get(horizon),
                    feature_names=linear_features,
                ),
                "history_date": previous[-1][0].isoformat() if previous else None,
            }
        )
    return predictions


def evaluate_baseline_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    model_names: Sequence[str] = ("climatology", "persistence", "trend", "linear"),
) -> list[dict[str, Any]]:
    """Return MAE/RMSE by split, horizon, and baseline model."""
    groups: dict[tuple[str, int, str], list[tuple[Any, Any]]] = {}
    for prediction in predictions:
        split = str(prediction["split"])
        horizon = _horizon(prediction["forecast_horizon_days"])
        actual = prediction.get("actual")
        for model_name in model_names:
            groups.setdefault((split, horizon, model_name), []).append(
                (actual, prediction.get(model_name))
            )
    results: list[dict[str, Any]] = []
    for (split, horizon, model_name), pairs in sorted(groups.items()):
        results.append(
            {
                "split": split,
                "forecast_horizon_days": horizon,
                "model": model_name,
                **regression_metrics(pairs),
            }
        )
    return results
