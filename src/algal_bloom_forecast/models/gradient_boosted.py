"""Small deterministic gradient-boosted regression baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from algal_bloom_forecast.evaluation.metrics import regression_metrics


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


def _matrix(
    records: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(
        [
            [
                value if (value := _number(record.get(feature))) is not None else np.nan
                for feature in feature_names
            ]
            for record in records
        ],
        dtype=float,
    )
    medians = np.array(
        [
            float(np.nanmedian(matrix[:, index])) if np.any(~np.isnan(matrix[:, index])) else 0.0
            for index in range(matrix.shape[1])
        ]
    )
    return np.where(np.isnan(matrix), medians, matrix), medians


def _fit_models(
    train_records: Sequence[Mapping[str, Any]],
    *,
    target_field: str,
    feature_names: Sequence[str],
    random_state: int,
) -> dict[int, tuple[HistGradientBoostingRegressor, np.ndarray]]:
    models: dict[int, tuple[HistGradientBoostingRegressor, np.ndarray]] = {}
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
        matrix, medians = _matrix(rows, feature_names)
        targets = np.array([_number(record.get(target_field)) for record in rows], dtype=float)
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
        model.fit(matrix, targets)
        models[horizon] = (model, medians)
    return models


def build_gradient_boosted_predictions(
    train_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    target_field: str = "ci_sum",
    feature_names: Sequence[str],
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Fit one gradient-boosted regressor per horizon on training rows only."""
    if not feature_names:
        raise ValueError("feature_names must not be empty")
    models = _fit_models(
        train_records,
        target_field=target_field,
        feature_names=feature_names,
        random_state=random_state,
    )
    predictions: list[dict[str, Any]] = []
    for record in evaluation_records:
        horizon = _horizon(record.get("forecast_horizon_days"))
        fitted = models.get(horizon)
        prediction: float | None = None
        if fitted is not None:
            model, medians = fitted
            matrix, _ = _matrix([record], feature_names)
            prediction = max(
                0.0, float(model.predict(np.where(np.isnan(matrix), medians, matrix))[0])
            )
        predictions.append(
            {
                "split": record.get("split"),
                "forecast_horizon_days": horizon,
                "observation_date": record.get("observation_date"),
                "actual": _number(record.get(target_field)),
                "gradient_boosted": prediction,
            }
        )
    return predictions


def evaluate_gradient_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return MAE/RMSE by split and horizon for the tree baseline."""
    groups: dict[tuple[str, int], list[tuple[Any, Any]]] = {}
    for prediction in predictions:
        key = (str(prediction["split"]), _horizon(prediction["forecast_horizon_days"]))
        groups.setdefault(key, []).append(
            (prediction.get("actual"), prediction.get("gradient_boosted"))
        )
    return [
        {
            "split": split,
            "forecast_horizon_days": horizon,
            "model": "gradient_boosted",
            **regression_metrics(pairs),
        }
        for (split, horizon), pairs in sorted(groups.items())
    ]
