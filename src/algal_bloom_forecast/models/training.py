"""Train and score the first unfitted candidate model from the frozen frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from algal_bloom_forecast.evaluation.metrics import regression_metrics


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _horizon(value: Any) -> int:
    horizon = int(float(value))
    if horizon <= 0:
        raise ValueError("forecast_horizon_days must be positive")
    return horizon


def _matrix(
    records: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    return np.array(
        [
            [
                value if (value := _number(record.get(feature))) is not None else np.nan
                for feature in feature_names
            ]
            for record in records
        ],
        dtype=float,
    )


def fit_medians(matrix: np.ndarray) -> np.ndarray:
    """Fit one missing-value replacement per feature on training rows only."""
    medians = np.empty(matrix.shape[1], dtype=float)
    for index in range(matrix.shape[1]):
        observed = matrix[:, index][~np.isnan(matrix[:, index])]
        medians[index] = float(np.median(observed)) if observed.size else 0.0
    return medians


def train_horizon_models(
    train_records: Sequence[Mapping[str, Any]],
    *,
    target_field: str,
    feature_names: Sequence[str],
    hyperparameters: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Fit one deterministic histogram-gradient model per forecast horizon."""
    if not feature_names:
        raise ValueError("feature_names must not be empty")
    models: dict[int, dict[str, Any]] = {}
    horizons = sorted({_horizon(record["forecast_horizon_days"]) for record in train_records})
    for horizon in horizons:
        rows = [
            record
            for record in train_records
            if _horizon(record["forecast_horizon_days"]) == horizon
            and _number(record.get(target_field)) is not None
        ]
        if not rows:
            continue
        matrix = _matrix(rows, feature_names)
        medians = fit_medians(matrix)
        model = HistGradientBoostingRegressor(**dict(hyperparameters))
        model.fit(np.where(np.isnan(matrix), medians, matrix), [record[target_field] for record in rows])
        models[horizon] = {
            "model": model,
            "medians": medians,
            "feature_names": tuple(feature_names),
            "target_field": target_field,
            "horizon": horizon,
            "training_rows": len(rows),
        }
    return models


def save_horizon_model(model_bundle: Mapping[str, Any], path: str) -> None:
    """Persist one fitted model bundle for later reproducible inference."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model_bundle), path)


def predict_records(
    models: Mapping[int, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    target_field: str,
    prediction_field: str = "hist_gradient_boosted",
) -> list[dict[str, Any]]:
    """Score held-out records with the model fitted for each horizon."""
    predictions: list[dict[str, Any]] = []
    for record in records:
        horizon = _horizon(record["forecast_horizon_days"])
        bundle = models.get(horizon)
        prediction: float | None = None
        if bundle is not None:
            feature_names = bundle["feature_names"]
            matrix = _matrix([record], feature_names)
            matrix = np.where(np.isnan(matrix), bundle["medians"], matrix)
            prediction = max(0.0, float(bundle["model"].predict(matrix)[0]))
        predictions.append(
            {
                "split": record.get("split"),
                "forecast_horizon_days": horizon,
                "observation_date": record.get("observation_date"),
                "actual": _number(record.get(target_field)),
                prediction_field: prediction,
            }
        )
    return predictions


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    prediction_field: str = "hist_gradient_boosted",
) -> list[dict[str, Any]]:
    """Return continuous metrics grouped by held-out split and horizon."""
    groups: dict[tuple[str, int], list[tuple[Any, Any]]] = {}
    for prediction in predictions:
        key = (str(prediction["split"]), _horizon(prediction["forecast_horizon_days"]))
        groups.setdefault(key, []).append(
            (prediction.get("actual"), prediction.get(prediction_field))
        )
    return [
        {
            "split": split,
            "forecast_horizon_days": horizon,
            "model": prediction_field,
            **regression_metrics(pairs),
        }
        for (split, horizon), pairs in sorted(groups.items())
    ]
