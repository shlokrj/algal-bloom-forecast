#!/usr/bin/env python3
"""Run the combined-core gradient-boosted continuous and event baseline."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.evaluation.events import evaluate_event_predictions, fit_event_thresholds
from algal_bloom_forecast.models.feature_groups import build_feature_groups
from algal_bloom_forecast.models.gradient_boosted import (
    build_gradient_boosted_predictions,
    evaluate_gradient_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} artifact found")
    return max(paths)


def _coerce_value(field: str, value: str | None) -> Any:
    if value in (None, ""):
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    if field.endswith(("_count", "_flag")):
        try:
            return int(value)
        except ValueError:
            pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _coerce_value(key, value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "forecast_horizon_days",
        "observation_date",
        "actual",
        "gradient_boosted",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_regression_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["split", "forecast_horizon_days", "model", "n", "mae", "rmse"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_event_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "forecast_horizon_days",
        "model",
        "event_threshold",
        "n",
        "positive_events",
        "predicted_events",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "brier",
        "observed_event_rate",
        "predicted_event_rate",
        "calibration_abs_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run(*, split_path: Path | None = None, feature_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    split_path = split_path or _latest(
        list((ROOT / "data/processed").glob("algal_bloom_temporal_splits_*.csv")),
        "temporal split",
    )
    feature_path = feature_path or _latest(
        list((ROOT / "data/processed").glob("algal_bloom_feature_table_*.csv")),
        "feature table",
    )
    split_records = _read_csv(split_path)
    train_records = [record for record in split_records if record.get("split") == "train"]
    evaluation_records = [
        record for record in split_records if record.get("split") in {"validation", "test"}
    ]
    feature_names = build_feature_groups(split_records)["combined_core"]
    predictions = build_gradient_boosted_predictions(
        train_records,
        evaluation_records,
        feature_names=feature_names,
    )
    regression_metrics = evaluate_gradient_predictions(predictions)
    thresholds = fit_event_thresholds(train_records)
    event_metrics = evaluate_event_predictions(
        predictions,
        thresholds,
        model_names=("gradient_boosted",),
    )
    prediction_path = ROOT / "results/tables" / f"algal_bloom_gradient_predictions_{run_id}.csv"
    regression_path = ROOT / "results/tables" / f"algal_bloom_gradient_metrics_{run_id}.csv"
    event_path = ROOT / "results/tables" / f"algal_bloom_gradient_event_metrics_{run_id}.csv"
    _write_predictions(prediction_path, predictions)
    _write_regression_metrics(regression_path, regression_metrics)
    _write_event_metrics(event_path, event_metrics)
    manifest = {
        "source_id": "algal_bloom_gradient_baseline_results",
        "retrieved_at": retrieved_at.isoformat(),
        "split_path": str(split_path.relative_to(ROOT)),
        "feature_path": str(feature_path.relative_to(ROOT)),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "regression_metrics_path": str(regression_path.relative_to(ROOT)),
        "event_metrics_path": str(event_path.relative_to(ROOT)),
        "target_field": "ci_sum",
        "feature_group": "combined_core",
        "feature_names": feature_names,
        "hyperparameters": {
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": 42,
        },
        "fit_policy": "one model per horizon fitted on training rows only",
        "regression_metrics": regression_metrics,
        "event_thresholds": {str(horizon): value for horizon, value in thresholds.items()},
        "event_metrics": event_metrics,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_gradient_baseline_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(predictions)} gradient predictions")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    run(split_path=args.splits, feature_path=args.features)


if __name__ == "__main__":
    main()
