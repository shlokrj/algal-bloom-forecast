#!/usr/bin/env python3
"""Evaluate provisional high-intensity event metrics for baseline predictions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.evaluation.events import evaluate_event_predictions, fit_event_thresholds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUANTILE = 0.75
MODEL_NAMES = ("climatology", "persistence", "trend", "linear")


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


def _write_metrics(path: Path, records: list[dict[str, Any]]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_predictions(
    path: Path, predictions: list[dict[str, Any]], thresholds: dict[int, float]
) -> None:
    fields = [
        "split",
        "forecast_horizon_days",
        "observation_date",
        "actual",
        "event_threshold",
        "actual_event",
    ]
    for model_name in MODEL_NAMES:
        fields.extend([f"{model_name}_score", f"{model_name}_event"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            horizon = int(prediction["forecast_horizon_days"])
            threshold = thresholds[horizon]
            actual = prediction.get("actual")
            row = {
                "split": prediction["split"],
                "forecast_horizon_days": horizon,
                "observation_date": prediction["observation_date"],
                "actual": actual,
                "event_threshold": threshold,
                "actual_event": int(actual >= threshold) if actual is not None else None,
            }
            for model_name in MODEL_NAMES:
                score = prediction.get(model_name)
                row[f"{model_name}_score"] = score
                row[f"{model_name}_event"] = int(score >= threshold) if score is not None else None
            writer.writerow(row)


def run(
    *,
    predictions_path: Path | None = None,
    split_path: Path | None = None,
    quantile: float = DEFAULT_QUANTILE,
) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    predictions_path = predictions_path or _latest(
        list((ROOT / "results/tables").glob("algal_bloom_baseline_predictions_*.csv")),
        "baseline predictions",
    )
    split_path = split_path or _latest(
        list((ROOT / "data/processed").glob("algal_bloom_temporal_splits_*.csv")),
        "temporal split",
    )
    predictions = _read_csv(predictions_path)
    split_records = _read_csv(split_path)
    train_records = [record for record in split_records if record.get("split") == "train"]
    thresholds = fit_event_thresholds(train_records, quantile=quantile)
    metrics = evaluate_event_predictions(predictions, thresholds, model_names=MODEL_NAMES)
    prediction_path = ROOT / "results/tables" / f"algal_bloom_event_predictions_{run_id}.csv"
    metrics_path = ROOT / "results/tables" / f"algal_bloom_event_metrics_{run_id}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _write_predictions(prediction_path, predictions, thresholds)
    _write_metrics(metrics_path, metrics)
    manifest = {
        "source_id": "algal_bloom_event_results",
        "retrieved_at": retrieved_at.isoformat(),
        "predictions_path": str(predictions_path.relative_to(ROOT)),
        "split_path": str(split_path.relative_to(ROOT)),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "target_field": "ci_sum",
        "event_definition": "provisional high-intensity event at or above the train-only target quantile",
        "threshold_policy": "train_quantile",
        "train_quantile": quantile,
        "thresholds_by_horizon": {
            str(horizon): threshold for horizon, threshold in thresholds.items()
        },
        "probability_policy": "hard 0/1 probability from intensity threshold; not calibrated probability",
        "metrics": metrics,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_event_results_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(metrics)} event metric rows")
    print(f"wrote metrics to {metrics_path}")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    args = parser.parse_args()
    run(
        predictions_path=args.predictions,
        split_path=args.splits,
        quantile=args.quantile,
    )


if __name__ == "__main__":
    main()
