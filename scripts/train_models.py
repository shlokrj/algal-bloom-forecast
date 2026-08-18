#!/usr/bin/env python3
"""Train and evaluate the first full-schema candidate model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.evaluation.events import evaluate_event_predictions, fit_event_thresholds
from algal_bloom_forecast.models.training import (
    evaluate_predictions,
    predict_records,
    save_horizon_model,
    train_horizon_models,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 7,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


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


def _write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*, frame_path: Path | None = None, manifest_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_path or _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_training_ready_*.json")),
        "training-ready manifest",
    )
    manifest_path = manifest_path.resolve()
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_path = (frame_path or ROOT / input_manifest["output_path"]).resolve()
    schema = input_manifest["schema"]
    target_field = str(schema["target"])
    feature_names = tuple(schema["feature_names"])
    records = _read_csv(frame_path)
    train_records = [record for record in records if record["split"] == "train"]
    evaluation_records = [record for record in records if record["split"] != "train"]
    models = train_horizon_models(
        train_records,
        target_field=target_field,
        feature_names=feature_names,
        hyperparameters=DEFAULT_HYPERPARAMETERS,
    )
    model_dir = ROOT / "models"
    model_paths: list[dict[str, Any]] = []
    for horizon, bundle in sorted(models.items()):
        model_path = model_dir / f"algal_bloom_hist_gradient_boosted_{run_id}_h{horizon}.joblib"
        save_horizon_model(bundle, str(model_path))
        model_paths.append(
            {
                "horizon": horizon,
                "path": str(model_path.relative_to(ROOT)),
                "sha256": _sha256(model_path),
                "size_bytes": model_path.stat().st_size,
                "training_rows": bundle["training_rows"],
            }
        )
    predictions = predict_records(
        models,
        evaluation_records,
        target_field=target_field,
    )
    regression_metrics = evaluate_predictions(predictions)
    thresholds = fit_event_thresholds(train_records, target_field=target_field)
    event_metrics = evaluate_event_predictions(
        predictions,
        thresholds,
        model_names=("hist_gradient_boosted",),
    )
    prediction_path = ROOT / "results/tables" / f"algal_bloom_trained_predictions_{run_id}.csv"
    metrics_path = ROOT / "results/tables" / f"algal_bloom_trained_metrics_{run_id}.csv"
    event_path = ROOT / "results/tables" / f"algal_bloom_trained_event_metrics_{run_id}.csv"
    prediction_fields = [
        "split",
        "forecast_horizon_days",
        "observation_date",
        "actual",
        "hist_gradient_boosted",
    ]
    metric_fields = ["split", "forecast_horizon_days", "model", "n", "mae", "rmse"]
    event_fields = [
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
    _write_csv(prediction_path, predictions, prediction_fields)
    _write_csv(metrics_path, regression_metrics, metric_fields)
    _write_csv(event_path, event_metrics, event_fields)
    manifest = {
        "source_id": "algal_bloom_training_run",
        "retrieved_at": retrieved_at.isoformat(),
        "training_ready_manifest": str(manifest_path.relative_to(ROOT)),
        "frame_path": str(frame_path.relative_to(ROOT)),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "event_metrics_path": str(event_path.relative_to(ROOT)),
        "target_field": target_field,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "model": "hist_gradient_boosted",
        "hyperparameters": DEFAULT_HYPERPARAMETERS,
        "preprocessing": {
            "missing_value_policy": "feature medians fit on training rows per horizon; no validation or test values used",
            "prediction_floor": 0.0,
        },
        "fit_policy": "one model per horizon fitted only on train split rows",
        "models": model_paths,
        "regression_metrics": regression_metrics,
        "event_thresholds": {str(horizon): value for horizon, value in thresholds.items()},
        "event_metrics": event_metrics,
    }
    output_manifest = ROOT / "data/manifests" / f"algal_bloom_training_run_{run_id}.json"
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output_manifest}")
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"trained {len(models)} horizon models with {len(feature_names)} features")
    print(f"wrote predictions to {prediction_path}")
    print(f"wrote manifest to {output_manifest}")
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    run(frame_path=args.frame, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
