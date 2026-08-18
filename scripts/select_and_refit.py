#!/usr/bin/env python3
"""Select by validation, refit on train plus validation, and score test once."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.evaluation.events import evaluate_event_predictions, fit_event_thresholds
from algal_bloom_forecast.evaluation.metrics import regression_metrics
from algal_bloom_forecast.evaluation.selection import select_by_validation_mae
from algal_bloom_forecast.models.baselines import build_baseline_predictions
from algal_bloom_forecast.models.feature_groups import build_feature_groups
from algal_bloom_forecast.models.gradient_boosted import build_gradient_boosted_predictions

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
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


def _candidate_metrics() -> tuple[list[dict[str, Any]], dict[str, str]]:
    sources = {
        "baseline": _latest(
            list((ROOT / "data/manifests").glob("algal_bloom_baseline_results_*.json")),
            "baseline manifest",
        ),
        "feature_ablation": _latest(
            list((ROOT / "data/manifests").glob("algal_bloom_feature_ablation_*.json")),
            "feature ablation manifest",
        ),
        "gradient_boosted": _latest(
            list((ROOT / "data/manifests").glob("algal_bloom_gradient_baseline_*.json")),
            "gradient baseline manifest",
        ),
        "hist_gradient_boosted": _latest(
            list((ROOT / "data/manifests").glob("algal_bloom_training_run_*.json")),
            "training run manifest",
        ),
    }
    rows: list[dict[str, Any]] = []
    for source, path in sources.items():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        metric_rows = manifest.get("regression_metrics", manifest.get("metrics", []))
        for metric in metric_rows:
            row = {
                "source": source,
                "candidate": str(metric["model"]),
                "split": metric["split"],
                "forecast_horizon_days": int(float(metric["forecast_horizon_days"])),
                "n": int(metric["n"]),
                "mae": metric["mae"],
                "rmse": metric["rmse"],
            }
            if source == "feature_ablation":
                row["candidate"] = f"linear:{metric['feature_group']}"
            rows.append(row)
    return rows, {source: str(path.relative_to(ROOT)) for source, path in sources.items()}


def _metrics_by_horizon(
    predictions: list[dict[str, Any]],
    *,
    prediction_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[tuple[Any, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["split"]), int(row["forecast_horizon_days"]))].append(
            (row["actual"], row[prediction_field])
        )
    return [
        {
            "split": split,
            "forecast_horizon_days": horizon,
            "model": "selected_model",
            **regression_metrics(pairs),
        }
        for (split, horizon), pairs in sorted(grouped.items())
    ]


def run(
    *,
    split_path: Path | None = None,
    feature_path: Path | None = None,
) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    split_path = split_path or _latest(
        list((ROOT / "data/processed").glob("algal_bloom_temporal_splits_*.csv")),
        "temporal split table",
    )
    feature_path = feature_path or _latest(
        list((ROOT / "data/processed").glob("algal_bloom_feature_table_*.csv")),
        "feature table",
    )
    split_records = _read_csv(split_path)
    feature_records = _read_csv(feature_path)
    candidates, candidate_sources = _candidate_metrics()
    selected = select_by_validation_mae(candidates)
    fit_records = [row for row in split_records if row["split"] in {"train", "validation"}]
    test_records = [row for row in split_records if row["split"] == "test"]
    feature_groups = build_feature_groups(split_records)

    baseline_predictions = build_baseline_predictions(
        fit_records,
        test_records,
        feature_records,
        linear_features=feature_groups["combined_core"],
    )
    gradient_predictions = build_gradient_boosted_predictions(
        fit_records,
        test_records,
        feature_names=feature_groups["combined_core"],
    )
    baseline_by_key = {
        (int(row["forecast_horizon_days"]), row["observation_date"]): row
        for row in baseline_predictions
    }
    gradient_by_key = {
        (int(row["forecast_horizon_days"]), row["observation_date"]): row
        for row in gradient_predictions
    }
    prediction_rows: list[dict[str, Any]] = []
    for record in test_records:
        horizon = int(record["forecast_horizon_days"])
        key = (horizon, record["observation_date"])
        candidate = selected[horizon]
        if candidate == "linear:combined_core":
            prediction = baseline_by_key[key]["linear"]
        elif candidate in {"climatology", "persistence", "trend"}:
            prediction = baseline_by_key[key][candidate]
        elif candidate == "linear":
            prediction = baseline_by_key[key]["linear"]
        elif candidate == "gradient_boosted":
            prediction = gradient_by_key[key]["gradient_boosted"]
        else:
            raise ValueError(f"refit implementation does not support selected candidate {candidate}")
        prediction_rows.append(
            {
                "split": "test",
                "forecast_horizon_days": horizon,
                "observation_date": record["observation_date"],
                "selected_model": candidate,
                "actual": record["ci_sum"],
                "prediction": prediction,
            }
        )

    regression_metrics = _metrics_by_horizon(prediction_rows, prediction_field="prediction")
    thresholds = fit_event_thresholds(fit_records, target_field="ci_sum")
    event_input = [
        {**row, "selected_model": row["prediction"]} for row in prediction_rows
    ]
    event_metrics = evaluate_event_predictions(
        event_input,
        thresholds,
        model_names=("selected_model",),
    )
    candidate_path = ROOT / "results/tables" / f"algal_bloom_model_selection_candidates_{run_id}.csv"
    candidate_fields = [
        "source",
        "candidate",
        "split",
        "forecast_horizon_days",
        "n",
        "mae",
        "rmse",
        "selected",
    ]
    for row in candidates:
        row["selected"] = row["candidate"] == selected.get(row["forecast_horizon_days"])
    _write_csv(candidate_path, candidates, candidate_fields)
    prediction_path = ROOT / "results/tables" / f"algal_bloom_selected_predictions_{run_id}.csv"
    metrics_path = ROOT / "results/tables" / f"algal_bloom_selected_metrics_{run_id}.csv"
    event_path = ROOT / "results/tables" / f"algal_bloom_selected_event_metrics_{run_id}.csv"
    _write_csv(
        prediction_path,
        prediction_rows,
        [
            "split",
            "forecast_horizon_days",
            "observation_date",
            "selected_model",
            "actual",
            "prediction",
        ],
    )
    _write_csv(metrics_path, regression_metrics, ["split", "forecast_horizon_days", "model", "n", "mae", "rmse"])
    _write_csv(
        event_path,
        event_metrics,
        [
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
        ],
    )
    manifest = {
        "source_id": "algal_bloom_model_selection",
        "retrieved_at": retrieved_at.isoformat(),
        "split_path": str(split_path.relative_to(ROOT)),
        "feature_path": str(feature_path.relative_to(ROOT)),
        "candidate_sources": candidate_sources,
        "candidate_metrics_path": str(candidate_path.relative_to(ROOT)),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "event_metrics_path": str(event_path.relative_to(ROOT)),
        "selection_metric": "validation_mae",
        "selection_policy": "select independently per horizon using validation only; do not inspect test metrics",
        "selected_models": {str(horizon): model for horizon, model in sorted(selected.items())},
        "refit_policy": "refit selected rules/models on train plus validation rows, then score test once",
        "fit_rows": len(fit_records),
        "test_rows": len(test_records),
        "regression_metrics": regression_metrics,
        "event_thresholds": {str(horizon): value for horizon, value in thresholds.items()},
        "event_metrics": event_metrics,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_model_selection_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"selected models: {manifest['selected_models']}")
    print(f"wrote test forecasts to {prediction_path}")
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
