#!/usr/bin/env python3
"""Run continuous baselines against the held-out-year evaluation table."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.models.baselines import (
    LINEAR_FEATURES,
    build_baseline_predictions,
    evaluate_baseline_predictions,
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


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "forecast_horizon_days",
        "observation_date",
        "actual",
        "climatology",
        "persistence",
        "trend",
        "linear",
        "history_date",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["split", "forecast_horizon_days", "model", "n", "mae", "rmse"]
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
    history_records = _read_csv(feature_path)
    train_records = [record for record in split_records if record.get("split") == "train"]
    evaluation_records = [
        record for record in split_records if record.get("split") in {"validation", "test"}
    ]
    predictions = build_baseline_predictions(
        train_records,
        evaluation_records,
        history_records,
        linear_features=LINEAR_FEATURES,
    )
    metrics = evaluate_baseline_predictions(predictions)
    prediction_path = ROOT / "results/tables" / f"algal_bloom_baseline_predictions_{run_id}.csv"
    metrics_path = ROOT / "results/tables" / f"algal_bloom_baseline_metrics_{run_id}.csv"
    _write_csv(prediction_path, predictions)
    _write_metrics(metrics_path, metrics)
    manifest = {
        "source_id": "algal_bloom_baseline_results",
        "retrieved_at": retrieved_at.isoformat(),
        "split_path": str(split_path.relative_to(ROOT)),
        "feature_path": str(feature_path.relative_to(ROOT)),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "target_field": "ci_sum",
        "models": ["climatology", "persistence", "trend", "linear"],
        "linear_features": list(LINEAR_FEATURES),
        "history_policy": "latest target observations at or before target_date_minus_horizon",
        "linear_policy": "fit per horizon on train rows, median-impute selected features, clamp predictions at zero",
        "metrics": metrics,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_baseline_results_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(predictions)} baseline predictions")
    print(f"wrote metrics to {metrics_path}")
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
