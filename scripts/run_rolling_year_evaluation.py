#!/usr/bin/env python3
"""Evaluate baselines with an expanding, year-by-year training window."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.data.normalization import build_normalization_contract
from algal_bloom_forecast.evaluation.metrics import regression_metrics
from algal_bloom_forecast.models.baselines import build_baseline_predictions
from algal_bloom_forecast.models.feature_groups import build_feature_groups
from algal_bloom_forecast.models.gradient_boosted import build_gradient_boosted_predictions

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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


def run(*, split_path: Path | None = None, feature_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    split_path = _resolve(split_path) if split_path else _latest(
        list((ROOT / "data/processed").glob("algal_bloom_temporal_splits_*.csv")),
        "temporal split table",
    )
    feature_path = _resolve(feature_path) if feature_path else _latest(
        list((ROOT / "data/processed").glob("algal_bloom_feature_table_*.csv")),
        "feature table",
    )
    records = _read_csv(split_path)
    history_records = _read_csv(feature_path)
    years = sorted({int(str(row["observation_date"])[:4]) for row in records})
    feature_groups = build_feature_groups(records)
    output_rows: list[dict[str, Any]] = []
    for evaluation_year in years:
        train_records = [
            row for row in records if int(str(row["observation_date"])[:4]) < evaluation_year
        ]
        evaluation_records = [
            row for row in records if int(str(row["observation_date"])[:4]) == evaluation_year
        ]
        if not train_records or not evaluation_records:
            continue
        baseline_predictions = build_baseline_predictions(
            train_records,
            evaluation_records,
            history_records,
            linear_features=feature_groups["combined_core"],
        )
        gradient_predictions = build_gradient_boosted_predictions(
            train_records,
            evaluation_records,
            feature_names=feature_groups["combined_core"],
        )
        predictions_by_model = {
            "climatology": baseline_predictions,
            "persistence": baseline_predictions,
            "trend": baseline_predictions,
            "linear:combined_core": baseline_predictions,
            "gradient_boosted": gradient_predictions,
        }
        for model, predictions in predictions_by_model.items():
            grouped: dict[int, list[tuple[Any, Any]]] = defaultdict(list)
            for prediction in predictions:
                prediction_field = "linear" if model == "linear:combined_core" else model
                grouped[int(prediction["forecast_horizon_days"])].append(
                    (prediction["actual"], prediction[prediction_field])
                )
            for horizon, pairs in sorted(grouped.items()):
                output_rows.append(
                    {
                        "evaluation_year": evaluation_year,
                        "train_through_year": evaluation_year - 1,
                        "forecast_horizon_days": horizon,
                        "model": model,
                        **regression_metrics(pairs),
                    }
                )
    output_rows.sort(
        key=lambda row: (
            int(row["evaluation_year"]),
            int(row["forecast_horizon_days"]),
            str(row["model"]),
        )
    )
    output_path = ROOT / "results/tables" / f"algal_bloom_rolling_year_metrics_{run_id}.csv"
    _write_csv(
        output_path,
        output_rows,
        [
            "evaluation_year",
            "train_through_year",
            "forecast_horizon_days",
            "model",
            "n",
            "mae",
            "rmse",
        ],
    )
    manifest = {
        "source_id": "algal_bloom_rolling_year_evaluation",
        "retrieved_at": retrieved_at.isoformat(),
        "split_path": str(split_path.relative_to(ROOT)),
        "feature_path": str(feature_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "target_definition": build_normalization_contract()["target"],
        "evaluation_years": sorted({row["evaluation_year"] for row in output_rows}),
        "training_policy": "expanding window; each evaluation year trains only on earlier years",
        "models": ["climatology", "persistence", "trend", "linear:combined_core", "gradient_boosted"],
        "rows": len(output_rows),
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_rolling_year_evaluation_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(output_rows)} rolling-year metric rows")
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
