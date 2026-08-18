#!/usr/bin/env python3
"""Audit the frozen training-ready frame before any new model fit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def validate(manifest_path: Path, frame_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest["schema"]
    if schema["status"] != "prepared_not_fitted":
        raise ValueError("training-ready manifest is not marked prepared_not_fitted")
    if manifest["validation"]["model_fit_started"]:
        raise ValueError("training-ready manifest says model fitting has started")

    with frame_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    expected_fields = [*schema["id_fields"], schema["target"], *schema["feature_names"]]
    if fields != expected_fields:
        raise ValueError("training-ready CSV fields do not match its manifest schema")
    if len(rows) != schema["rows"]:
        raise ValueError("training-ready CSV row count does not match its manifest schema")

    seen: set[tuple[str, int, str]] = set()
    target_name = str(schema["target"])
    future_rows: list[str] = []
    for row in rows:
        key = (row["split"], int(float(row["forecast_horizon_days"])), row["observation_date"])
        if key in seen:
            raise ValueError(f"duplicate training-ready row: {key}")
        seen.add(key)
        if row[target_name] == "":
            raise ValueError(f"missing training target in row: {key}")
        observation = date.fromisoformat(row["observation_date"])
        predictor = date.fromisoformat(row["predictor_date"])
        cutoff = observation - timedelta(days=key[1])
        if predictor > cutoff:
            future_rows.append(f"{key}: {predictor} > {cutoff}")
    if future_rows:
        raise ValueError(f"future predictor dates found: {future_rows[0]}")
    forbidden = {"ci_sum", "target_ci_sum", "bloom_area_sqkm"}
    leakage_fields = sorted(forbidden & set(schema["feature_names"]))
    if leakage_fields:
        raise ValueError(f"target fields were included as predictors: {leakage_fields}")

    split_counts = Counter(row["split"] for row in rows)
    if dict(sorted(split_counts.items())) != schema["split_counts"]:
        raise ValueError("training-ready split counts do not match its manifest schema")
    return {
        "status": schema["status"],
        "rows": len(rows),
        "features": schema["feature_count"],
        "split_counts": dict(sorted(split_counts.items())),
        "future_predictor_rows": 0,
        "model_fit_started": manifest["validation"]["model_fit_started"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frame", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_training_ready_*.json")),
        "training-ready manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_path = args.frame or ROOT / manifest["output_path"]
    report = validate(manifest_path, frame_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
