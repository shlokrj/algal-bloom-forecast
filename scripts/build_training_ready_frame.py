#!/usr/bin/env python3
"""Freeze the model-input frame and schema without fitting a model."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.data.normalization import build_normalization_contract
from algal_bloom_forecast.models.dataset import build_training_frame, build_training_schema

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("no temporal split table found")
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(*, input_path: Path | None = None, target_field: str = "ci_sum") -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    input_path = _resolve(input_path) if input_path else _latest(
        list((ROOT / "data/processed").glob("algal_bloom_temporal_splits_*.csv"))
    )
    frame = build_training_frame(_read_csv(input_path), target_field=target_field)
    fields = [*frame.id_fields, frame.target_name, *frame.feature_names]
    output_path = ROOT / "data/processed" / f"algal_bloom_training_ready_{run_id}.csv"
    _write_csv(output_path, frame.rows, fields)
    manifest = {
        "source_id": "algal_bloom_training_ready",
        "retrieved_at": retrieved_at.isoformat(),
        "input_path": str(input_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "fields": fields,
        "target_definition": build_normalization_contract()["target"],
        "schema": build_training_schema(frame),
        "validation": {
            "future_predictor_dates_rejected": True,
            "duplicate_split_horizon_date_rows_rejected": True,
            "target_missing_rows_rejected": True,
            "model_fit_started": False,
        },
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_training_ready_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(frame.rows)} training-ready rows with {frame.feature_count} features")
    print(f"wrote frame to {output_path}")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--target-field", default="ci_sum")
    args = parser.parse_args()
    run(input_path=args.input, target_field=args.target_field)


if __name__ == "__main__":
    main()
