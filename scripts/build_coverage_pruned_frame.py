#!/usr/bin/env python3
"""Build a lower-missingness training frame using train-only feature selection."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.models.feature_selection import select_by_train_missing_rate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_MISSING_RATE = 0.5


def _base_training_ready_manifests() -> list[Path]:
    return [
        path
        for path in (ROOT / "data/manifests").glob("algal_bloom_training_ready_*.json")
        if "coverage_pruned" not in path.name
    ]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    input_path: Path | None = None,
    input_manifest_path: Path | None = None,
    max_missing_rate: float = DEFAULT_MAX_MISSING_RATE,
) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    input_manifest_path = input_manifest_path or _latest(
        _base_training_ready_manifests(),
        "training-ready manifest",
    )
    input_manifest_path = input_manifest_path.resolve()
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    input_path = (input_path or ROOT / input_manifest["output_path"]).resolve()
    rows = _read_csv(input_path)
    schema = input_manifest["schema"]
    feature_names, train_missing_rates = select_by_train_missing_rate(
        rows,
        schema["feature_names"],
        max_missing_rate=max_missing_rate,
    )
    fields = [*schema["id_fields"], schema["target"], *feature_names]
    output_rows = [{field: row.get(field, "") for field in fields} for row in rows]
    output_path = (
        ROOT / "data/processed" / f"algal_bloom_training_ready_coverage_pruned_{run_id}.csv"
    )
    _write_csv(output_path, output_rows, fields)
    manifest = {
        "source_id": "algal_bloom_training_ready_coverage_pruned",
        "retrieved_at": retrieved_at.isoformat(),
        "input_manifest": str(input_manifest_path.relative_to(ROOT)),
        "input_path": str(input_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "fields": fields,
        "schema": {
            "status": "coverage_pruned_prepared_not_fitted",
            "target": schema["target"],
            "id_fields": schema["id_fields"],
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "excluded_fields": sorted(set(schema["feature_names"]) - set(feature_names)),
        },
        "selection_policy": {
            "criterion": "feature missing rate on train rows only",
            "max_missing_rate": max_missing_rate,
            "train_missing_rate_by_feature": train_missing_rates,
        },
        "model_fit_started": False,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_training_ready_coverage_pruned_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"selected {len(feature_names)} of {len(schema['feature_names'])} features")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-missing-rate", type=float, default=DEFAULT_MAX_MISSING_RATE)
    args = parser.parse_args()
    run(
        input_path=args.input,
        input_manifest_path=args.manifest,
        max_missing_rate=args.max_missing_rate,
    )


if __name__ == "__main__":
    main()
