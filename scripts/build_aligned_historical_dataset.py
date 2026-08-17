#!/usr/bin/env python3
"""Align the historical target with daily predictors for every forecast horizon."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.features.alignment import align_daily_predictors_to_targets

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HORIZONS = (1, 3, 7, 14)


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} table found")
    return max(paths)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: (value if value != "" else None) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _write_csv(path: Path, records: list[dict[str, Any]]) -> list[str]:
    leading = ["forecast_horizon_days", "observation_date", "predictor_date", "feature_lag_days"]
    remaining = sorted({field for record in records for field in record if field not in leading})
    fields = leading + remaining
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return fields


def run(
    *,
    target_path: Path | None = None,
    predictor_path: Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> Path:
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain only positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")

    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    target_path = target_path or _latest(
        list((ROOT / "data/processed").glob("noaa_western_lake_erie_historical_target_*.csv")),
        "historical target",
    )
    predictor_path = predictor_path or _latest(
        list((ROOT / "data/processed").glob("algal_bloom_daily_predictors_*.csv")),
        "daily predictor",
    )
    target_records = _read_csv(target_path)
    predictor_records = _read_csv(predictor_path)

    aligned_records: list[dict[str, Any]] = []
    horizon_profiles: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        aligned = align_daily_predictors_to_targets(
            target_records,
            predictor_records,
            horizon_days=horizon,
        )
        for record in aligned:
            aligned_records.append({"forecast_horizon_days": horizon, **record})
        horizon_profiles[str(horizon)] = {
            "target_records": len(aligned),
            "predictor_matches": sum(record["predictor_date"] is not None for record in aligned),
            "missing_ci_sum": sum(record.get("ci_sum") is None for record in aligned),
            "missing_bloom_area_sqkm": sum(
                record.get("bloom_area_sqkm") is None for record in aligned
            ),
            "predictor_date_rule": f"latest predictor date <= target date minus {horizon} days",
        }

    output_path = ROOT / "data/processed" / f"algal_bloom_aligned_training_{run_id}.csv"
    fields = _write_csv(output_path, aligned_records)
    manifest = {
        "source_id": "algal_bloom_aligned_training",
        "retrieved_at": retrieved_at.isoformat(),
        "target_path": str(target_path.relative_to(ROOT)),
        "predictor_path": str(predictor_path.relative_to(ROOT)),
        "horizons_days": list(horizons),
        "fields": fields,
        "records": len(aligned_records),
        "alignment_policy": {
            "predictor_cutoff": "target_date_minus_horizon",
            "missing_target_policy": "preserve_null",
            "interpolation": "disabled",
        },
        "horizon_profiles": horizon_profiles,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_aligned_training_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(aligned_records)} aligned rows to {output_path}")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path)
    parser.add_argument("--predictors", type=Path)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    args = parser.parse_args()
    run(
        target_path=args.target,
        predictor_path=args.predictors,
        horizons=tuple(args.horizons),
    )


if __name__ == "__main__":
    main()
