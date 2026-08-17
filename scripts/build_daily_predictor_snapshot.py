#!/usr/bin/env python3
"""Build an immutable daily predictor snapshot from the ingested source slices."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.data.glerl import aggregate_glerl_continuous
from algal_bloom_forecast.data.ndbc import (
    aggregate_standard_meteorology,
    parse_standard_meteorology,
)
from algal_bloom_forecast.data.usgs import parse_daily_values_payload
from algal_bloom_forecast.features.daily import merge_daily_predictor_records

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} source found")
    return max(paths)


def _coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [str(record["observation_date"]) for record in records]
    return {
        "daily_records": len(records),
        "observed_start": min(dates) if dates else None,
        "observed_end": max(dates) if dates else None,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> list[str]:
    fields = ["observation_date"] + sorted(
        {field for record in records for field in record if field != "observation_date"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return fields


def run(*, usgs_path: Path | None = None, ndbc_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    usgs_path = usgs_path or _latest(list((ROOT / "data/raw/usgs").glob("*.json")), "USGS")
    ndbc_path = ndbc_path or _latest(list((ROOT / "data/raw/ndbc").glob("*.txt.gz")), "NDBC")
    glerl_paths = sorted((ROOT / "data/raw/noaa/glerl_observations").rglob("*_annual_summary.csv"))
    if not glerl_paths:
        raise FileNotFoundError("no GLERL annual-summary sources found")

    usgs_records = parse_daily_values_payload(json.loads(usgs_path.read_text(encoding="utf-8")))
    ndbc_raw_records = parse_standard_meteorology(ndbc_path)
    ndbc_records = aggregate_standard_meteorology(ndbc_raw_records)
    glerl_records = aggregate_glerl_continuous(glerl_paths)
    records = merge_daily_predictor_records(
        {"usgs": usgs_records, "ndbc": ndbc_records, "glerl": glerl_records}
    )

    output_path = ROOT / "data/processed" / f"algal_bloom_daily_predictors_{run_id}.csv"
    fields = _write_csv(output_path, records)
    manifest = {
        "source_id": "algal_bloom_daily_predictors",
        "retrieved_at": retrieved_at.isoformat(),
        "output_path": str(output_path.relative_to(ROOT)),
        "fields": fields,
        "daily_coverage": _coverage(records),
        "missing_value_policy": "outer join by UTC calendar date; no interpolation",
        "sources": {
            "usgs": {
                "local_path": str(usgs_path.relative_to(ROOT)),
                **_coverage(usgs_records),
            },
            "ndbc": {
                "local_path": str(ndbc_path.relative_to(ROOT)),
                "raw_records": len(ndbc_raw_records),
                **_coverage(ndbc_records),
                "time_basis": "UTC",
            },
            "glerl": {
                "local_paths": [str(path.relative_to(ROOT)) for path in glerl_paths],
                **_coverage(glerl_records),
                "file_count": len(glerl_paths),
                "time_basis": "UTC as documented by annual-summary timestamp rows",
            },
        },
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_daily_predictors_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} daily rows to {output_path}")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usgs", type=Path)
    parser.add_argument("--ndbc", type=Path)
    args = parser.parse_args()
    run(usgs_path=args.usgs, ndbc_path=args.ndbc)


if __name__ == "__main__":
    main()
