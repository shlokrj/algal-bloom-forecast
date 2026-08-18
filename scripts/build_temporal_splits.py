#!/usr/bin/env python3
"""Build the deterministic held-out-year train/validation/test table."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.evaluation.splits import build_temporal_splits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_YEARS = (2023,)
DEFAULT_TEST_YEARS = (2024,)


def _latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("no feature table found")
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


def _write_csv(path: Path, records: list[dict[str, Any]]) -> list[str]:
    leading = ["split", "forecast_horizon_days", "observation_date"]
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
    input_path: Path | None = None,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
    test_years: tuple[int, ...] = DEFAULT_TEST_YEARS,
) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    input_path = input_path or _latest(
        list((ROOT / "data/processed").glob("algal_bloom_feature_table_*.csv"))
    )
    records = _read_csv(input_path)
    split_records, report = build_temporal_splits(
        records,
        validation_years=validation_years,
        test_years=test_years,
    )
    output_path = ROOT / "data/processed" / f"algal_bloom_temporal_splits_{run_id}.csv"
    fields = _write_csv(output_path, split_records)
    manifest = {
        "source_id": "algal_bloom_temporal_splits",
        "retrieved_at": retrieved_at.isoformat(),
        "input_path": str(input_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "fields": fields,
        "split_strategy": "held_out_year",
        "report": report,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_temporal_splits_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(split_records)} eligible rows to {output_path}")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--validation-years", nargs="+", type=int, default=DEFAULT_VALIDATION_YEARS)
    parser.add_argument("--test-years", nargs="+", type=int, default=DEFAULT_TEST_YEARS)
    args = parser.parse_args()
    run(
        input_path=args.input,
        validation_years=tuple(args.validation_years),
        test_years=tuple(args.test_years),
    )


if __name__ == "__main__":
    main()
