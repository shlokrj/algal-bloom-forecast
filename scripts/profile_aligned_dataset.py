#!/usr/bin/env python3
"""Validate and profile the latest aligned historical dataset."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.features.quality import profile_aligned_records

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("no aligned dataset found")
    return max(paths)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: (value if value != "" else None) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def run(*, input_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    input_path = input_path or _latest(
        list((ROOT / "data/processed").glob("algal_bloom_aligned_training_*.csv"))
    )
    records = _read_csv(input_path)
    profile = profile_aligned_records(records)
    manifest = {
        "source_id": "algal_bloom_aligned_dataset_quality",
        "retrieved_at": retrieved_at.isoformat(),
        "input_path": str(input_path.relative_to(ROOT)),
        "validation": "all predictor dates satisfy target_date minus forecast_horizon_days",
        "profile": profile,
    }
    output_path = ROOT / "data/manifests" / f"algal_bloom_aligned_quality_{run_id}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output_path}")
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"validated {len(records)} aligned rows")
    print(f"wrote manifest to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    run(input_path=args.input)


if __name__ == "__main__":
    main()
