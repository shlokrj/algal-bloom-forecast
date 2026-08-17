#!/usr/bin/env python3
"""Build an immutable daily target table from local NOAA GeoTIFF files."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.satellite import build_daily_target_records

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://app.coastalscience.noaa.gov/habs_explorer/"
FIELDNAMES = [
    "observation_date",
    "acquisition_window",
    "timestamp_semantics",
    "source_filename",
    "mean_intensity",
    "valid_pixel_count",
    "total_pixel_count",
    "valid_pixel_fraction",
    "missing_reason",
]


def run(input_paths: list[Path], output_path: Path | None = None) -> Path:
    if not input_paths:
        raise ValueError("At least one GeoTIFF input is required")
    run_timestamp = datetime.now(UTC)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    records = build_daily_target_records(input_paths)
    target_path = output_path or (
        ROOT / "data/processed" / f"noaa_western_lake_erie_daily_target_{run_id}.csv"
    )
    if not target_path.is_absolute():
        target_path = ROOT / target_path
    if target_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing target table: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    manifest = ArtifactManifest.from_file(
        source_id="noaa_western_lake_erie_daily_target",
        source_url=SOURCE_URL,
        file_path=target_path,
        local_path=str(target_path.relative_to(ROOT)),
        metadata={
            "records": len(records),
            "input_filenames": [path.name for path in input_paths],
            "observation_start": records[0]["observation_date"],
            "observation_end": records[-1]["observation_date"],
            "timestamp_semantics": records[0]["timestamp_semantics"],
        },
        retrieved_at=run_timestamp,
    )
    manifest_path = ROOT / "data/manifests" / f"noaa_western_lake_erie_daily_target_{run_id}.json"
    manifest.write(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="data/raw/noaa/satellite/*.tif",
        help="glob for local NOAA GeoTIFF files",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    input_paths = sorted(ROOT.glob(args.input_glob))
    manifest_path = run(input_paths, args.output)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"{payload['source_id']}: {manifest_path}")


if __name__ == "__main__":
    main()
