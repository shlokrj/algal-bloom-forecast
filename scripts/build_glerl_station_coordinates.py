#!/usr/bin/env python3
"""Build an immutable normalized GLERL station-coordinate table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from algal_bloom_forecast.data.coordinates import parse_glerl_station_coordinates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/noaa/glerl_observations/0187718/2.2/lake_erie_habs_field_sampling_master_coordinates.csv"
STATION_BBOX = {"north": 41.834, "south": 41.617, "east": -83.009, "west": -83.424}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*, input_path: Path = DEFAULT_INPUT) -> Path:
    input_path = _resolve(input_path)
    records = parse_glerl_station_coordinates(input_path, bbox=STATION_BBOX)
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = ROOT / "data/processed" / f"algal_bloom_glerl_station_coordinates_{run_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["station", "source_station", "latitude", "longitude"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "source_id": "algal_bloom_glerl_station_coordinates",
        "retrieved_at": retrieved_at.isoformat(),
        "source_path": str(input_path.relative_to(ROOT)),
        "source_sha256": _sha256(input_path),
        "output_path": str(output_path.relative_to(ROOT)),
        "records": len(records),
        "bbox": STATION_BBOX,
        "normalization": {
            "station_id": "canonicalize WE2/WE02-style labels to zero-padded WE##",
            "latitude_longitude": "validate decimal degrees and configured dataset bbox",
            "model_usage": "keep as a separate metadata table; do not spatially interpolate predictors",
        },
        "validation": {
            "duplicate_stations_rejected": True,
            "invalid_coordinates_rejected": True,
            "outside_bbox_rejected": True,
        },
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_glerl_station_coordinates_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "records": len(records)}, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    run(input_path=args.input)


if __name__ == "__main__":
    main()
