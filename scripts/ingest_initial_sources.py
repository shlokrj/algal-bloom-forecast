#!/usr/bin/env python3
"""Download the first reproducible source slices and write immutable manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen

from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.ndbc import (
    build_standard_meteorology_url,
    parse_standard_meteorology,
)
from algal_bloom_forecast.data.usgs import DailyValuesQuery, fetch_daily_values


ROOT = Path(__file__).resolve().parents[1]


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "algal-bloom-forecast/0.1"})
    with urlopen(request, timeout=120) as response:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.read())


def run(start_date: str, end_date: str, ndbc_year: int) -> list[Path]:
    run_timestamp = datetime.now(timezone.utc)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    manifest_paths: list[Path] = []

    usgs_query = DailyValuesQuery(
        monitoring_location_id="USGS-04193500",
        parameter_code="00060",
        statistic_id="00003",
        start_date=start_date,
        end_date=end_date,
    )
    usgs_path = ROOT / "data/raw/usgs" / f"maumee_daily_discharge_{run_id}.json"
    usgs_payload, usgs_url = fetch_daily_values(usgs_query, output_path=usgs_path)
    usgs_features = usgs_payload.get("features", [])
    usgs_times = sorted(feature["properties"]["time"] for feature in usgs_features)
    usgs_manifest = ArtifactManifest.from_file(
        source_id="usgs_maumee_daily_discharge",
        source_url=usgs_url,
        file_path=usgs_path,
        local_path=str(usgs_path.relative_to(ROOT)),
        metadata={
            "records": len(usgs_features),
            "observed_start": usgs_times[0] if usgs_times else None,
            "observed_end": usgs_times[-1] if usgs_times else None,
            "monitoring_location_id": usgs_query.monitoring_location_id,
            "parameter_code": usgs_query.parameter_code,
            "statistic_id": usgs_query.statistic_id,
        },
        retrieved_at=run_timestamp,
    )
    usgs_manifest_path = ROOT / "data/manifests" / f"usgs_maumee_daily_discharge_{run_id}.json"
    usgs_manifest.write(usgs_manifest_path)
    manifest_paths.append(usgs_manifest_path)

    ndbc_url = build_standard_meteorology_url("45005", ndbc_year)
    ndbc_path = ROOT / "data/raw/ndbc" / f"45005_stdmet_{ndbc_year}_{run_id}.txt.gz"
    download_file(ndbc_url, ndbc_path)
    ndbc_records = parse_standard_meteorology(ndbc_path)
    ndbc_manifest = ArtifactManifest.from_file(
        source_id="noaa_ndbc_45005_standard_meteorology",
        source_url=ndbc_url,
        file_path=ndbc_path,
        local_path=str(ndbc_path.relative_to(ROOT)),
        metadata={
            "station": "45005",
            "year": ndbc_year,
            "records": len(ndbc_records),
            "observed_start": ndbc_records[0]["timestamp"] if ndbc_records else None,
            "observed_end": ndbc_records[-1]["timestamp"] if ndbc_records else None,
            "time_basis": "UTC",
        },
        retrieved_at=run_timestamp,
    )
    ndbc_manifest_path = ROOT / "data/manifests" / f"noaa_ndbc_45005_{ndbc_year}_{run_id}.json"
    ndbc_manifest.write(ndbc_manifest_path)
    manifest_paths.append(ndbc_manifest_path)

    return manifest_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2012-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--ndbc-year", type=int, default=2024)
    args = parser.parse_args()

    for manifest_path in run(args.start_date, args.end_date, args.ndbc_year):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"{payload['source_id']}: {manifest_path}")


if __name__ == "__main__":
    main()
