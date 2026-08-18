#!/usr/bin/env python3
"""Download reproducible NDBC standard-meteorology history for one station."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.ndbc import (
    build_standard_meteorology_url,
    parse_standard_meteorology,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATION = "45005"
DEFAULT_START_YEAR = 2012
DEFAULT_END_YEAR = 2024
_YEAR_FILE = re.compile(r"^(?P<station>\d+)_stdmet_(?P<year>\d{4})_.+\.txt\.gz$")


def download_file(url: str, destination: Path) -> None:
    """Download to a temporary sibling and publish only a complete file."""
    request = Request(url, headers={"User-Agent": "algal-bloom-forecast/0.1"})
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        with urlopen(request, timeout=120) as response:
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(response.read())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _existing_paths(station: str, start_year: int, end_year: int) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in (ROOT / "data/raw/ndbc").glob(f"{station}_stdmet_*.txt.gz"):
        match = _YEAR_FILE.match(path.name)
        if not match or match.group("station") != station:
            continue
        year = int(match.group("year"))
        if start_year <= year <= end_year:
            paths[year] = max(path, paths.get(year, path))
    return paths


def _profile(path: Path, *, station: str, year: int, url: str, retrieved_at: datetime) -> dict:
    records = parse_standard_meteorology(path)
    artifact = ArtifactManifest.from_file(
        source_id="noaa_ndbc_45005_standard_meteorology",
        source_url=url,
        file_path=path,
        local_path=str(path.relative_to(ROOT)),
        metadata={
            "station": station,
            "year": year,
            "records": len(records),
            "observed_start": records[0]["timestamp"] if records else None,
            "observed_end": records[-1]["timestamp"] if records else None,
            "time_basis": "UTC",
        },
        retrieved_at=retrieved_at,
    )
    return {"year": year, "status": "available", "artifact": asdict(artifact)}


def run(
    *,
    station: str = DEFAULT_STATION,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> Path:
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    existing = _existing_paths(station, start_year, end_year)
    year_results: list[dict] = []
    for year in range(start_year, end_year + 1):
        url = build_standard_meteorology_url(station, year)
        path = existing.get(year)
        try:
            if path is None:
                path = ROOT / "data/raw/ndbc" / f"{station}_stdmet_{year}_{run_id}.txt.gz"
                download_file(url, path)
                source = "downloaded"
            else:
                source = "existing"
            profile = _profile(path, station=station, year=year, url=url, retrieved_at=retrieved_at)
            profile["source"] = source
            year_results.append(profile)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            year_results.append(
                {
                    "year": year,
                    "status": "unavailable",
                    "source_url": url,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    available = [result for result in year_results if result["status"] == "available"]
    if not available:
        raise RuntimeError("no NDBC history files were available")
    manifest = {
        "source_id": "noaa_ndbc_45005_history",
        "retrieved_at": retrieved_at.isoformat(),
        "station": station,
        "requested_years": [start_year, end_year],
        "available_years": [result["year"] for result in available],
        "unavailable_years": [
            result["year"] for result in year_results if result["status"] == "unavailable"
        ],
        "time_basis": "UTC",
        "year_results": year_results,
        "download_policy": "reuse the latest local file for each year; never overwrite raw artifacts",
    }
    manifest_path = ROOT / "data/manifests" / f"noaa_ndbc_45005_history_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"available NDBC years: {manifest['available_years']}")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    args = parser.parse_args()
    run(station=args.station, start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
