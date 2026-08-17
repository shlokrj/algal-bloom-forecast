#!/usr/bin/env python3
"""Download one NOAA western Lake Erie GeoTIFF and write its manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.noaa_hab import (
    EXPLORER_ROOT_URL,
    download_entry,
    fetch_explorer_listing,
    find_directory_url,
    matching_downloads,
)


ROOT = Path(__file__).resolve().parents[1]
WESTERN_CI_CYANO_PATTERN = r"^sentinel-3\..*\.CI-CIcyano\.WesternLErie\.tif$"


def run(filename: str | None = None) -> Path:
    run_timestamp = datetime.now(timezone.utc)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    listing_url = find_directory_url(
        EXPLORER_ROOT_URL,
        ["data", "web", "olci_western_le", "tif_archive"],
    )
    matches = matching_downloads(
        fetch_explorer_listing(listing_url),
        WESTERN_CI_CYANO_PATTERN,
    )
    if filename is not None:
        matches = [entry for entry in matches if entry.label == filename]
    if not matches:
        requested = filename or WESTERN_CI_CYANO_PATTERN
        raise LookupError(f"No NOAA satellite file matched {requested!r}")

    entry = matches[0]
    raw_path = ROOT / "data/raw/noaa/satellite" / entry.label
    download_entry(entry, raw_path)
    manifest = ArtifactManifest.from_file(
        source_id="noaa_western_lake_erie_satellite",
        source_url=entry.url,
        file_path=raw_path,
        local_path=str(raw_path.relative_to(ROOT)),
        metadata={
            "filename": entry.label,
            "listing_url": listing_url,
            "listed_size": entry.size_label,
            "product_family": "CI-CIcyano.WesternLErie GeoTIFF",
            "target_semantics": "pending_pixel_profile",
        },
        retrieved_at=run_timestamp,
    )
    manifest_path = ROOT / "data/manifests" / f"noaa_western_lake_erie_satellite_{run_id}.json"
    manifest.write(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="download this exact archive filename")
    args = parser.parse_args()
    manifest_path = run(args.filename)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"{payload['source_id']}: {manifest_path}")


if __name__ == "__main__":
    main()
