#!/usr/bin/env python3
"""Ingest the current NOAA western Lake Erie CI-CIcyano archive listing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from build_noaa_daily_targets import run as build_daily_targets

from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.noaa_hab import (
    EXPLORER_ROOT_URL,
    download_entry,
    fetch_explorer_listing,
    find_directory_url,
    matching_downloads,
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_PATTERN = r"^sentinel-3\..*\.CI-CIcyano\.WesternLErie\.tif$"


def run() -> tuple[Path, Path]:
    run_timestamp = datetime.now(UTC)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    listing_url = find_directory_url(
        EXPLORER_ROOT_URL,
        ["data", "web", "olci_western_le", "tif_archive"],
    )
    entries = sorted(
        matching_downloads(fetch_explorer_listing(listing_url), PRODUCT_PATTERN),
        key=lambda entry: entry.label,
    )
    if not entries:
        raise LookupError("NOAA archive returned no western Lake Erie CI-CIcyano files")

    raw_paths: list[Path] = []
    artifacts: list[dict[str, object]] = []
    for entry in entries:
        raw_path = ROOT / "data/raw/noaa/satellite" / entry.label
        if not raw_path.exists():
            download_entry(entry, raw_path)
        artifact = ArtifactManifest.from_file(
            source_id="noaa_western_lake_erie_satellite",
            source_url=entry.url,
            file_path=raw_path,
            local_path=str(raw_path.relative_to(ROOT)),
            metadata={
                "filename": entry.label,
                "listing_url": listing_url,
                "listed_size": entry.size_label,
                "product_family": "CI-CIcyano.WesternLErie GeoTIFF",
            },
            retrieved_at=run_timestamp,
        )
        raw_paths.append(raw_path)
        artifacts.append(asdict(artifact))

    archive_manifest_path = (
        ROOT / "data/manifests" / f"noaa_western_lake_erie_satellite_archive_{run_id}.json"
    )
    if archive_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest: {archive_manifest_path}")
    archive_manifest_path.write_text(
        json.dumps(
            {
                "source_id": "noaa_western_lake_erie_satellite_archive",
                "listing_url": listing_url,
                "retrieved_at": run_timestamp.isoformat(),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    target_output = (
        ROOT / "data/processed" / f"noaa_western_lake_erie_daily_target_{run_id}.csv"
    )
    target_manifest_path = build_daily_targets(raw_paths, target_output)
    return archive_manifest_path, target_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    archive_manifest, target_manifest = run()
    print(json.dumps({"archive_manifest": str(archive_manifest), "target_manifest": str(target_manifest)}))


if __name__ == "__main__":
    main()
