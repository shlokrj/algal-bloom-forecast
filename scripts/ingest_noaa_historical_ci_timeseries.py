#!/usr/bin/env python3
"""Ingest the compact historical Western Lake Erie CIcyano target series."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.ncei import RemoteFile, download_remote_file, list_ftp_directory
from algal_bloom_forecast.data.noaa_ci_timeseries import (
    merge_ci_timeseries,
    parse_ci_timeseries,
    profile_ci_timeseries,
)

ROOT = Path(__file__).resolve().parents[1]
ACCESSION = "0312614"
ACCESSION_URL = f"https://www.ncei.noaa.gov/archive/accession/{ACCESSION}"
DATA_ROOT = (
    "ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0243/0312614/1.1/data/0-data/"
    "HAB_GreatLakes_Monitoring/intensity-extent-timeseries-csvs/GLErie/"
)
TARGET_NAMES = {
    "Western Lake Erie_BloomArea_sqkm_timeseries.csv",
    "Western Lake Erie_CIsum_timeseries.csv",
}


def _csv_value(value: object | None) -> str:
    return "" if value is None else str(value)


def run() -> tuple[Path, Path]:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    remote_files = {file.name: file for file in list_ftp_directory(DATA_ROOT)}
    missing = sorted(TARGET_NAMES - set(remote_files))
    if missing:
        raise LookupError(f"Missing historical NOAA CIcyano files: {missing}")

    raw_root = ROOT / "data/raw/noaa/satellite/historical_ci_timeseries"
    artifacts: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    parsed_series: dict[str, list[dict[str, object | None]]] = {}
    for name in sorted(TARGET_NAMES):
        remote_file = remote_files[name]
        remote_file = RemoteFile(
            name=remote_file.name,
            url=f"{DATA_ROOT}{quote(remote_file.name)}",
            size_bytes=remote_file.size_bytes,
        )
        local_path = raw_root / name
        if not local_path.exists():
            download_remote_file(remote_file, local_path)
        metric, rows, _ = parse_ci_timeseries(local_path)
        parsed_series[metric] = rows
        profile = profile_ci_timeseries(local_path)
        profile.update(
            {
                "accession": ACCESSION,
                "source_url": remote_file.url,
                "listed_size_bytes": remote_file.size_bytes,
            }
        )
        profiles.append(profile)
        artifacts.append(
            asdict(
                ArtifactManifest.from_file(
                    source_id="noaa_western_lake_erie_historical_ci_timeseries",
                    source_url=remote_file.url,
                    file_path=local_path,
                    local_path=str(local_path.relative_to(ROOT)),
                    metadata={
                        "accession": ACCESSION,
                        "metric": metric,
                        "time_basis": "date-only 10-day composite center date; no timezone",
                        "listed_size_bytes": remote_file.size_bytes,
                    },
                    retrieved_at=retrieved_at,
                )
            )
        )

    merged_rows = merge_ci_timeseries(parsed_series)
    table_path = ROOT / "data/processed" / f"noaa_western_lake_erie_historical_target_{run_id}.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("observation_date,ci_sum,bloom_area_sqkm\n")
        for row in merged_rows:
            handle.write(
                f"{row['observation_date']},{_csv_value(row.get('ci_sum'))},{_csv_value(row.get('bloom_area_sqkm'))}\n"
            )

    derived_table = ArtifactManifest.from_file(
        source_id="noaa_western_lake_erie_historical_target_table",
        source_url=ACCESSION_URL,
        file_path=table_path,
        local_path=str(table_path.relative_to(ROOT)),
        metadata={
            "records": len(merged_rows),
            "target_semantics": "10-day composite center date; no interpolation",
            "metrics": ["ci_sum", "bloom_area_sqkm"],
        },
        retrieved_at=retrieved_at,
    )
    manifest = {
        "source_id": "noaa_western_lake_erie_historical_ci_timeseries",
        "accession": ACCESSION,
        "accession_url": ACCESSION_URL,
        "doi": "https://doi.org/10.25921/wzk1-r208",
        "data_root": DATA_ROOT,
        "retrieved_at": retrieved_at.isoformat(),
        "target_semantics": "cross-sensor fused 10-day composite center dates; no timezone",
        "profiles": profiles,
        "artifacts": artifacts,
        "derived_table": asdict(derived_table),
    }
    manifest_path = (
        ROOT / "data/manifests" / f"noaa_western_lake_erie_historical_ci_timeseries_{run_id}.json"
    )
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path, table_path


def main() -> None:
    manifest_path, table_path = run()
    print(json.dumps({"manifest": str(manifest_path), "table": str(table_path)}))


if __name__ == "__main__":
    main()
