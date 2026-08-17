#!/usr/bin/env python3
"""Download and profile representative NCEI GLERL/CIGLR fluoroprobe files."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.noaa_glerl import (
    FLUOROPROBE_DATA_ROOT,
    FLUOROPROBE_PROFILES_ROOT,
    download_remote_file,
    list_ftp_directory,
    parse_fluoroprobe_coordinates,
    parse_fluoroprobe_dictionary,
    profile_fluoroprobe_csv,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PROFILE_NAMES = (
    "noaa-glerl-fluoroprobe-WE12-20170711.csv",
    "noaa-glerl-fluoroprobe-WE12-20200713.csv",
    "noaa-glerl-fluoroprobe-WE12-20220817.csv",
)


def run() -> Path:
    run_timestamp = datetime.now(UTC)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    root_files = {file.name: file for file in list_ftp_directory(FLUOROPROBE_DATA_ROOT)}
    profile_files = {
        file.name: file for file in list_ftp_directory(FLUOROPROBE_PROFILES_ROOT)
    }
    required_names = {
        "lake_erie_habs_fluoroprobe_data_dictionary_2017-2022.csv",
        "lake_erie_habs_fluoroprobe_master_coordinates_2017-2022.csv",
        *SAMPLE_PROFILE_NAMES,
    }
    available_names = set(root_files) | set(profile_files)
    missing_names = sorted(required_names - available_names)
    if missing_names:
        raise LookupError(f"Missing expected GLERL/CIGLR files: {missing_names}")

    selected_files = [
        root_files["lake_erie_habs_fluoroprobe_data_dictionary_2017-2022.csv"],
        root_files["lake_erie_habs_fluoroprobe_master_coordinates_2017-2022.csv"],
        *(profile_files[name] for name in SAMPLE_PROFILE_NAMES),
    ]
    raw_directory = ROOT / "data/raw/noaa/glerl_fluoroprobe"
    artifacts: list[dict[str, object]] = []
    local_paths: dict[str, Path] = {}
    for remote_file in selected_files:
        local_path = raw_directory / remote_file.name
        if not local_path.exists():
            download_remote_file(remote_file, local_path)
        local_paths[remote_file.name] = local_path
        artifact = ArtifactManifest.from_file(
            source_id="noaa_glerl_ciglr_fluoroprobe",
            source_url=remote_file.url,
            file_path=local_path,
            local_path=str(local_path.relative_to(ROOT)),
            metadata={
                "listed_size_bytes": remote_file.size_bytes,
                "subset": "fluoroprobe_depth_profiles",
                "time_basis": "Eastern Daylight Time as documented by the data dictionary",
            },
            retrieved_at=run_timestamp,
        )
        artifacts.append(asdict(artifact))

    dictionary_path = local_paths[
        "lake_erie_habs_fluoroprobe_data_dictionary_2017-2022.csv"
    ]
    coordinates_path = local_paths[
        "lake_erie_habs_fluoroprobe_master_coordinates_2017-2022.csv"
    ]
    profile_paths = [local_paths[name] for name in SAMPLE_PROFILE_NAMES]
    report = {
        "source_id": "noaa_glerl_ciglr_fluoroprobe_sample",
        "source_url": FLUOROPROBE_DATA_ROOT,
        "retrieved_at": run_timestamp.isoformat(),
        "subset": "fluoroprobe_depth_profiles",
        "time_basis": "Eastern Daylight Time as documented by the data dictionary",
        "dictionary_fields": parse_fluoroprobe_dictionary(dictionary_path),
        "coordinate_count": len(parse_fluoroprobe_coordinates(coordinates_path)),
        "profiles": [profile_fluoroprobe_csv(path) for path in profile_paths],
        "artifacts": artifacts,
    }
    manifest_path = (
        ROOT / "data/manifests" / f"noaa_glerl_ciglr_fluoroprobe_sample_{run_id}.json"
    )
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
