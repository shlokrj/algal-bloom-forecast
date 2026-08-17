#!/usr/bin/env python3
"""Download and profile the CSV observation classes selected by an NCEI inventory."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from algal_bloom_forecast.data.glerl import profile_glerl_csv
from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.ncei import RemoteFile, download_remote_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "data/manifests/ncei_source_inventory_20260817T221557Z.json"
SOURCE_CLASSES = {"discrete_sampling", "moored_buoy_or_continuous"}


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _selected_files(
    inventory: dict[str, object], *, full_fluoroprobe_profiles: bool = False
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for source in inventory["sources"]:
        if source["collection"] != "glerl_ciglr_water_quality":
            continue
        versions = sorted(source["versions"], key=lambda value: _version_key(value["version"]))
        version = versions[-1]
        for item in version["items"]:
            if (
                item["kind"] == "file"
                and str(item["path"]).lower().endswith(".csv")
                and item["classification"] in SOURCE_CLASSES
            ):
                selected.append(
                    {
                        "accession": source["accession"],
                        "version": version["version"],
                        "label": source["label"],
                        **item,
                    }
                )
    if full_fluoroprobe_profiles:
        return selected

    non_profiles = [item for item in selected if not str(item["path"]).startswith("profiles/")]
    profile_groups: dict[tuple[str, str], dict[str, object]] = {}
    for item in selected:
        path = str(item["path"])
        match = re.search(r"fluoroprobe-([^-]+)-(\d{8})\.csv$", path)
        if not match:
            continue
        group = (match.group(1), match.group(2)[:4])
        profile_groups.setdefault(group, item)
    return non_profiles + [profile_groups[group] for group in sorted(profile_groups)]


def run(
    inventory_path: Path = DEFAULT_INVENTORY, *, full_fluoroprobe_profiles: bool = False
) -> Path:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    selected = _selected_files(
        inventory,
        full_fluoroprobe_profiles=full_fluoroprobe_profiles,
    )
    if not selected:
        raise LookupError("NCEI inventory did not contain selected GLERL observation CSVs")

    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    artifacts: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    raw_root = ROOT / "data/raw/noaa/glerl_observations"
    for index, item in enumerate(selected, start=1):
        relative_path = Path(str(item["path"]))
        local_path = raw_root / str(item["accession"]) / str(item["version"]) / relative_path
        remote_file = RemoteFile(
            name=relative_path.name,
            url=str(item["url"]),
            size_bytes=int(item["size_bytes"]),
        )
        if not local_path.exists():
            download_remote_file(remote_file, local_path)
        profile = profile_glerl_csv(local_path, source_class=str(item["classification"]))
        profile.update(
            {
                "accession": item["accession"],
                "version": item["version"],
                "listed_path": item["path"],
                "listed_size_bytes": item["size_bytes"],
            }
        )
        profiles.append(profile)
        artifact = ArtifactManifest.from_file(
            source_id="noaa_glerl_ciglr_observations",
            source_url=str(item["url"]),
            file_path=local_path,
            local_path=str(local_path.relative_to(ROOT)),
            metadata={
                "accession": item["accession"],
                "version": item["version"],
                "classification": item["classification"],
                "listed_size_bytes": item["size_bytes"],
            },
            retrieved_at=retrieved_at,
        )
        artifacts.append(asdict(artifact))
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"profiled {index}/{len(selected)} GLERL CSV files", flush=True)

    class_counts = Counter(str(profile["source_kind"]) for profile in profiles)
    report = {
        "source_id": "noaa_glerl_ciglr_observations",
        "retrieved_at": retrieved_at.isoformat(),
        "inventory_manifest": str(inventory_path.relative_to(ROOT)),
        "selection_rule": (
            "latest listed version per accession; all non-profile observation CSVs plus one "
            "fluoroprobe profile per station-year"
            if not full_fluoroprobe_profiles
            else "latest listed version per accession; all CSV files classified as discrete_sampling or moored_buoy_or_continuous"
        ),
        "file_count": len(profiles),
        "class_counts": dict(sorted(class_counts.items())),
        "profiles": profiles,
        "artifacts": artifacts,
    }
    output_path = ROOT / "data/manifests" / f"noaa_glerl_observations_{run_id}.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest: {output_path}")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--full-fluoroprobe-profiles",
        action="store_true",
        help="download every fluoroprobe profile instead of one per station-year",
    )
    args = parser.parse_args()
    print(
        run(
            args.inventory,
            full_fluoroprobe_profiles=args.full_fluoroprobe_profiles,
        )
    )


if __name__ == "__main__":
    main()
