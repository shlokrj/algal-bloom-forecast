#!/usr/bin/env python3
"""Validate immutable georeferenced spatial map outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.data.spatial import validate_exported_map_arrays

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile(dataset: Any) -> dict[str, Any]:
    return {
        "width": int(dataset.width),
        "height": int(dataset.height),
        "count": int(dataset.count),
        "dtype": dataset.dtypes[0],
        "crs": str(dataset.crs) if dataset.crs else None,
        "transform": [float(value) for value in dataset.transform],
    }


def run(*, map_manifest_path: Path | None = None) -> Path:
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError("rasterio is required to validate map outputs") from error

    map_manifest_path = _resolve(map_manifest_path) if map_manifest_path else _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_spatial_maps_*.json")),
        "spatial map manifest",
    )
    map_manifest = json.loads(map_manifest_path.read_text(encoding="utf-8"))
    expected_grid = map_manifest["grid"]
    intensity_nodata = float(map_manifest["map_contract"]["intensity_nodata"])
    records: list[dict[str, Any]] = []
    checksum_mismatches: list[str] = []
    profile_mismatches: list[str] = []

    for item in map_manifest["artifacts"]:
        observation_date = str(item["observation_date"])
        intensity_path = _resolve(Path(str(item["intensity_path"])))
        mask_path = _resolve(Path(str(item["valid_mask_path"])))
        for path in (intensity_path, mask_path):
            if not path.exists():
                raise FileNotFoundError(f"missing exported map artifact: {path}")
        intensity_sha256_matches = _sha256(intensity_path) == item["intensity_sha256"]
        mask_sha256_matches = _sha256(mask_path) == item["valid_mask_sha256"]
        if not intensity_sha256_matches or not mask_sha256_matches:
            checksum_mismatches.append(observation_date)

        with rasterio.open(intensity_path) as intensity_dataset, rasterio.open(mask_path) as mask_dataset:
            intensity_profile = _profile(intensity_dataset)
            mask_profile = _profile(mask_dataset)
            expected_profile = {
                "width": int(expected_grid["width"]),
                "height": int(expected_grid["height"]),
                "count": 1,
                "dtype": "float32",
                "crs": expected_grid["crs"],
                "transform": [float(value) for value in expected_grid["transform"]],
            }
            mask_expected_profile = {**expected_profile, "dtype": "uint8"}
            profiles_match = intensity_profile == expected_profile and mask_profile == mask_expected_profile
            if not profiles_match:
                profile_mismatches.append(observation_date)
            intensity = intensity_dataset.read(1)
            mask = mask_dataset.read(1)
            array_validation = validate_exported_map_arrays(
                intensity,
                mask,
                intensity_nodata=intensity_nodata,
            )
            nodata_matches = (
                intensity_dataset.nodata == intensity_nodata and mask_dataset.nodata == 0
            )
        records.append(
            {
                "observation_date": observation_date,
                "intensity_sha256_matches": intensity_sha256_matches,
                "valid_mask_sha256_matches": mask_sha256_matches,
                "profiles_match": profiles_match,
                "nodata_values_match": nodata_matches,
                **array_validation,
            }
        )

    validation_passed = not checksum_mismatches and not profile_mismatches and all(
        record["nodata_values_match"] and record["validation_passed"] for record in records
    )
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    validation_manifest = {
        "source_id": "algal_bloom_spatial_map_validation",
        "retrieved_at": retrieved_at.isoformat(),
        "map_manifest": str(map_manifest_path.relative_to(ROOT)),
        "map_manifest_sha256": _sha256(map_manifest_path),
        "records": len(records),
        "validation": {
            "status": "map_validation_complete" if validation_passed else "validation_failed",
            "checksums_match": not checksum_mismatches,
            "profiles_match": not profile_mismatches,
            "nodata_and_mask_contract_validated": all(
                record["nodata_values_match"] and record["validation_passed"]
                for record in records
            ),
            "checksum_mismatches": checksum_mismatches,
            "profile_mismatches": profile_mismatches,
        },
        "map_contract": map_manifest["map_contract"],
        "artifacts": records,
    }
    output = ROOT / "data/manifests" / f"algal_bloom_spatial_map_validation_{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(json.dumps(validation_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "records": len(records)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-manifest", type=Path)
    args = parser.parse_args()
    run(map_manifest_path=args.map_manifest)


if __name__ == "__main__":
    main()
