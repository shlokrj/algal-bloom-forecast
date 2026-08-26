#!/usr/bin/env python3
"""Validate materialized spatial masks and summarize their coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from algal_bloom_forecast.data.satellite import profile_ci_cyano_pixels
from algal_bloom_forecast.data.spatial import validate_spatial_arrays

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


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _calendar_gaps(dates: list[date]) -> list[dict[str, Any]]:
    gaps = []
    for previous, current in pairwise(dates):
        gap_days = (current - previous).days
        if gap_days > 1:
            gaps.append(
                {
                    "after": previous.isoformat(),
                    "before": current.isoformat(),
                    "missing_day_count": gap_days - 1,
                }
            )
    return gaps


def run(*, spatial_manifest_path: Path | None = None) -> Path:
    spatial_manifest_path = _resolve(spatial_manifest_path) if spatial_manifest_path else _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_spatial_target_*.json")),
        "spatial target manifest",
    )
    spatial_manifest = json.loads(spatial_manifest_path.read_text(encoding="utf-8"))
    artifacts = list(spatial_manifest["artifacts"])
    if not artifacts:
        raise ValueError("spatial target manifest contains no artifacts")

    dates = [_parse_date(str(item["observation_date"])) for item in artifacts]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError("spatial target artifacts must have unique sorted observation dates")

    grid = spatial_manifest["grid"]
    expected_shape = (int(grid["height"]), int(grid["width"]))
    valid_dn_range = tuple(spatial_manifest["mask_contract"]["valid_dn_range"])
    aggregate_flags: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    valid_fractions: list[float] = []
    all_invalid_dates: list[str] = []
    output_checksum_mismatches: list[str] = []
    profile_mismatches: list[str] = []

    for item in artifacts:
        observation_date = str(item["observation_date"])
        output_path = _resolve(Path(str(item["output_path"])))
        if not output_path.exists():
            raise FileNotFoundError(f"missing spatial array bundle: {output_path}")
        actual_sha256 = _sha256(output_path)
        if actual_sha256 != item["output_sha256"]:
            output_checksum_mismatches.append(observation_date)

        arrays = _load_arrays(output_path)
        required_fields = {"intensity", "valid_mask", "raw_dn"}
        if set(arrays) != required_fields:
            raise ValueError(
                f"{output_path} must contain exactly {sorted(required_fields)}, "
                f"got {sorted(arrays)}"
            )
        if tuple(arrays["intensity"].shape) != expected_shape:
            raise ValueError(
                f"{observation_date} array shape {arrays['intensity'].shape} "
                f"does not match manifest grid {expected_shape}"
            )

        validation = validate_spatial_arrays(
            arrays["intensity"],
            arrays["valid_mask"],
            arrays["raw_dn"],
            valid_dn_range=valid_dn_range,
        )
        actual_profile = profile_ci_cyano_pixels(arrays["raw_dn"])
        if actual_profile != item["pixel_profile"]:
            profile_mismatches.append(observation_date)
        aggregate_flags.update(actual_profile["flag_pixel_counts"])
        valid_fraction = float(validation["valid_pixel_fraction"])
        valid_fractions.append(valid_fraction)
        if validation["valid_pixel_count"] == 0:
            all_invalid_dates.append(observation_date)
        records.append(
            {
                "observation_date": observation_date,
                "output_path": str(item["output_path"]),
                "output_sha256_matches": actual_sha256 == item["output_sha256"],
                "pixel_profile_matches": actual_profile == item["pixel_profile"],
                **validation,
                "flag_pixel_counts": actual_profile["flag_pixel_counts"],
            }
        )

    intervals = [(current - previous).days for previous, current in pairwise(dates)]
    persistence_paths = list(
        (ROOT / "data/manifests").glob("algal_bloom_spatial_persistence_*.json")
    )
    persistence_manifest_path = _latest(persistence_paths, "spatial persistence manifest") if persistence_paths else None
    validation_passed = not output_checksum_mismatches and not profile_mismatches
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "source_id": "algal_bloom_spatial_quality",
        "retrieved_at": retrieved_at.isoformat(),
        "spatial_target_manifest": str(spatial_manifest_path.relative_to(ROOT)),
        "spatial_target_manifest_sha256": _sha256(spatial_manifest_path),
        "spatial_persistence_manifest": (
            str(persistence_manifest_path.relative_to(ROOT))
            if persistence_manifest_path
            else None
        ),
        "records": len(records),
        "observation_start": dates[0].isoformat(),
        "observation_end": dates[-1].isoformat(),
        "grid": grid,
        "validation": {
            "status": "mask_validation_complete" if validation_passed else "validation_failed",
            "grid_shape_consistent": all(
                tuple(record["shape"]) == expected_shape for record in records
            ),
            "valid_mask_matches_raw_dn": all(
                record["mask_matches_raw_dn"] for record in records
            ),
            "intensity_nan_policy_validated": all(
                record["valid_intensity_nonfinite_count"] == 0
                and record["invalid_intensity_non_nan_count"] == 0
                for record in records
            ),
            "output_checksums_match": not output_checksum_mismatches,
            "pixel_profiles_match": not profile_mismatches,
            "output_checksum_mismatches": output_checksum_mismatches,
            "pixel_profile_mismatches": profile_mismatches,
        },
        "coverage": {
            "valid_fraction_min": min(valid_fractions),
            "valid_fraction_median": median(valid_fractions),
            "valid_fraction_max": max(valid_fractions),
            "all_invalid_dates": all_invalid_dates,
            "interval_days_min": min(intervals) if intervals else None,
            "interval_days_median": median(intervals) if intervals else None,
            "interval_days_max": max(intervals) if intervals else None,
            "missing_calendar_gaps": _calendar_gaps(dates),
            "aggregate_flag_pixel_counts": dict(sorted(aggregate_flags.items())),
        },
        "spatial_model_gate": {
            "status": "deferred",
            "reason": (
                "current archive is a single 55-observation season with variable masked "
                "coverage and no held-out season; retain spatial persistence as descriptive"
            ),
        },
        "map_output_gate": {
            "status": "rules_validated",
            "policy": "map outputs may expose only decoded intensity where valid_mask is one",
            "interpolation": "disabled",
        },
        "artifacts": records,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_spatial_quality_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "records": len(records)}, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-manifest", type=Path)
    args = parser.parse_args()
    run(spatial_manifest_path=args.spatial_manifest)


if __name__ == "__main__":
    main()
