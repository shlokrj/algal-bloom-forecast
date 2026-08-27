#!/usr/bin/env python3
"""Summarize validated spatial intensity and coverage by observation date."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from algal_bloom_forecast.data.spatial import (
    summarize_spatial_intensity,
    validate_spatial_arrays,
)

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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    spatial_manifest_path: Path | None = None,
    quality_manifest_path: Path | None = None,
) -> Path:
    spatial_manifest_path = _resolve(spatial_manifest_path) if spatial_manifest_path else _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_spatial_target_*.json")),
        "spatial target manifest",
    )
    quality_manifest_path = _resolve(quality_manifest_path) if quality_manifest_path else _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_spatial_quality_*.json")),
        "spatial quality manifest",
    )
    spatial_manifest = json.loads(spatial_manifest_path.read_text(encoding="utf-8"))
    quality_manifest = json.loads(quality_manifest_path.read_text(encoding="utf-8"))
    if quality_manifest["validation"]["status"] != "mask_validation_complete":
        raise ValueError("spatial summary requires a passing spatial quality manifest")
    if quality_manifest["spatial_target_manifest"] != str(spatial_manifest_path.relative_to(ROOT)):
        raise ValueError("quality manifest does not describe the requested spatial target")

    rows: list[dict[str, Any]] = []
    valid_dn_range = tuple(spatial_manifest["mask_contract"]["valid_dn_range"])
    for item in sorted(spatial_manifest["artifacts"], key=lambda value: value["observation_date"]):
        arrays = _load_arrays(_resolve(Path(str(item["output_path"]))))
        validation = validate_spatial_arrays(
            arrays["intensity"],
            arrays["valid_mask"],
            arrays["raw_dn"],
            valid_dn_range=valid_dn_range,
        )
        summary = summarize_spatial_intensity(arrays["intensity"], arrays["valid_mask"])
        rows.append(
            {
                "observation_date": item["observation_date"],
                "total_pixel_count": validation["total_pixel_count"],
                "valid_pixel_count": validation["valid_pixel_count"],
                "valid_pixel_fraction": validation["valid_pixel_fraction"],
                "intensity_min": summary["intensity_min"],
                "intensity_mean": summary["intensity_mean"],
                "intensity_median": summary["intensity_median"],
                "intensity_p95": summary["intensity_p95"],
                "intensity_max": summary["intensity_max"],
            }
        )

    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = ROOT / "results/tables" / f"algal_bloom_spatial_summary_{run_id}.csv"
    fields = [
        "observation_date",
        "total_pixel_count",
        "valid_pixel_count",
        "valid_pixel_fraction",
        "intensity_min",
        "intensity_mean",
        "intensity_median",
        "intensity_p95",
        "intensity_max",
    ]
    _write_csv(output_path, rows, fields)
    manifest = {
        "source_id": "algal_bloom_spatial_summary",
        "retrieved_at": retrieved_at.isoformat(),
        "spatial_target_manifest": str(spatial_manifest_path.relative_to(ROOT)),
        "spatial_target_manifest_sha256": _sha256(spatial_manifest_path),
        "quality_manifest": str(quality_manifest_path.relative_to(ROOT)),
        "quality_manifest_sha256": _sha256(quality_manifest_path),
        "output_path": str(output_path.relative_to(ROOT)),
        "rows": len(rows),
        "fields": fields,
        "summary_contract": {
            "statistics": "min, mean, median, p95, and max over valid decoded pixels",
            "coverage": "valid pixel count and fraction retained per date",
            "invalid_pixel_policy": "excluded from intensity statistics; no interpolation",
            "warning_threshold": "not applied; operational threshold is not defined",
        },
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_spatial_summary_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "rows": len(rows)}, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-manifest", type=Path)
    parser.add_argument("--quality-manifest", type=Path)
    args = parser.parse_args()
    run(
        spatial_manifest_path=args.spatial_manifest,
        quality_manifest_path=args.quality_manifest,
    )


if __name__ == "__main__":
    main()
