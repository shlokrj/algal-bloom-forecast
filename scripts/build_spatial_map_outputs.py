#!/usr/bin/env python3
"""Export validated spatial targets as georeferenced, no-interpolation maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from algal_bloom_forecast.data.spatial import validate_spatial_arrays

ROOT = Path(__file__).resolve().parents[1]
MAP_NODATA = -9999.0


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


def run(
    *,
    spatial_manifest_path: Path | None = None,
    quality_manifest_path: Path | None = None,
) -> Path:
    try:
        import rasterio
        from affine import Affine
    except ImportError as error:
        raise RuntimeError(
            "rasterio and affine are required to build georeferenced maps; install the data extras"
        ) from error

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
        raise ValueError("map export requires a passing spatial quality manifest")
    if quality_manifest["spatial_target_manifest"] != str(spatial_manifest_path.relative_to(ROOT)):
        raise ValueError("quality manifest does not describe the requested spatial target")

    artifacts = sorted(spatial_manifest["artifacts"], key=lambda item: item["observation_date"])
    grid = spatial_manifest["grid"]
    expected_shape = (int(grid["height"]), int(grid["width"]))
    output_timestamp = datetime.now(UTC)
    run_id = output_timestamp.strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "results/figures" / f"algal_bloom_spatial_maps_{run_id}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite map output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    try:
        for item in artifacts:
            observation_date = str(item["observation_date"])
            arrays = _load_arrays(_resolve(Path(str(item["output_path"]))))
            validation = validate_spatial_arrays(
                arrays["intensity"],
                arrays["valid_mask"],
                arrays["raw_dn"],
                valid_dn_range=tuple(spatial_manifest["mask_contract"]["valid_dn_range"]),
            )
            if tuple(validation["shape"]) != expected_shape:
                raise ValueError(f"{observation_date} arrays do not match the manifest grid")

            intensity = arrays["intensity"].astype(np.float32, copy=True)
            valid_mask = arrays["valid_mask"].astype(np.uint8, copy=False)
            intensity[valid_mask == 0] = MAP_NODATA
            intensity_path = output_dir / f"{observation_date}.intensity.tif"
            mask_path = output_dir / f"{observation_date}.valid-mask.tif"
            profile = {
                "driver": "GTiff",
                "width": int(grid["width"]),
                "height": int(grid["height"]),
                "count": 1,
                "crs": grid["crs"],
                "transform": Affine(*[float(value) for value in grid["transform"]]),
                "compress": "deflate",
                "predictor": 3,
                "tiled": True,
            }
            intensity_profile = {
                **profile,
                "dtype": "float32",
                "nodata": MAP_NODATA,
            }
            with rasterio.open(intensity_path, "w", **intensity_profile) as dataset:
                dataset.write(intensity, 1)
                dataset.update_tags(
                    product="NOAA CI-CIcyano",
                    observation_date=observation_date,
                    missing_pixel_policy="invalid pixels are nodata; no interpolation",
                )
            mask_profile = {
                **profile,
                "dtype": "uint8",
                "nodata": 0,
                "predictor": 1,
            }
            with rasterio.open(mask_path, "w", **mask_profile) as dataset:
                dataset.write(valid_mask, 1)
                dataset.update_tags(
                    product="NOAA CI-CIcyano valid-pixel mask",
                    observation_date=observation_date,
                    mask_definition="raw DN in inclusive range 1..249",
                )
            records.append(
                {
                    "observation_date": observation_date,
                    "intensity_path": str(intensity_path.relative_to(ROOT)),
                    "intensity_sha256": _sha256(intensity_path),
                    "intensity_size_bytes": intensity_path.stat().st_size,
                    "valid_mask_path": str(mask_path.relative_to(ROOT)),
                    "valid_mask_sha256": _sha256(mask_path),
                    "valid_mask_size_bytes": mask_path.stat().st_size,
                    "valid_pixel_count": validation["valid_pixel_count"],
                    "valid_pixel_fraction": validation["valid_pixel_fraction"],
                }
            )
    except Exception:
        shutil.rmtree(output_dir)
        raise

    manifest = {
        "source_id": "algal_bloom_spatial_map_outputs",
        "retrieved_at": output_timestamp.isoformat(),
        "spatial_target_manifest": str(spatial_manifest_path.relative_to(ROOT)),
        "spatial_target_manifest_sha256": _sha256(spatial_manifest_path),
        "quality_manifest": str(quality_manifest_path.relative_to(ROOT)),
        "quality_manifest_sha256": _sha256(quality_manifest_path),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "records": len(records),
        "grid": grid,
        "map_contract": {
            "intensity_field": "decoded CI-CIcyano intensity",
            "intensity_units": spatial_manifest["mask_contract"].get(
                "current_raster_units", "dimensionless (dl)"
            ),
            "valid_mask_field": "valid_mask",
            "intensity_nodata": MAP_NODATA,
            "mask_nodata": 0,
            "invalid_pixel_policy": "write intensity as nodata and mask as zero",
            "spatial_interpolation": "disabled",
        },
        "artifacts": records,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_spatial_maps_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "records": len(records)}, indent=2))
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
