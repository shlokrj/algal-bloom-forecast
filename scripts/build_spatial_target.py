#!/usr/bin/env python3
"""Materialize decoded spatial targets and explicit observation masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.data.satellite import (
    PRODUCT_FLAG_LABELS,
    decode_ci_cyano,
    parse_satellite_filename,
    profile_ci_cyano_pixels,
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


def _grid_profile(dataset: Any) -> dict[str, Any]:
    return {
        "width": dataset.width,
        "height": dataset.height,
        "count": dataset.count,
        "dtype": dataset.dtypes[0],
        "crs": str(dataset.crs) if dataset.crs else None,
        "transform": [float(value) for value in dataset.transform],
        "bounds": [float(value) for value in dataset.bounds],
        "nodata": dataset.nodata,
    }


def _same_grid(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def run(*, archive_manifest_path: Path | None = None) -> Path:
    try:
        import numpy as np
        import rasterio
    except ImportError as error:
        raise RuntimeError(
            "numpy and rasterio are required to build spatial targets; install the data extras"
        ) from error

    archive_manifest_path = _resolve(archive_manifest_path) if archive_manifest_path else _latest(
        list((ROOT / "data/manifests").glob("noaa_western_lake_erie_satellite_archive_*.json")),
        "satellite archive manifest",
    )
    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    artifacts = archive_manifest["artifacts"]
    if not artifacts:
        raise ValueError("satellite archive manifest contains no artifacts")

    verified_artifacts: list[tuple[dict[str, Any], Path]] = []
    for artifact in artifacts:
        local_path = _resolve(Path(str(artifact["local_path"])))
        if not local_path.exists():
            raise FileNotFoundError(f"missing archived raster: {local_path}")
        actual_sha256 = _sha256(local_path)
        if actual_sha256 != artifact["sha256"]:
            raise ValueError(
                f"satellite checksum mismatch for {local_path}: {actual_sha256} != {artifact['sha256']}"
            )
        verified_artifacts.append((artifact, local_path))

    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "data/processed" / f"algal_bloom_spatial_target_{run_id}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite spatial output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    reference_grid: dict[str, Any] | None = None
    try:
        for artifact, local_path in verified_artifacts:
            filename_metadata = parse_satellite_filename(local_path.name)
            with rasterio.open(local_path) as dataset:
                if dataset.count != 1:
                    raise ValueError(f"expected one band in {local_path}, found {dataset.count}")
                grid = _grid_profile(dataset)
                if reference_grid is None:
                    reference_grid = grid
                elif not _same_grid(reference_grid, grid):
                    raise ValueError(f"raster grid differs from reference grid: {local_path}")
                raw_dn = dataset.read(1)
                raster_tags = dataset.tags()

            decoded, valid = decode_ci_cyano(raw_dn)
            intensity = np.asarray(decoded, dtype=np.float32)
            valid_mask = np.asarray(valid, dtype=np.uint8)
            raw_dn = np.asarray(raw_dn, dtype=np.uint8)
            profile = profile_ci_cyano_pixels(raw_dn)
            output_path = output_dir / f"{filename_metadata.observation_date}.npz"
            np.savez_compressed(
                output_path,
                intensity=intensity,
                valid_mask=valid_mask,
                raw_dn=raw_dn,
            )
            records.append(
                {
                    "observation_date": filename_metadata.observation_date,
                    "acquisition_window": filename_metadata.acquisition_window,
                    "source_filename": local_path.name,
                    "source_path": str(local_path.relative_to(ROOT)),
                    "source_sha256": artifact["sha256"],
                    "output_path": str(output_path.relative_to(ROOT)),
                    "output_sha256": _sha256(output_path),
                    "output_size_bytes": output_path.stat().st_size,
                    "pixel_profile": profile,
                    "product_tags": {
                        key: value
                        for key, value in raster_tags.items()
                        if key.startswith("SAPS_product_")
                    },
                }
            )
    except Exception:
        shutil.rmtree(output_dir)
        raise

    if reference_grid is None:
        raise ValueError("no raster grid profile was created")
    records.sort(key=lambda record: record["observation_date"])
    manifest = {
        "source_id": "algal_bloom_spatial_target",
        "retrieved_at": retrieved_at.isoformat(),
        "archive_manifest": str(archive_manifest_path.relative_to(ROOT)),
        "archive_manifest_sha256": _sha256(archive_manifest_path),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "records": len(records),
        "observation_start": records[0]["observation_date"],
        "observation_end": records[-1]["observation_date"],
        "grid": reference_grid,
        "mask_contract": {
            "raw_dn_field": "raw_dn",
            "decoded_intensity_field": "intensity",
            "valid_mask_field": "valid_mask",
            "valid_dn_range": [1, 249],
            "invalid_product_flags": {
                str(code): label for code, label in sorted(PRODUCT_FLAG_LABELS.items())
            },
            "missing_pixel_policy": "retain invalid pixels in raw_dn and set intensity to NaN; valid_mask is zero",
            "spatial_interpolation": "disabled",
        },
        "spatial_target_status": "gridded_target_materialized; spatial_baseline_not_fitted",
        "artifacts": records,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_spatial_target_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "records": len(records)}, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-manifest", type=Path)
    args = parser.parse_args()
    run(archive_manifest_path=args.archive_manifest)


if __name__ == "__main__":
    main()
