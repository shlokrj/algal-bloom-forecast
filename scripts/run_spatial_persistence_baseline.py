#!/usr/bin/env python3
"""Evaluate a no-fit spatial persistence baseline on the masked raster target."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HORIZONS = (1, 3, 7, 14)


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _pixel_metrics(target: np.ndarray, baseline: np.ndarray, overlap: np.ndarray) -> dict[str, float | int | None]:
    target_values = target[overlap].astype(np.float64)
    baseline_values = baseline[overlap].astype(np.float64)
    if target_values.size == 0:
        return {"overlap_pixel_count": 0, "mae": None, "rmse": None}
    errors = baseline_values - target_values
    return {
        "overlap_pixel_count": int(target_values.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(np.mean(errors**2))),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def run(*, spatial_manifest_path: Path | None = None, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> Path:
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain only positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must contain unique values")

    spatial_manifest_path = _resolve(spatial_manifest_path) if spatial_manifest_path else _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_spatial_target_*.json")),
        "spatial target manifest",
    )
    manifest = json.loads(spatial_manifest_path.read_text(encoding="utf-8"))
    artifacts = sorted(manifest["artifacts"], key=lambda item: item["observation_date"])
    dates = [_parse_date(item["observation_date"]) for item in artifacts]
    arrays = {
        item["observation_date"]: _load_arrays(_resolve(Path(item["output_path"])))
        for item in artifacts
    }

    rows: list[dict[str, Any]] = []
    aggregate_errors: dict[int, list[float]] = defaultdict(list)
    aggregate_squared_errors: dict[int, list[float]] = defaultdict(list)
    scored_counts: dict[int, int] = defaultdict(int)
    for target_item, target_date in zip(artifacts, dates):
        target_arrays = arrays[target_item["observation_date"]]
        target_mask = target_arrays["valid_mask"].astype(bool)
        for horizon in horizons:
            cutoff = target_date - timedelta(days=horizon)
            eligible_dates = [candidate for candidate in dates if candidate <= cutoff]
            baseline_date = max(eligible_dates) if eligible_dates else None
            row: dict[str, Any] = {
                "target_date": target_item["observation_date"],
                "forecast_horizon_days": horizon,
                "baseline_date": baseline_date.isoformat() if baseline_date else None,
                "target_valid_pixel_count": int(target_mask.sum()),
                "status": "no_prior_observation",
                "overlap_pixel_count": 0,
                "mae": None,
                "rmse": None,
            }
            if baseline_date is not None:
                baseline_arrays = arrays[baseline_date.isoformat()]
                baseline_mask = baseline_arrays["valid_mask"].astype(bool)
                overlap = target_mask & baseline_mask
                metrics = _pixel_metrics(
                    target_arrays["intensity"], baseline_arrays["intensity"], overlap
                )
                row.update(metrics)
                row["status"] = "scored" if metrics["overlap_pixel_count"] else "no_valid_overlap"
                if metrics["overlap_pixel_count"]:
                    scored_counts[horizon] += 1
                    error_values = (
                        baseline_arrays["intensity"][overlap].astype(np.float64)
                        - target_arrays["intensity"][overlap].astype(np.float64)
                    )
                    aggregate_errors[horizon].extend(np.abs(error_values).tolist())
                    aggregate_squared_errors[horizon].extend((error_values**2).tolist())
            rows.append(row)

    rows.sort(key=lambda row: (row["target_date"], int(row["forecast_horizon_days"])))
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = ROOT / "results/tables" / f"algal_bloom_spatial_persistence_{run_id}.csv"
    fields = [
        "target_date",
        "forecast_horizon_days",
        "baseline_date",
        "target_valid_pixel_count",
        "status",
        "overlap_pixel_count",
        "mae",
        "rmse",
    ]
    _write_csv(output_path, rows, fields)
    aggregate = []
    for horizon in sorted(horizons):
        absolute_errors = aggregate_errors[horizon]
        squared_errors = aggregate_squared_errors[horizon]
        aggregate.append(
            {
                "forecast_horizon_days": horizon,
                "target_raster_count": len(artifacts),
                "scored_raster_count": scored_counts[horizon],
                "overlap_pixel_count": len(absolute_errors),
                "mae": float(np.mean(absolute_errors)) if absolute_errors else None,
                "rmse": float(math.sqrt(np.mean(squared_errors))) if squared_errors else None,
            }
        )
    result_manifest = {
        "source_id": "algal_bloom_spatial_persistence_baseline",
        "retrieved_at": retrieved_at.isoformat(),
        "spatial_target_manifest": str(spatial_manifest_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "horizons_days": list(horizons),
        "rows": len(rows),
        "evaluation_policy": {
            "baseline": "latest prior raster at or before target date minus horizon",
            "valid_pixel_policy": "score only pixels valid in both target and baseline masks",
            "interpolation": "disabled",
            "fit_started": False,
            "interpretation": "descriptive spatial baseline; no held-out-season claim",
        },
        "aggregate": aggregate,
    }
    manifest_output = ROOT / "data/manifests" / f"algal_bloom_spatial_persistence_{run_id}.json"
    if manifest_output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_output}")
    manifest_output.write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_output), "rows": len(rows)}, indent=2))
    return manifest_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-manifest", type=Path)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    args = parser.parse_args()
    run(spatial_manifest_path=args.spatial_manifest, horizons=tuple(args.horizons))


if __name__ == "__main__":
    main()
