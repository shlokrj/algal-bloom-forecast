#!/usr/bin/env python3
"""Profile current, lagged, and rolling feature availability by source."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _base_training_ready_manifests() -> list[Path]:
    return [
        path
        for path in (ROOT / "data/manifests").glob("algal_bloom_training_ready_*.json")
        if "coverage_pruned" not in path.name
    ]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def _family(field: str) -> str:
    for family in ("ndbc", "glerl", "usgs"):
        if family in field:
            return family
    if field.startswith("seasonal_"):
        return "seasonal"
    return "derived_other"


def _window(field: str) -> str:
    for window in ("lag_1d", "lag_3d", "lag_7d", "rolling_3d", "rolling_7d", "rolling_14d"):
        if field.startswith(window):
            return window
    return "current"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(*, frame_path: Path | None = None, manifest_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_path or _latest(
        _base_training_ready_manifests(),
        "training-ready manifest",
    )
    manifest_path = manifest_path.resolve()
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_path = (frame_path or ROOT / source_manifest["output_path"]).resolve()
    feature_names = tuple(source_manifest["schema"]["feature_names"])
    with frame_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, int, str, str], dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "missing": 0}
    )
    for row in rows:
        split = row["split"]
        horizon = int(float(row["forecast_horizon_days"]))
        for field in feature_names:
            bucket = grouped[(split, horizon, _family(field), _window(field))]
            bucket["rows"] += 1
            bucket["missing"] += row.get(field, "") == ""
    output_rows = [
        {
            "split": split,
            "forecast_horizon_days": horizon,
            "source_family": family,
            "window": window,
            "feature_count": sum(
                _family(field) == family and _window(field) == window for field in feature_names
            ),
            **values,
            "missing_rate": values["missing"] / values["rows"] if values["rows"] else None,
        }
        for (split, horizon, family, window), values in sorted(grouped.items())
    ]
    output_path = ROOT / "results/tables" / f"algal_bloom_lag_analysis_{run_id}.csv"
    _write_csv(
        output_path,
        output_rows,
        [
            "split",
            "forecast_horizon_days",
            "source_family",
            "window",
            "feature_count",
            "rows",
            "missing",
            "missing_rate",
        ],
    )
    manifest = {
        "source_id": "algal_bloom_lag_analysis",
        "retrieved_at": retrieved_at.isoformat(),
        "training_ready_manifest": str(manifest_path.relative_to(ROOT)),
        "frame_path": str(frame_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "rows": len(output_rows),
        "policy": "availability profiling only; no model selection or imputation",
    }
    manifest_output = ROOT / "data/manifests" / f"algal_bloom_lag_analysis_{run_id}.json"
    if manifest_output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_output}")
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(output_rows)} lag-availability rows")
    print(f"wrote manifest to {manifest_output}")
    return manifest_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    run(frame_path=args.frame, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
