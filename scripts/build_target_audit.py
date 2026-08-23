#!/usr/bin/env python3
"""Create an immutable audit of the regional target contract."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.data.normalization import build_normalization_contract
from algal_bloom_forecast.data.target_audit import audit_target_records

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_target(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {"observation_date": row.get("observation_date", "")}
            for field in ("ci_sum", "bloom_area_sqkm"):
                value = (row.get(field) or "").strip()
                parsed[field] = None if value == "" else float(value)
            rows.append(parsed)
    return rows


def run(*, manifest_path: Path | None = None, target_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    manifest_path = _resolve(manifest_path) if manifest_path else _latest(
        list((ROOT / "data/manifests").glob("noaa_western_lake_erie_historical_ci_timeseries_*.json")),
        "historical target manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    derived = manifest["derived_table"]
    target_path = _resolve(target_path) if target_path else _resolve(Path(derived["local_path"]))
    report = audit_target_records(_read_target(target_path))
    target_contract = build_normalization_contract()["target"]

    audit = {
        "source_id": "algal_bloom_target_audit",
        "retrieved_at": retrieved_at.isoformat(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "target_path": str(target_path.relative_to(ROOT)),
        "target_contract": {
            "region": "western_lake_erie",
            **target_contract,
            "target_field": target_contract["field"],
            "target_semantics": manifest["target_semantics"],
            "missing_target_policy": target_contract["missing_policy"],
            "interpolation": "disabled",
            "spatial_status": "regional fused series; coordinates and raster footprint are not present",
            "quality_flag_status": "missing values are explicit; per-record source quality flags are not present",
            "revision_policy": "preserve source and derived artifacts by immutable manifest ID; never overwrite",
        },
        "audit": report,
        "readiness": {
            "temporal_dates_validated": True,
            "missingness_profiled": True,
            "units_normalized": False,
            "target_unit_calibration_required": True,
            "spatial_coverage_profiled": False,
            "quality_flags_normalized": False,
            "daily_horizon_labels_ready": False,
            "status": "target_audited_normalization_pending",
        },
    }
    output = ROOT / "data/manifests" / f"algal_bloom_target_audit_{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "records": report["records"]}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    run(manifest_path=args.manifest, target_path=args.target)


if __name__ == "__main__":
    main()
