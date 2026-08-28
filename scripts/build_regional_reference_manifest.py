#!/usr/bin/env python3
"""Pin the validated regional pipeline for later model or spatial extensions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.validate_training_ready import validate
except ModuleNotFoundError:
    from validate_training_ready import validate

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATTERNS = {
    "target_audit": "algal_bloom_target_audit_*.json",
    "glerl_flag_audit": "algal_bloom_glerl_flag_audit_*.json",
    "station_coordinates": "algal_bloom_glerl_station_coordinates_*.json",
    "feature_table": "algal_bloom_feature_table_*.json",
    "temporal_splits": "algal_bloom_temporal_splits_*.json",
    "training_ready": "algal_bloom_training_ready_*.json",
    "feature_coverage": "algal_bloom_feature_coverage_*.json",
    "rolling_year_evaluation": "algal_bloom_rolling_year_evaluation_*.json",
}

SPATIAL_REFERENCE_PATTERNS = {
    "spatial_target": "algal_bloom_spatial_target_*.json",
    "spatial_persistence": "algal_bloom_spatial_persistence_*.json",
    "spatial_quality": "algal_bloom_spatial_quality_*.json",
    "spatial_summary": "algal_bloom_spatial_summary_*.json",
    "spatial_maps": "algal_bloom_spatial_maps_*.json",
    "spatial_map_validation": "algal_bloom_spatial_map_validation_*.json",
}


def _latest(pattern: str, label: str) -> Path:
    paths = sorted(
        path
        for path in (ROOT / "data/manifests").glob(pattern)
        if "coverage_pruned" not in path.name
    )
    if not paths:
        raise FileNotFoundError(f"no {label} manifest found")
    return paths[-1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_reference_manifests() -> dict[str, Path]:
    """Select the newest non-pruned manifest for each regional pipeline stage."""
    return {
        key: _latest(pattern, key)
        for key, pattern in REFERENCE_PATTERNS.items()
    }


def select_spatial_reference_manifests() -> dict[str, Path]:
    """Select the newest immutable artifact for each spatial extension stage."""
    return {
        key: _latest(pattern, key)
        for key, pattern in SPATIAL_REFERENCE_PATTERNS.items()
    }


def _describe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        manifest_path = str(path.relative_to(ROOT))
    except ValueError:
        manifest_path = str(path)
    return {
        "manifest_path": manifest_path,
        "source_id": payload.get("source_id"),
        "retrieved_at": payload.get("retrieved_at"),
        "sha256": _sha256(path),
    }


def run(*, output_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    selected = select_reference_manifests()
    selected_spatial = select_spatial_reference_manifests()
    manifests = {
        "regional": {key: _describe(path) for key, path in selected.items()},
        "spatial": {key: _describe(path) for key, path in selected_spatial.items()},
    }

    training_manifest = json.loads(selected["training_ready"].read_text(encoding="utf-8"))
    training_frame = _resolve(ROOT / training_manifest["output_path"])
    training_validation = validate(selected["training_ready"], training_frame)
    coverage = json.loads(selected["feature_coverage"].read_text(encoding="utf-8"))
    rolling = json.loads(selected["rolling_year_evaluation"].read_text(encoding="utf-8"))
    target_definition = training_manifest["target_definition"]

    reference = {
        "source_id": "algal_bloom_regional_reference",
        "retrieved_at": retrieved_at.isoformat(),
        "reference_scope": (
            "western Lake Erie regional forecast with a validated descriptive spatial extension"
        ),
        "manifests": manifests,
        "target_definition": target_definition,
        "validation": {
            "training_ready": training_validation,
            "feature_count": training_manifest["schema"]["feature_count"],
            "coverage_summary_rows": len(coverage["summary"]),
            "rolling_metric_rows": rolling["rows"],
            "model_fit_started": training_manifest["validation"]["model_fit_started"],
        },
        "gates": {
            "regional_reference_status": "validated",
            "spatial_target_status": "validated_descriptive_extension",
            "spatial_model_status": (
                "deferred pending a held-out season and stronger spatial coverage"
            ),
            "temporal_neural_status": "deferred pending stronger data coverage and independent validation",
            "calibrated_probability_status": "deferred pending more independent validation cases",
            "open_source_definitions": [
                "historical summed ci_sum unit and calibration",
                (
                    "GLERL quality-flag meanings for codes or annual-summary accessions "
                    "outside the documented 0190201 subset"
                ),
                "exact daily t-plus-horizon target labels",
            ],
        },
    }

    output_path = _resolve(output_path) if output_path else (
        ROOT / "data/manifests" / f"algal_bloom_regional_reference_{run_id}.json"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output_path}")
    output_path.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output_path), "status": "validated"}, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(output_path=args.output)


if __name__ == "__main__":
    main()
