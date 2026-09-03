#!/usr/bin/env python3
"""Index the completed research artifacts for a reproducible project handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_PATTERNS = {
    "regional_reference": "algal_bloom_regional_reference_????????T??????Z.json",
    "regional_reference_validation": "algal_bloom_regional_reference_validation_*.json",
    "glerl_annual_summary_scope": "algal_bloom_glerl_annual_summary_scope_validation_*.json",
    "training_ready": "algal_bloom_training_ready_????????T??????Z.json",
    "baseline_results": "algal_bloom_baseline_results_*.json",
    "gradient_baseline": "algal_bloom_gradient_baseline_*.json",
    "training_run": "algal_bloom_training_run_*.json",
    "model_selection": "algal_bloom_model_selection_*.json",
    "event_results": "algal_bloom_event_results_*.json",
    "feature_ablation": "algal_bloom_feature_ablation_*.json",
    "error_diagnostics": "algal_bloom_selected_error_diagnostics_*.json",
    "lag_analysis": "algal_bloom_lag_analysis_*.json",
    "horizon_summary": "algal_bloom_horizon_summary_*.json",
    "spatial_persistence": "algal_bloom_spatial_persistence_*.json",
    "spatial_summary": "algal_bloom_spatial_summary_*.json",
    "spatial_map_validation": "algal_bloom_spatial_map_validation_*.json",
}


def _latest(pattern: str, label: str) -> Path:
    paths = sorted((ROOT / "data/manifests").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {label} manifest found")
    return paths[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _source_revision() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": revision, "tracked_worktree": "clean" if not status else "dirty"}


def run(*, output_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    selected = {
        key: _latest(pattern, key) for key, pattern in ARTIFACT_PATTERNS.items()
    }
    reference_validation = json.loads(selected["regional_reference_validation"].read_text())
    training_ready = reference_validation["training_ready"]
    release = {
        "source_id": "algal_bloom_project_release",
        "retrieved_at": retrieved_at.isoformat(),
        "project": "algal-bloom-forecast",
        "release_status": "research_prototype_complete",
        "source_revision": _source_revision(),
        "scope": {
            "region": "western Lake Erie",
            "regional_horizons_days": [1, 3, 7, 14],
            "primary_target": "historical cross-sensor fused 10-day ci_sum composite series",
            "spatial_extension": "validated descriptive raster extension; not a spatial forecast model",
        },
        "artifacts": {key: _describe(path) for key, path in selected.items()},
        "validated_boundary": {
            "reference_validation_status": reference_validation["status"],
            "pinned_manifest_count": reference_validation["pinned_manifest_count"],
            "training_ready": training_ready,
            "candidate_model_runs_present": True,
            "raw_data_committed": False,
        },
        "conclusions_boundary": {
            "model_selection": "candidate comparisons and validation-only selection are recorded; no production model is promoted",
            "event_probabilities": "hard-threshold diagnostics only; calibrated probabilities are not claimed",
            "causal_interpretation": "not supported by this observational workflow",
        },
        "follow_up_gates": [
            "resolve the historical ci_sum aggregate unit and 10-day calibration mapping",
            "define exact daily t-plus-horizon labels from a calibrated target product",
            "add calibrated event probabilities after more independent validation cases",
            "add a temporal neural model only if expanded data coverage justifies it",
            "add a spatial model only after a held-out-season archive is available",
        ],
    }
    output = output_path or ROOT / "data/manifests" / f"algal_bloom_project_release_{run_id}.json"
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "status": release["release_status"]}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(output_path=args.output)


if __name__ == "__main__":
    main()
