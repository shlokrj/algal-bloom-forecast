#!/usr/bin/env python3
"""Validate the immutable regional-plus-spatial reference bundle."""

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


def _validate_pinned_manifests(reference: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for scope, manifests in reference["manifests"].items():
        for key, identity in manifests.items():
            path = _resolve(Path(str(identity["manifest_path"])))
            if not path.exists():
                raise FileNotFoundError(f"missing pinned {scope}/{key} manifest: {path}")
            actual_sha256 = _sha256(path)
            if actual_sha256 != identity["sha256"]:
                raise ValueError(
                    f"checksum mismatch for pinned {scope}/{key}: "
                    f"{actual_sha256} != {identity['sha256']}"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("source_id") != identity["source_id"]:
                raise ValueError(f"source ID mismatch for pinned {scope}/{key}")
            checks.append(
                {
                    "scope": scope,
                    "key": key,
                    "manifest_path": identity["manifest_path"],
                    "sha256_matches": True,
                    "source_id_matches": True,
                }
            )
    return checks


def run(*, reference_path: Path | None = None) -> Path:
    reference_path = _resolve(reference_path) if reference_path else _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_regional_reference_*.json")),
        "regional reference manifest",
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    pinned_checks = _validate_pinned_manifests(reference)
    training_identity = reference["manifests"]["regional"]["training_ready"]
    training_manifest_path = _resolve(Path(str(training_identity["manifest_path"])))
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    training_frame_path = _resolve(Path(str(training_manifest["output_path"])))
    training_validation = validate(training_manifest_path, training_frame_path)

    spatial_quality = json.loads(
        _resolve(
            Path(str(reference["manifests"]["spatial"]["spatial_quality"]["manifest_path"]))
        ).read_text(encoding="utf-8")
    )
    spatial_map_validation = json.loads(
        _resolve(
            Path(
                str(
                    reference["manifests"]["spatial"]["spatial_map_validation"]["manifest_path"]
                )
            )
        ).read_text(encoding="utf-8")
    )
    gate_checks = {
        "regional_reference_status": reference["gates"]["regional_reference_status"] == "validated",
        "training_ready_status": training_validation["status"] == "prepared_not_fitted",
        "model_fit_started": training_validation["model_fit_started"],
        "spatial_target_status": reference["gates"]["spatial_target_status"]
        == "validated_descriptive_extension",
        "spatial_quality_status": spatial_quality["validation"]["status"]
        == "mask_validation_complete",
        "spatial_map_validation_status": spatial_map_validation["validation"]["status"]
        == "map_validation_complete",
    }
    if gate_checks["model_fit_started"]:
        raise ValueError("reference bundle unexpectedly says model fitting has started")
    positive_gate_checks = {
        key: value for key, value in gate_checks.items() if key != "model_fit_started"
    }
    if not all(positive_gate_checks.values()):
        raise ValueError(f"reference gate failure: {gate_checks}")

    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    validation_manifest = {
        "source_id": "algal_bloom_regional_reference_validation",
        "retrieved_at": retrieved_at.isoformat(),
        "reference_manifest": str(reference_path.relative_to(ROOT)),
        "reference_manifest_sha256": _sha256(reference_path),
        "pinned_manifest_count": len(pinned_checks),
        "pinned_manifests": pinned_checks,
        "training_ready": training_validation,
        "gates": gate_checks,
        "status": "reference_validation_complete",
    }
    output = ROOT / "data/manifests" / f"algal_bloom_regional_reference_validation_{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(
        json.dumps(validation_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output), "pinned_manifest_count": len(pinned_checks)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    run(reference_path=args.reference)


if __name__ == "__main__":
    main()
