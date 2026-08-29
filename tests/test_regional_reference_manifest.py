from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_regional_reference_manifest import (
    REFERENCE_PATTERNS,
    _describe,
    select_spatial_reference_manifests,
)
from scripts.validate_regional_reference import _validate_pinned_manifests


def test_describe_records_manifest_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"source_id": "example", "retrieved_at": "2026-08-24T00:00:00Z"}),
        encoding="utf-8",
    )

    description = _describe(manifest)

    assert description["source_id"] == "example"
    assert len(description["sha256"]) == 64


def test_spatial_reference_selection_covers_validated_extension() -> None:
    selected = select_spatial_reference_manifests()

    assert set(selected) == {
        "spatial_target",
        "spatial_persistence",
        "spatial_quality",
        "spatial_summary",
        "spatial_maps",
        "spatial_map_validation",
    }


def test_regional_reference_selection_includes_curated_ci_reference() -> None:
    assert REFERENCE_PATTERNS["curated_annual_ci_reference"] == (
        "algal_bloom_noaa_curated_wle_annual_ci_*.json"
    )


def test_pinned_manifest_validation_records_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {"source_id": "example"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    reference = {
        "manifests": {
            "regional": {
                "example": {
                    "manifest_path": str(manifest),
                    "source_id": "example",
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                }
            }
        }
    }

    checks = _validate_pinned_manifests(reference)

    assert checks == [
        {
            "scope": "regional",
            "key": "example",
            "manifest_path": str(manifest),
            "sha256_matches": True,
            "source_id_matches": True,
        }
    ]
