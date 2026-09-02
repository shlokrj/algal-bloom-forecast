from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_project_release_manifest import ARTIFACT_PATTERNS, _describe


def test_release_artifact_index_covers_model_and_spatial_outputs() -> None:
    assert "regional_reference_validation" in ARTIFACT_PATTERNS
    assert "training_run" in ARTIFACT_PATTERNS
    assert "model_selection" in ARTIFACT_PATTERNS
    assert "spatial_map_validation" in ARTIFACT_PATTERNS


def test_release_describe_records_immutable_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {"source_id": "example", "retrieved_at": "2026-09-01T00:00:00Z"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    description = _describe(manifest)

    assert description["source_id"] == "example"
    assert description["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
