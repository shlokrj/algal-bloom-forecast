from __future__ import annotations

import json
from pathlib import Path

from scripts.build_regional_reference_manifest import _describe


def test_describe_records_manifest_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"source_id": "example", "retrieved_at": "2026-08-24T00:00:00Z"}),
        encoding="utf-8",
    )

    description = _describe(manifest)

    assert description["source_id"] == "example"
    assert len(description["sha256"]) == 64
