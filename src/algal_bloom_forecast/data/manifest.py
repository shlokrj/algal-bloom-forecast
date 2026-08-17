"""Create immutable manifests for downloaded research artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactManifest:
    """Provenance and content identity for one local source artifact."""

    source_id: str
    source_url: str
    retrieved_at: str
    local_path: str
    sha256: str
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        *,
        source_id: str,
        source_url: str,
        file_path: Path,
        local_path: str,
        metadata: dict[str, Any] | None = None,
        retrieved_at: datetime | None = None,
    ) -> ArtifactManifest:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

        timestamp = retrieved_at or datetime.now(UTC)
        return cls(
            source_id=source_id,
            source_url=source_url,
            retrieved_at=timestamp.astimezone(UTC).isoformat(),
            local_path=local_path,
            sha256=digest.hexdigest(),
            size_bytes=file_path.stat().st_size,
            metadata=metadata or {},
        )

    def write(self, destination: Path) -> None:
        """Write this manifest without modifying an existing manifest."""
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite immutable manifest: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
