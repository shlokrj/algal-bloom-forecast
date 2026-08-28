#!/usr/bin/env python3
"""Validate a GLERL flag audit against its immutable raw source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.data.glerl import (
    QARTOD_FLAG_LABELS,
    QARTOD_FLAG_MAPPING_SCOPE,
    QARTOD_FLAG_REFERENCES,
    profile_glerl_flag_codes,
)

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _validate_profile(
    profile: dict[str, Any],
    *,
    source_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    local_path = _resolve(Path(str(profile["local_path"])))
    if not local_path.exists():
        raise FileNotFoundError(f"missing audited GLERL file: {local_path}")

    source_artifact = source_artifacts.get(str(profile["local_path"]))
    if source_artifact is None:
        raise ValueError(f"audited file is absent from source manifest: {profile['local_path']}")
    actual_sha256 = _sha256(local_path)
    if actual_sha256 != profile["sha256"]:
        raise ValueError(f"audited file checksum mismatch: {local_path}")
    if actual_sha256 != source_artifact["sha256"]:
        raise ValueError(f"source manifest checksum mismatch: {local_path}")

    recomputed = profile_glerl_flag_codes(local_path)
    compared_fields = (
        "source_filename",
        "flag_columns",
        "timestamped_records",
        "flag_value_counts",
        "observed_flag_tokens",
        "documented_flag_mapping",
        "mapped_observed_flag_tokens",
        "unmapped_observed_flag_tokens",
        "mapping_references",
        "mapping_scope",
        "mapping_status",
    )
    for field in compared_fields:
        if profile.get(field) != recomputed[field]:
            raise ValueError(f"recomputed GLERL profile differs for {local_path}: {field}")
    return {
        "local_path": profile["local_path"],
        "sha256_matches": True,
        "source_manifest_match": True,
        "profile_recomputed": True,
    }


def run(
    *,
    manifest_path: Path,
    output_path: Path | None = None,
) -> Path:
    manifest_path = _resolve(manifest_path)
    audit = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit.get("source_id") != "algal_bloom_glerl_flag_audit":
        raise ValueError("unexpected GLERL audit source ID")
    if audit.get("documented_flag_mapping") != QARTOD_FLAG_LABELS:
        raise ValueError("GLERL audit mapping differs from the documented mapping")
    if audit.get("mapping_references") != QARTOD_FLAG_REFERENCES:
        raise ValueError("GLERL audit mapping references differ from the documented references")
    if audit.get("mapping_scope") != QARTOD_FLAG_MAPPING_SCOPE:
        raise ValueError("GLERL audit mapping scope differs from the documented scope")

    source_manifest_path = _resolve(Path(str(audit["source_manifest"])))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_artifacts = {
        str(artifact["local_path"]): artifact
        for artifact in source_manifest["artifacts"]
        if artifact["metadata"].get("classification") == "moored_buoy_or_continuous"
        and not str(artifact["local_path"]).endswith("_phosphate.csv")
    }
    profiles = audit.get("profiles", [])
    if not profiles:
        raise ValueError("GLERL audit contains no profiles")
    local_paths = [str(profile["local_path"]) for profile in profiles]
    if len(local_paths) != len(set(local_paths)):
        raise ValueError("GLERL audit contains duplicate profiles")
    if len(profiles) != len(source_artifacts):
        raise ValueError("GLERL audit profile count differs from selected source artifacts")

    profile_checks = [
        _validate_profile(profile, source_artifacts=source_artifacts) for profile in profiles
    ]
    observed_tokens = sorted(
        {
            token
            for profile in profiles
            for token in profile["observed_flag_tokens"]
        }
    )
    unmapped_tokens = sorted(set(observed_tokens) - set(QARTOD_FLAG_LABELS))
    if audit.get("observed_flag_tokens") != observed_tokens:
        raise ValueError("top-level observed GLERL tokens do not match the profiles")
    if audit.get("unmapped_observed_flag_tokens") != unmapped_tokens:
        raise ValueError("top-level unmapped GLERL tokens do not match the profiles")

    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    validation = {
        "source_id": "algal_bloom_glerl_flag_validation",
        "retrieved_at": retrieved_at.isoformat(),
        "audit_manifest": _relative_or_absolute(manifest_path),
        "audit_manifest_sha256": _sha256(manifest_path),
        "source_manifest": _relative_or_absolute(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "file_count": len(profile_checks),
        "documented_flag_mapping": dict(QARTOD_FLAG_LABELS),
        "mapping_references": dict(QARTOD_FLAG_REFERENCES),
        "observed_flag_tokens": observed_tokens,
        "unmapped_observed_flag_tokens": unmapped_tokens,
        "unmapped_token_policy": "retain raw sequences; do not infer meanings",
        "profiles": profile_checks,
        "validation": {
            "status": "flag_audit_validation_complete",
            "raw_file_checksums_match": True,
            "source_manifest_checksums_match": True,
            "profiles_recomputed": True,
            "mapping_consistent": True,
        },
    }
    output = _resolve(output_path) if output_path else (
        ROOT / "data/manifests" / f"algal_bloom_glerl_flag_validation_{run_id}.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "file_count": len(profile_checks)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(manifest_path=args.manifest, output_path=args.output)


if __name__ == "__main__":
    main()
