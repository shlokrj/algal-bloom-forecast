#!/usr/bin/env python3
"""Audit GLERL quality-flag sequences without guessing their meanings."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from algal_bloom_forecast.data.glerl import (
    QARTOD_FLAG_LABELS,
    QARTOD_FLAG_MAPPING_SCOPE,
    QARTOD_FLAG_REFERENCES,
    profile_glerl_flag_codes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/manifests/noaa_glerl_observations_20260817T221729Z.json"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(*, manifest_path: Path = DEFAULT_MANIFEST, include_phosphate: bool = False) -> Path:
    manifest_path = _resolve(manifest_path)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = [
        artifact
        for artifact in source_manifest["artifacts"]
        if artifact["metadata"].get("classification") == "moored_buoy_or_continuous"
        and (include_phosphate or not artifact["local_path"].endswith("_phosphate.csv"))
    ]
    if not artifacts:
        raise FileNotFoundError("no selected GLERL continuous artifacts found")

    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    profiles = []
    for artifact in artifacts:
        local_path = ROOT / artifact["local_path"]
        profile = profile_glerl_flag_codes(local_path)
        profiles.append(
            {
                "local_path": artifact["local_path"],
                "sha256": artifact["sha256"],
                **profile,
            }
        )

    observed_tokens = sorted(
        {
            token
            for profile in profiles
            for token in profile["observed_flag_tokens"]
        }
    )
    unmapped_tokens = sorted(set(observed_tokens) - set(QARTOD_FLAG_LABELS))

    report = {
        "source_id": "algal_bloom_glerl_flag_audit",
        "retrieved_at": retrieved_at.isoformat(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "selection_policy": (
            "continuous GLERL annual-summary artifacts excluding phosphate tables"
            if not include_phosphate
            else "all continuous GLERL annual-summary artifacts including phosphate tables"
        ),
        "file_count": len(profiles),
        "profiles": profiles,
        "mapping_status": (
            "audit complete; documented flag subset mapped; unlisted observed tokens remain "
            "unresolved"
        ),
        "documented_flag_mapping": dict(QARTOD_FLAG_LABELS),
        "mapping_references": dict(QARTOD_FLAG_REFERENCES),
        "mapping_scope": QARTOD_FLAG_MAPPING_SCOPE,
        "observed_flag_tokens": observed_tokens,
        "unmapped_observed_flag_tokens": unmapped_tokens,
        "unmapped_token_policy": "retain raw sequences; do not infer meanings",
        "spatial_metadata": {
            "dataset_bbox": {
                "north": 43.3,
                "south": 41.8,
                "east": -82.8,
                "west": -83.7084,
            },
            "bbox_source": "NCEI GLERL-CIGLR-HAB-LakeErie-water-qual metadata",
            "feature_coordinate_mapping": "not implemented; retain station prefixes and do not spatially interpolate",
        },
    }
    output = ROOT / "data/manifests" / f"algal_bloom_glerl_flag_audit_{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "file_count": len(profiles)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--include-phosphate", action="store_true")
    args = parser.parse_args()
    run(manifest_path=args.manifest, include_phosphate=args.include_phosphate)


if __name__ == "__main__":
    main()
