#!/usr/bin/env python3
"""Validate the GLERL annual-summary accession scope in the immutable inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "data/manifests/ncei_source_inventory_20260817T221557Z.json"
AUDITED_ACCESSIONS = ("0190201", "0190729", "0194301", "0194302")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    """Return the annual-summary scope without downloading or altering source data."""
    records: list[dict[str, Any]] = []
    for source in inventory.get("sources", []):
        accession = str(source.get("accession", ""))
        for version in source.get("versions", []):
            for item in version.get("items", []):
                path = str(item.get("path", ""))
                if "annual_summary" not in path.lower():
                    continue
                records.append(
                    {
                        "accession": accession,
                        "path": path,
                        "classification": item.get("classification"),
                        "url": item.get("url"),
                    }
                )
    accessions = sorted({record["accession"] for record in records})
    outside_scope = [
        record for record in records if record["accession"] not in AUDITED_ACCESSIONS
    ]
    classifications = Counter(str(record["classification"]) for record in records)
    return {
        "annual_summary_file_count": len(records),
        "annual_summary_accessions": accessions,
        "annual_summary_files_by_accession": dict(
            sorted(Counter(record["accession"] for record in records).items())
        ),
        "annual_summary_classifications": dict(sorted(classifications.items())),
        "outside_audited_scope": outside_scope,
        "annual_summary_files": records,
    }


def run(*, inventory_path: Path = DEFAULT_INVENTORY) -> Path:
    inventory_path = _resolve(inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("source_id") != "ncei_historical_source_inventory":
        raise ValueError("unexpected NCEI inventory source ID")
    scope = inspect_inventory(inventory)
    if scope["outside_audited_scope"]:
        raise ValueError(
            "annual-summary files found outside the audited accessions: "
            f"{scope['outside_audited_scope']}"
        )
    if scope["annual_summary_accessions"] != list(AUDITED_ACCESSIONS):
        raise ValueError(
            "annual-summary accession set differs from the audited set: "
            f"{scope['annual_summary_accessions']}"
        )
    if set(scope["annual_summary_classifications"]) != {"moored_buoy_or_continuous"}:
        raise ValueError("annual-summary inventory contains an unexpected classification")

    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    report = {
        "source_id": "algal_bloom_glerl_annual_summary_scope_validation",
        "retrieved_at": retrieved_at.isoformat(),
        "source_manifest": str(inventory_path.relative_to(ROOT)),
        "source_manifest_sha256": _sha256(inventory_path),
        "audited_accessions": list(AUDITED_ACCESSIONS),
        "scope": scope,
        "validation": {
            "status": "annual_summary_scope_validation_complete",
            "inventory_scope_only": True,
            "no_additional_annual_summary_accessions_in_inventory": True,
            "all_annual_summary_files_are_moored_buoy_or_continuous": True,
            "quality_flag_mapping_ready_for_scope": True,
            "future_accession_policy": (
                "rerun this validation and obtain accession-level metadata before using any "
                "new annual-summary accession"
            ),
        },
    }
    output = ROOT / "data/manifests" / f"algal_bloom_glerl_annual_summary_scope_validation_{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "annual_summary_file_count": scope["annual_summary_file_count"]}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()
    run(inventory_path=args.inventory)


if __name__ == "__main__":
    main()
