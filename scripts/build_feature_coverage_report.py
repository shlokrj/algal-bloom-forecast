#!/usr/bin/env python3
"""Profile training-ready feature missingness by source family and split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _base_training_ready_manifests() -> list[Path]:
    return [
        path
        for path in (ROOT / "data/manifests").glob("algal_bloom_training_ready_*.json")
        if "coverage_pruned" not in path.name
    ]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def _family(field: str) -> str:
    for family in ("ndbc", "glerl", "usgs"):
        if family in field:
            return family
    if field.startswith("seasonal_"):
        return "seasonal"
    return "derived_other"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(*, frame_path: Path | None = None, manifest_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_path or _latest(
        _base_training_ready_manifests(),
        "training-ready manifest",
    )
    manifest_path = manifest_path.resolve()
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_path = (frame_path or ROOT / source_manifest["output_path"]).resolve()
    feature_names = tuple(source_manifest["schema"]["feature_names"])
    rows: list[dict[str, str]]
    with frame_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "values": 0, "missing": 0}
    )
    feature_rows: list[dict[str, Any]] = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for field in feature_names:
            family = _family(field)
            missing = sum(row.get(field, "") == "" for row in split_rows)
            feature_rows.append(
                {
                    "split": split,
                    "feature_name": field,
                    "source_family": family,
                    "rows": len(split_rows),
                    "missing": missing,
                    "missing_rate": missing / len(split_rows) if split_rows else None,
                }
            )
            bucket = counts[(split, family)]
            bucket["rows"] += len(split_rows)
            bucket["values"] += len(split_rows)
            bucket["missing"] += missing
    summary_rows = [
        {
            "split": split,
            "source_family": family,
            "feature_count": sum(
                row["source_family"] == family and row["split"] == split for row in feature_rows
            ),
            **values,
            "missing_rate": values["missing"] / values["values"] if values["values"] else None,
        }
        for (split, family), values in sorted(counts.items())
    ]
    output_path = ROOT / "results/tables" / f"algal_bloom_feature_coverage_{run_id}.csv"
    _write_csv(
        output_path,
        feature_rows,
        ["split", "feature_name", "source_family", "rows", "missing", "missing_rate"],
    )
    manifest = {
        "source_id": "algal_bloom_feature_coverage",
        "retrieved_at": retrieved_at.isoformat(),
        "training_ready_manifest": str(manifest_path.relative_to(ROOT)),
        "frame_path": str(frame_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "feature_count": len(feature_names),
        "summary": summary_rows,
        "interpretation_policy": "coverage diagnostics only; no feature removal or model tuning is performed here",
    }
    manifest_output = ROOT / "data/manifests" / f"algal_bloom_feature_coverage_{run_id}.json"
    if manifest_output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_output}")
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(feature_rows)} feature coverage rows")
    print(f"wrote manifest to {manifest_output}")
    return manifest_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    run(frame_path=args.frame, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
