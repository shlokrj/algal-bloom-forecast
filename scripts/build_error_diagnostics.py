#!/usr/bin/env python3
"""Build false-alarm, missed-event, and continuous-error case-study tables."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.evaluation.errors import build_error_records, build_error_summary

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} found")
    return max(paths)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run(*, selection_manifest_path: Path | None = None) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    selection_manifest_path = selection_manifest_path or _latest(
        list((ROOT / "data/manifests").glob("algal_bloom_model_selection_*.json")),
        "model selection manifest",
    )
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    prediction_path = ROOT / selection_manifest["prediction_path"]
    predictions = _read_csv(prediction_path)
    thresholds = {
        int(horizon): float(value)
        for horizon, value in selection_manifest["event_thresholds"].items()
    }
    rows = build_error_records(predictions, thresholds)
    summary = build_error_summary(rows)
    case_path = ROOT / "results/tables" / f"algal_bloom_selected_error_cases_{run_id}.csv"
    summary_path = ROOT / "results/tables" / f"algal_bloom_selected_error_summary_{run_id}.csv"
    case_fields = [
        "split",
        "forecast_horizon_days",
        "observation_date",
        "selected_model",
        "actual",
        "prediction",
        "signed_error",
        "absolute_error",
        "event_threshold",
        "event_case",
    ]
    summary_fields = [
        "forecast_horizon_days",
        "n",
        "mae",
        "max_absolute_error",
        "worst_observation_date",
        "true_positive",
        "true_negative",
        "false_alarm",
        "missed_event",
    ]
    _write_csv(case_path, rows, case_fields)
    _write_csv(summary_path, summary, summary_fields)
    manifest = {
        "source_id": "algal_bloom_selected_error_diagnostics",
        "retrieved_at": retrieved_at.isoformat(),
        "selection_manifest": str(selection_manifest_path.relative_to(ROOT)),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "case_path": str(case_path.relative_to(ROOT)),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "case_policy": "event cases use thresholds fit on train plus validation; no test labels influence selection",
        "rows": len(rows),
        "summary": summary,
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_selected_error_diagnostics_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} error cases")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path)
    args = parser.parse_args()
    run(selection_manifest_path=args.selection_manifest)


if __name__ == "__main__":
    main()
