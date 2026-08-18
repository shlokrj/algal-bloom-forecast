#!/usr/bin/env python3
"""Build one long-form horizon-performance report from all model runs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.evaluation.summary import build_best_summary, build_summary_rows

ROOT = Path(__file__).resolve().parents[1]


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} table found")
    return max(paths)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "metric_family",
        "source",
        "split",
        "forecast_horizon_days",
        "model",
        "metric",
        "value",
        "n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    baseline_metrics: Path | None = None,
    ablation_metrics: Path | None = None,
    gradient_metrics: Path | None = None,
    training_metrics: Path | None = None,
    baseline_event_metrics: Path | None = None,
    gradient_event_metrics: Path | None = None,
    training_event_metrics: Path | None = None,
) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    baseline_metrics = baseline_metrics or _latest(
        list((ROOT / "results/tables").glob("algal_bloom_baseline_metrics_*.csv")),
        "baseline metrics",
    )
    ablation_metrics = ablation_metrics or _latest(
        list((ROOT / "results/tables").glob("algal_bloom_feature_ablation_metrics_*.csv")),
        "feature ablation metrics",
    )
    gradient_metrics = gradient_metrics or _latest(
        list((ROOT / "results/tables").glob("algal_bloom_gradient_metrics_*.csv")),
        "gradient metrics",
    )
    training_metrics = training_metrics or _latest(
        list((ROOT / "results/tables").glob("algal_bloom_trained_metrics_*.csv")),
        "trained model metrics",
    )
    baseline_event_metrics = baseline_event_metrics or _latest(
        list((ROOT / "results/tables").glob("algal_bloom_event_metrics_*.csv")),
        "baseline event metrics",
    )
    gradient_event_metrics = gradient_event_metrics or _latest(
        list((ROOT / "results/tables").glob("algal_bloom_gradient_event_metrics_*.csv")),
        "gradient event metrics",
    )
    training_event_metrics = training_event_metrics or _latest(
        list((ROOT / "results/tables").glob("algal_bloom_trained_event_metrics_*.csv")),
        "trained model event metrics",
    )
    rows = build_summary_rows(
        {
            "baseline": _read_csv(baseline_metrics),
            "feature_ablation": _read_csv(ablation_metrics),
            "gradient_boosted": _read_csv(gradient_metrics),
            "trained_model": _read_csv(training_metrics),
        },
        {
            "baseline_event": _read_csv(baseline_event_metrics),
            "gradient_event": _read_csv(gradient_event_metrics),
            "trained_event": _read_csv(training_event_metrics),
        },
    )
    summary_path = ROOT / "results/tables" / f"algal_bloom_horizon_summary_{run_id}.csv"
    _write_rows(summary_path, rows)
    manifest = {
        "source_id": "algal_bloom_horizon_summary",
        "retrieved_at": retrieved_at.isoformat(),
        "inputs": {
            "baseline_metrics": str(baseline_metrics.relative_to(ROOT)),
            "feature_ablation_metrics": str(ablation_metrics.relative_to(ROOT)),
            "gradient_metrics": str(gradient_metrics.relative_to(ROOT)),
            "training_metrics": str(training_metrics.relative_to(ROOT)),
            "baseline_event_metrics": str(baseline_event_metrics.relative_to(ROOT)),
            "gradient_event_metrics": str(gradient_event_metrics.relative_to(ROOT)),
            "training_event_metrics": str(training_event_metrics.relative_to(ROOT)),
        },
        "summary_path": str(summary_path.relative_to(ROOT)),
        "rows": len(rows),
        "best_by_split_and_horizon": build_best_summary(rows),
        "selection_policy": "report best values descriptively; do not tune or select on the test split",
        "caveat": "held-out validation and test samples are small and event probabilities are hard threshold diagnostics",
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_horizon_summary_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} horizon summary rows")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--ablation-metrics", type=Path)
    parser.add_argument("--gradient-metrics", type=Path)
    parser.add_argument("--training-metrics", type=Path)
    parser.add_argument("--baseline-event-metrics", type=Path)
    parser.add_argument("--gradient-event-metrics", type=Path)
    parser.add_argument("--training-event-metrics", type=Path)
    args = parser.parse_args()
    run(
        baseline_metrics=args.baseline_metrics,
        ablation_metrics=args.ablation_metrics,
        gradient_metrics=args.gradient_metrics,
        training_metrics=args.training_metrics,
        baseline_event_metrics=args.baseline_event_metrics,
        gradient_event_metrics=args.gradient_event_metrics,
        training_event_metrics=args.training_event_metrics,
    )


if __name__ == "__main__":
    main()
