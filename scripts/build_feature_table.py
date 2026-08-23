#!/usr/bin/env python3
"""Build the first normalized, lagged, rolling, and seasonal feature table."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algal_bloom_forecast.data.normalization import build_normalization_contract
from algal_bloom_forecast.features.engineering import (
    DEFAULT_LAG_DAYS,
    DEFAULT_ROLLING_WINDOWS_DAYS,
    build_feature_records,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HORIZONS = (1, 3, 7, 14)


def _latest(paths: list[Path], label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"no {label} table found")
    return max(paths)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _coerce_value(field: str, value: str | None) -> Any:
    if value in (None, ""):
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    if field.endswith(("_count", "_flag")):
        try:
            return int(value)
        except ValueError:
            pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _coerce_value(key, value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _write_csv(path: Path, records: list[dict[str, Any]]) -> list[str]:
    leading = ["forecast_horizon_days", "observation_date", "predictor_date", "feature_lag_days"]
    remaining = sorted({field for record in records for field in record if field not in leading})
    fields = leading + remaining
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return fields


def run(
    *,
    target_path: Path | None = None,
    predictor_path: Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    target_path = _resolve(target_path) if target_path else _latest(
        list((ROOT / "data/processed").glob("noaa_western_lake_erie_historical_target_*.csv")),
        "historical target",
    )
    predictor_path = _resolve(predictor_path) if predictor_path else _latest(
        list((ROOT / "data/processed").glob("algal_bloom_daily_predictors_*.csv")),
        "daily predictor",
    )
    target_records = _read_csv(target_path)
    predictor_records = _read_csv(predictor_path)
    feature_records = build_feature_records(
        target_records,
        predictor_records,
        horizons=horizons,
    )
    output_path = ROOT / "data/processed" / f"algal_bloom_feature_table_{run_id}.csv"
    fields = _write_csv(output_path, feature_records)
    manifest = {
        "source_id": "algal_bloom_feature_table",
        "retrieved_at": retrieved_at.isoformat(),
        "target_path": str(target_path.relative_to(ROOT)),
        "predictor_path": str(predictor_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "records": len(feature_records),
        "fields": fields,
        "horizons_days": list(horizons),
        "lag_days": list(DEFAULT_LAG_DAYS),
        "rolling_windows_days": list(DEFAULT_ROLLING_WINDOWS_DAYS),
        "normalization": build_normalization_contract(),
        "target_definition": build_normalization_contract()["target"],
        "leakage_policy": "all snapshots, lags, and rolling windows use predictor dates at or before target_date_minus_horizon",
        "missing_value_policy": "no interpolation; unavailable and invalid values remain null",
    }
    manifest_path = ROOT / "data/manifests" / f"algal_bloom_feature_table_{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(feature_records)} feature rows with {len(fields)} fields to {output_path}")
    print(f"wrote manifest to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path)
    parser.add_argument("--predictors", type=Path)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    args = parser.parse_args()
    run(
        target_path=args.target,
        predictor_path=args.predictors,
        horizons=tuple(args.horizons),
    )


if __name__ == "__main__":
    main()
