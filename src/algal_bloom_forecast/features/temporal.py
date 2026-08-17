"""Temporal contracts for forecast origins, features, and future targets."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable


def _require_aware(timestamp: datetime, *, name: str) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


def assert_features_available(
    feature_timestamps: Iterable[datetime],
    forecast_origin: datetime,
) -> None:
    """Reject any feature observation that was not available at the origin."""
    origin = _require_aware(forecast_origin, name="forecast_origin")
    for feature_timestamp in feature_timestamps:
        feature_time = _require_aware(feature_timestamp, name="feature_timestamp")
        if feature_time > origin:
            raise ValueError(
                "feature timestamp is after forecast origin: "
                f"{feature_time.isoformat()} > {origin.isoformat()}"
            )


def build_target_timestamp(forecast_origin: datetime, horizon_days: int) -> datetime:
    """Build a future target timestamp for a positive forecast horizon."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    origin = _require_aware(forecast_origin, name="forecast_origin")
    return origin + timedelta(days=horizon_days)


def assert_target_after_origin(target_timestamp: datetime, forecast_origin: datetime) -> None:
    """Reject targets that are not strictly after the forecast origin."""
    target = _require_aware(target_timestamp, name="target_timestamp")
    origin = _require_aware(forecast_origin, name="forecast_origin")
    if target <= origin:
        raise ValueError(
            "target timestamp must be after forecast origin: "
            f"{target.isoformat()} <= {origin.isoformat()}"
        )
