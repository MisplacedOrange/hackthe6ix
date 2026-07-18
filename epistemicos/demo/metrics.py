"""Deterministic in-memory metric aggregation for G11 release rehearsal.

``MetricsCollector`` consumes the numeric metric mapping emitted by G08 for
each processed event plus deterministic verdict/outcome metadata.  It does
not call providers, inspect raw evidence, or own any security decision.

Snapshot conventions:

* latency percentiles include fast-lane events only (``escalated == 0``);
* percentiles use linear interpolation over the sorted observations;
* escalation and reversal percentages use the 0--100 scale;
* cache-hit fraction is derived from aggregate cached/total token counts;
* an empty committed set is vacuously reversal-complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from threading import RLock
from typing import Any, Iterable, Mapping

from core.types import Verdict

__all__ = ["EscrowOutcome", "MetricsCollector"]


_REQUIRED_METRICS = ("latency_ms", "gemini_calls", "escalated")
_TOKEN_METRICS = (
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "cached_tokens",
    "total_tokens",
)


class EscrowOutcome(StrEnum):
    """Terminal result for evidence previously placed in escrow."""

    RELEASE = "release"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class _MetricRecord:
    latency_ms: float
    escalated: int
    gemini_calls: int
    tokens: tuple[int, int, int, int, int]
    verdict: Verdict
    escrow_outcome: EscrowOutcome | None


@dataclass(frozen=True, slots=True)
class _ValidatedMetrics:
    latency_ms: float
    escalated: int
    gemini_calls: int
    tokens: tuple[int, int, int, int, int]


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-mutated copy."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _validate_metrics(metrics: Mapping[str, float | int]) -> _ValidatedMetrics:
    """Validate and normalize numeric fields without mutating collector state."""

    for key in _REQUIRED_METRICS:
        if key not in metrics:
            raise ValueError(f"missing required metric: {key}")

    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"metric {key} must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"metric {key} must be finite and non-negative")

    latency_ms = float(metrics["latency_ms"])
    gemini_calls = _nonnegative_integer(metrics["gemini_calls"], "gemini_calls")
    escalated = _nonnegative_integer(metrics["escalated"], "escalated")
    if escalated not in (0, 1):
        raise ValueError("metric escalated must be 0 or 1")

    tokens = (
        _nonnegative_integer(metrics.get("input_tokens", 0), "input_tokens"),
        _nonnegative_integer(metrics.get("output_tokens", 0), "output_tokens"),
        _nonnegative_integer(metrics.get("thinking_tokens", 0), "thinking_tokens"),
        _nonnegative_integer(metrics.get("cached_tokens", 0), "cached_tokens"),
        _nonnegative_integer(metrics.get("total_tokens", 0), "total_tokens"),
    )
    cached_tokens = tokens[3]
    total_tokens = tokens[4]
    if cached_tokens > total_tokens:
        raise ValueError("metric cached_tokens cannot exceed total_tokens")

    cache_fraction = metrics.get("cache_hit_fraction")
    if cache_fraction is not None and float(cache_fraction) > 1.0:
        raise ValueError("metric cache_hit_fraction must be in [0, 1]")

    return _ValidatedMetrics(
        latency_ms=latency_ms,
        escalated=escalated,
        gemini_calls=gemini_calls,
        tokens=tokens,
    )


def _nonnegative_integer(value: float | int, key: str) -> int:
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"metric {key} must be an integer")
    return int(numeric)


def _verdict(value: Verdict | str) -> Verdict:
    try:
        return Verdict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid verdict: {value!r}") from exc


def _escrow_outcome(value: EscrowOutcome | str | None) -> EscrowOutcome | None:
    if value is None:
        return None
    try:
        return EscrowOutcome(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid escrow_outcome: {value!r}") from exc


def _identifiers(values: Iterable[str], field_name: str) -> frozenset[str]:
    normalized: set[str] = set()
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} identifiers must be iterable") from exc
    for value in iterator:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} identifier must be a string")
        if not value.strip():
            raise ValueError(f"{field_name} identifier must be non-empty")
        normalized.add(value)
    return frozenset(normalized)


class MetricsCollector:
    """Thread-safe, deterministic accumulator for per-event G08 metrics."""

    def __init__(self) -> None:
        self._records: list[_MetricRecord] = []
        self._committed_identifiers: set[str] = set()
        self._reversed_identifiers: set[str] = set()
        self._retraction_expected: set[str] = set()
        self._retraction_reversed: set[str] = set()
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def ingest(
        self,
        metrics: Mapping[str, float | int],
        *,
        verdict: Verdict | str,
        escrow_outcome: EscrowOutcome | str | None = None,
        committed_identifiers: Iterable[str] = (),
        reversed_identifiers: Iterable[str] = (),
    ) -> None:
        """Validate and atomically ingest one event's metrics and metadata."""

        base = _validate_metrics(metrics)
        normalized_verdict = _verdict(verdict)
        normalized_outcome = _escrow_outcome(escrow_outcome)
        committed = _identifiers(committed_identifiers, "committed")
        reversed_ids = _identifiers(reversed_identifiers, "reversed")
        record = _MetricRecord(
            latency_ms=base.latency_ms,
            escalated=base.escalated,
            gemini_calls=base.gemini_calls,
            tokens=base.tokens,
            verdict=normalized_verdict,
            escrow_outcome=normalized_outcome,
        )

        with self._lock:
            self._records.append(record)
            self._committed_identifiers.update(committed)
            self._reversed_identifiers.update(reversed_ids)

    def record_reversals(self, identifiers: Iterable[str]) -> None:
        """Record ledger reversals without inventing another pipeline event.

        Retraction and reversal are ledger operations, so they must update the
        reversal audit denominator without increasing the evidence-event count
        or distorting latency/token aggregates.
        """

        reversed_ids = _identifiers(identifiers, "reversed")
        with self._lock:
            self._reversed_identifiers.update(reversed_ids)

    def record_retraction(
        self,
        expected_identifiers: Iterable[str],
        reversed_identifiers: Iterable[str],
    ) -> None:
        """Record the exact target and outcome of a retraction operation."""
        expected = _identifiers(expected_identifiers, "expected")
        reversed_ids = _identifiers(reversed_identifiers, "reversed")
        if not reversed_ids <= expected:
            raise ValueError("reversed identifiers must belong to the retraction target")
        with self._lock:
            self._retraction_expected.update(expected)
            self._retraction_reversed.update(reversed_ids)
            self._reversed_identifiers.update(reversed_ids)

    def snapshot(self) -> dict[str, Any]:
        """Return a fresh, JSON-ready aggregate snapshot."""

        with self._lock:
            records = list(self._records)
            committed = set(self._committed_identifiers)
            reversed_ids = set(self._reversed_identifiers)
            retraction_expected = set(self._retraction_expected)
            retraction_reversed = set(self._retraction_reversed)

        count = len(records)
        fast_latencies = [
            record.latency_ms for record in records if record.escalated == 0
        ]
        escalated_count = sum(record.escalated for record in records)
        gemini_calls = sum(record.gemini_calls for record in records)
        token_totals = {
            key: sum(record.tokens[index] for record in records)
            for index, key in enumerate(_TOKEN_METRICS)
        }

        verdict_counts = {verdict.value: 0 for verdict in Verdict}
        escrow_outcomes = {outcome.value: 0 for outcome in EscrowOutcome}
        for record in records:
            verdict_counts[record.verdict.value] += 1
            if record.escrow_outcome is not None:
                escrow_outcomes[record.escrow_outcome.value] += 1

        reversal_target = retraction_expected or committed
        reversal_actual = retraction_reversed if retraction_expected else reversed_ids
        matched_reversals = reversal_target & reversal_actual
        missing = sorted(reversal_target - reversal_actual)
        committed_count = len(reversal_target)
        reversed_count = len(matched_reversals)
        reversal_fraction = (
            reversed_count / committed_count if committed_count else 1.0
        )
        total_tokens = token_totals["total_tokens"]
        cache_fraction = (
            token_totals["cached_tokens"] / total_tokens if total_tokens else 0.0
        )

        return {
            "count": count,
            "fast_lane_latency_ms": {
                "p50": _percentile(fast_latencies, 0.50),
                "p95": _percentile(fast_latencies, 0.95),
            },
            "escalation_percentage": (
                escalated_count / count * 100.0 if count else 0.0
            ),
            "average_gemini_calls_per_event": (
                gemini_calls / count if count else 0.0
            ),
            "token_totals": token_totals,
            "cache_hit_token_fraction": cache_fraction,
            "verdict_counts": verdict_counts,
            "escrow_outcomes": escrow_outcomes,
            "reversal_completeness": {
                "committed_count": committed_count,
                "reversed_count": reversed_count,
                "fraction": reversal_fraction,
                "percentage": reversal_fraction * 100.0,
                "complete": not missing,
                "missing_identifiers": missing,
            },
        }
