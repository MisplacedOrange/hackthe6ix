"""Provider-free tests for deterministic G11 metric collection."""

from __future__ import annotations

import json
import math

import pytest

from core.types import Verdict
from demo.metrics import EscrowOutcome, MetricsCollector


def event_metrics(
    *,
    latency_ms: float = 10.0,
    escalated: int = 0,
    gemini_calls: int = 2,
    input_tokens: int = 100,
    output_tokens: int = 20,
    thinking_tokens: int = 5,
    cached_tokens: int = 25,
    total_tokens: int = 125,
) -> dict[str, float | int]:
    return {
        "latency_ms": latency_ms,
        "escalated": escalated,
        "gemini_calls": gemini_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
        "cache_hit_fraction": cached_tokens / total_tokens if total_tokens else 0.0,
    }


def test_empty_snapshot_has_stable_json_ready_shape() -> None:
    snapshot = MetricsCollector().snapshot()

    assert snapshot == {
        "count": 0,
        "fast_lane_latency_ms": {"p50": 0.0, "p95": 0.0},
        "escalation_percentage": 0.0,
        "average_gemini_calls_per_event": 0.0,
        "token_totals": {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        },
        "cache_hit_token_fraction": 0.0,
        "verdict_counts": {
            "commit": 0,
            "provisional": 0,
            "escrow": 0,
            "reject": 0,
        },
        "escrow_outcomes": {"release": 0, "reject": 0},
        "reversal_completeness": {
            "committed_count": 0,
            "reversed_count": 0,
            "fraction": 1.0,
            "percentage": 100.0,
            "complete": True,
            "missing_identifiers": [],
        },
    }
    assert json.loads(json.dumps(snapshot, allow_nan=False, sort_keys=True)) == snapshot


def test_single_fast_lane_event_reports_its_latency_and_metadata() -> None:
    collector = MetricsCollector()
    collector.ingest(
        event_metrics(latency_ms=17.5),
        verdict=Verdict.COMMIT,
        committed_identifiers=["EV-1"],
        reversed_identifiers=["EV-1"],
    )

    snapshot = collector.snapshot()
    assert snapshot["count"] == 1
    assert snapshot["fast_lane_latency_ms"] == {"p50": 17.5, "p95": 17.5}
    assert snapshot["escalation_percentage"] == 0.0
    assert snapshot["average_gemini_calls_per_event"] == 2.0
    assert snapshot["cache_hit_token_fraction"] == pytest.approx(0.2)
    assert snapshot["verdict_counts"]["commit"] == 1
    assert snapshot["reversal_completeness"]["complete"] is True


def test_aggregation_uses_fast_lane_percentiles_and_global_token_totals() -> None:
    collector = MetricsCollector()
    events = [
        (event_metrics(latency_ms=10.0, cached_tokens=10), Verdict.COMMIT),
        (event_metrics(latency_ms=20.0, cached_tokens=20), Verdict.PROVISIONAL),
        (
            event_metrics(
                latency_ms=999.0,
                escalated=1,
                gemini_calls=3,
                cached_tokens=0,
            ),
            Verdict.ESCROW,
        ),
        (event_metrics(latency_ms=40.0, gemini_calls=1, cached_tokens=40), Verdict.REJECT),
    ]
    for metrics, verdict in events:
        collector.ingest(metrics, verdict=verdict)

    snapshot = collector.snapshot()
    assert snapshot["count"] == 4
    # Linear interpolation on fast-lane latencies [10, 20, 40].
    assert snapshot["fast_lane_latency_ms"]["p50"] == pytest.approx(20.0)
    assert snapshot["fast_lane_latency_ms"]["p95"] == pytest.approx(38.0)
    assert snapshot["escalation_percentage"] == pytest.approx(25.0)
    assert snapshot["average_gemini_calls_per_event"] == pytest.approx(2.0)
    assert snapshot["token_totals"] == {
        "input_tokens": 400,
        "output_tokens": 80,
        "thinking_tokens": 20,
        "cached_tokens": 70,
        "total_tokens": 500,
    }
    assert snapshot["cache_hit_token_fraction"] == pytest.approx(70 / 500)
    assert snapshot["verdict_counts"] == {
        "commit": 1,
        "provisional": 1,
        "escrow": 1,
        "reject": 1,
    }


def test_all_escalated_events_leave_fast_lane_percentiles_at_zero() -> None:
    collector = MetricsCollector()
    collector.ingest(event_metrics(latency_ms=50.0, escalated=1), verdict="escrow")
    collector.ingest(event_metrics(latency_ms=70.0, escalated=1), verdict="reject")

    snapshot = collector.snapshot()
    assert snapshot["fast_lane_latency_ms"] == {"p50": 0.0, "p95": 0.0}
    assert snapshot["escalation_percentage"] == 100.0


def test_optional_token_metrics_default_to_zero_for_rejected_fast_lane_input() -> None:
    collector = MetricsCollector()
    collector.ingest(
        {
            "latency_ms": 1.5,
            "gemini_calls": 0,
            "escalated": 0,
            "cache_hit_fraction": 0.0,
        },
        verdict=Verdict.REJECT,
    )

    snapshot = collector.snapshot()
    assert snapshot["token_totals"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }
    assert snapshot["cache_hit_token_fraction"] == 0.0


def test_escrow_outcomes_accept_enums_and_stable_strings() -> None:
    collector = MetricsCollector()
    collector.ingest(
        event_metrics(),
        verdict=Verdict.ESCROW,
        escrow_outcome=EscrowOutcome.RELEASE,
    )
    collector.ingest(
        event_metrics(),
        verdict=Verdict.REJECT,
        escrow_outcome="reject",
    )

    assert collector.snapshot()["escrow_outcomes"] == {"release": 1, "reject": 1}


def test_reversal_completeness_uses_unique_intersection_and_sorted_missing_ids() -> None:
    collector = MetricsCollector()
    collector.ingest(
        event_metrics(),
        verdict="commit",
        committed_identifiers=["EV-2", "EV-1", "EV-2"],
    )
    collector.ingest(
        event_metrics(),
        verdict="commit",
        committed_identifiers=["EV-3"],
        reversed_identifiers=["EV-1", "UNKNOWN"],
    )

    reversal = collector.snapshot()["reversal_completeness"]
    assert reversal == {
        "committed_count": 3,
        "reversed_count": 1,
        "fraction": pytest.approx(1 / 3),
        "percentage": pytest.approx(100 / 3),
        "complete": False,
        "missing_identifiers": ["EV-2", "EV-3"],
    }


def test_explicit_retraction_scope_reports_operation_completeness() -> None:
    collector = MetricsCollector()
    collector.ingest(
        event_metrics(),
        verdict="commit",
        committed_identifiers=["ROOT-A-1", "ROOT-B-1"],
    )
    collector.record_retraction(["ROOT-A-1"], ["ROOT-A-1"])

    assert collector.snapshot()["reversal_completeness"] == {
        "committed_count": 1,
        "reversed_count": 1,
        "fraction": 1.0,
        "percentage": 100.0,
        "complete": True,
        "missing_identifiers": [],
    }


def test_ingest_copies_inputs_and_snapshot_returns_fresh_data() -> None:
    collector = MetricsCollector()
    metrics = event_metrics(latency_ms=12.0)
    committed = ["EV-1"]
    collector.ingest(
        metrics,
        verdict=Verdict.COMMIT,
        committed_identifiers=committed,
    )
    metrics["latency_ms"] = 999.0
    committed.append("EV-2")

    first = collector.snapshot()
    first["fast_lane_latency_ms"]["p50"] = 999.0
    first["verdict_counts"]["commit"] = 99
    second = collector.snapshot()

    assert second["fast_lane_latency_ms"]["p50"] == 12.0
    assert second["verdict_counts"]["commit"] == 1
    assert second["reversal_completeness"]["committed_count"] == 1


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    [
        "latency_ms",
        "gemini_calls",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
        "cached_tokens",
        "total_tokens",
        "cache_hit_fraction",
    ],
)
def test_rejects_negative_or_nonfinite_metric_values(field: str, bad: float) -> None:
    metrics = event_metrics()
    metrics[field] = bad

    with pytest.raises(ValueError, match=field):
        MetricsCollector().ingest(metrics, verdict=Verdict.COMMIT)


@pytest.mark.parametrize("missing", ["latency_ms", "gemini_calls", "escalated"])
def test_rejects_missing_required_metrics(missing: str) -> None:
    metrics = event_metrics()
    del metrics[missing]

    with pytest.raises(ValueError, match=missing):
        MetricsCollector().ingest(metrics, verdict=Verdict.COMMIT)


def test_rejects_non_numeric_metrics_and_non_binary_escalation() -> None:
    metrics = event_metrics()
    metrics["latency_ms"] = "fast"  # type: ignore[assignment]
    with pytest.raises(TypeError, match="latency_ms"):
        MetricsCollector().ingest(metrics, verdict=Verdict.COMMIT)

    metrics = event_metrics(escalated=2)
    with pytest.raises(ValueError, match="escalated"):
        MetricsCollector().ingest(metrics, verdict=Verdict.COMMIT)


def test_rejects_inconsistent_cache_metrics() -> None:
    metrics = event_metrics(cached_tokens=126, total_tokens=125)
    with pytest.raises(ValueError, match="cached_tokens"):
        MetricsCollector().ingest(metrics, verdict=Verdict.COMMIT)

    metrics = event_metrics()
    metrics["cache_hit_fraction"] = 1.1
    with pytest.raises(ValueError, match="cache_hit_fraction"):
        MetricsCollector().ingest(metrics, verdict=Verdict.COMMIT)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"verdict": "approve"}, "verdict"),
        ({"verdict": "commit", "escrow_outcome": "maybe"}, "escrow_outcome"),
        ({"verdict": "commit", "committed_identifiers": [""]}, "identifier"),
        ({"verdict": "commit", "reversed_identifiers": [1]}, "identifier"),
    ],
)
def test_rejects_invalid_event_metadata(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        MetricsCollector().ingest(event_metrics(), **kwargs)  # type: ignore[arg-type]


def test_failed_ingest_is_atomic() -> None:
    collector = MetricsCollector()
    collector.ingest(event_metrics(), verdict="commit")

    with pytest.raises(ValueError):
        collector.ingest(event_metrics(latency_ms=-1.0), verdict="reject")

    assert collector.snapshot()["count"] == 1
    assert collector.snapshot()["verdict_counts"]["reject"] == 0
