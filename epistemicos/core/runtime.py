"""Deterministic runtime-state reconstruction from the verified event ledger.

The SQLite ledger is the durable source of truth.  This module rebuilds the
in-memory state that otherwise disappears on process restart: root-budget
spend, the engine spend audit log, retracted roots, source submission counts,
and idempotent response metadata.  It deliberately selects response fields
instead of copying whole payloads, so raw L0 input can never enter the cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from core.engine import ROOT_BUDGET, Engine, SpendRecord
from core.ledger import Event, EventType, Ledger
from core.types import Integrity, Verdict

__all__ = [
    "ReconstructedRuntime",
    "RuntimeReconstructionError",
    "reconstruct_runtime",
]


_INGESTION_VERDICTS: dict[EventType, Verdict] = {
    EventType.EVIDENCE_COMMITTED: Verdict.COMMIT,
    EventType.EVIDENCE_PROVISIONAL: Verdict.PROVISIONAL,
    EventType.EVIDENCE_ESCROWED: Verdict.ESCROW,
    EventType.EVIDENCE_REJECTED: Verdict.REJECT,
}
_APPLYING = frozenset(
    {EventType.EVIDENCE_COMMITTED, EventType.ESCROW_RELEASED}
)
_ENGINE_FIELDS = (
    "prior",
    "raw_bf",
    "bounded_delta",
    "root_spent",
    "posterior",
    "integrity",
)


class RuntimeReconstructionError(RuntimeError):
    """Raised when durable events cannot safely reconstruct runtime state."""


@dataclass(slots=True)
class ReconstructedRuntime:
    """Fresh mutable containers suitable for installation by the API layer."""

    engine: Engine
    retracted_roots: set[str]
    source_counts: dict[str, int]
    idempotent_responses: dict[str, dict[str, Any]]


def _error(event: Event | None, reason: str) -> RuntimeReconstructionError:
    prefix = "ledger" if event is None else f"event {event.seq}"
    return RuntimeReconstructionError(f"{prefix}: {reason}")


def _nonempty_string(event: Event, payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise _error(event, f"{field} must be a non-empty string")
    return value


def _finite_number(
    event: Event,
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(event, f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise _error(event, f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise _error(event, f"{field} must be >= {minimum}")
    return result


def _integrity(event: Event, value: object) -> Integrity:
    if isinstance(value, bool):
        raise _error(event, "integrity must be an Integrity value")
    try:
        return Integrity(value)
    except (TypeError, ValueError) as exc:
        raise _error(event, "integrity must be an Integrity value") from exc


def _reversed_sequences(events: list[Event]) -> set[int]:
    applying = {event.seq: event for event in events if event.event_type in _APPLYING}
    reversed_sequences: set[int] = set()
    for event in events:
        if event.event_type is not EventType.REVERSAL:
            continue
        sequence = event.payload.get("reverses_seq")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise _error(event, "reverses_seq must be an integer")
        original = applying.get(sequence)
        if original is None or sequence >= event.seq:
            raise _error(event, "reverses_seq must identify a prior applying event")
        if sequence in reversed_sequences:
            raise _error(event, f"applying event {sequence} is reversed more than once")

        original_delta = _finite_number(original, original.payload, "delta")
        reversal_delta = _finite_number(event, event.payload, "delta")
        if reversal_delta != -original_delta:
            raise _error(event, "reversal delta does not exactly negate original delta")
        if event.payload.get("claim_id") != original.payload.get("claim_id"):
            raise _error(event, "reversal claim_id does not match original event")
        if event.payload.get("evidence_id") != original.payload.get("evidence_id"):
            raise _error(event, "reversal evidence_id does not match original event")
        reversed_sequences.add(sequence)
    return reversed_sequences


def _spend_record(event: Event) -> SpendRecord:
    payload = event.payload
    engine = payload.get("engine")
    if not isinstance(engine, Mapping):
        raise _error(event, "applying event is missing engine breakdown")

    prior = _finite_number(event, engine, "prior", minimum=0.0)
    posterior = _finite_number(event, engine, "posterior", minimum=0.0)
    raw_bf = _finite_number(event, engine, "raw_bf", minimum=0.0)
    delta = _finite_number(event, engine, "bounded_delta")
    root_spent = _finite_number(event, engine, "root_spent", minimum=0.0)
    integrity = _integrity(event, engine.get("integrity"))
    payload_delta = _finite_number(event, payload, "delta")
    root_cost = _finite_number(event, payload, "root_cost", minimum=0.0)
    logit_delta = _finite_number(event, payload, "logit_delta")
    event_kl = _finite_number(event, payload, "event_kl", minimum=0.0)

    if raw_bf <= 0.0:
        raise _error(event, "raw_bf must be > 0")
    if posterior > 1.0 or prior > 1.0:
        raise _error(event, "prior and posterior must be in [0, 1]")
    if payload_delta != delta or prior + delta != posterior:
        raise _error(event, "engine delta is inconsistent with committed payload")
    if root_spent > ROOT_BUDGET + 1e-12:
        raise _error(event, "engine root_spent exceeds ROOT_BUDGET")
    if not math.isclose(root_cost, abs(logit_delta), rel_tol=0.0, abs_tol=1e-12):
        raise _error(event, "root_cost does not match absolute logit_delta")

    return SpendRecord(
        claim_id=_nonempty_string(event, payload, "claim_id"),
        root_experiment_id=_nonempty_string(event, payload, "root_experiment_id"),
        source_id=_nonempty_string(event, payload, "source_id"),
        prior=prior,
        posterior=posterior,
        delta=delta,
        kl=event_kl,
        logit_delta=logit_delta,
        root_cost=root_cost,
        integrity=integrity,
    )


def _safe_metrics(event: Event, value: object) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise _error(event, "metrics must be a mapping")
    metrics: dict[str, float | int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise _error(event, "metric names must be strings")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _error(event, f"metric {key!r} must be numeric")
        if not math.isfinite(float(item)):
            raise _error(event, f"metric {key!r} must be finite")
        metrics[key] = item
    return metrics


def _safe_engine(event: Event, value: object) -> dict[str, float | int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _error(event, "engine metadata must be a mapping or null")
    if set(value) != set(_ENGINE_FIELDS):
        raise _error(event, "engine metadata fields differ from the contract")
    safe: dict[str, float | int] = {}
    for field in _ENGINE_FIELDS:
        if field == "integrity":
            safe[field] = int(_integrity(event, value[field]))
        else:
            safe[field] = _finite_number(event, value, field)
    return safe


def _safe_response(event: Event, verdict: Verdict) -> tuple[str, dict[str, Any]]:
    payload = event.payload
    evidence_id = _nonempty_string(event, payload, "evidence_id")
    reasons = payload.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise _error(event, "reasons must be a list of strings")
    integrity = _integrity(event, payload.get("integrity"))
    shock = _finite_number(event, payload, "shock", minimum=0.0)
    recorded_verdict = payload.get("verdict")
    if recorded_verdict != verdict.value:
        raise _error(event, "payload verdict does not match event type")

    return evidence_id, {
        "monitor": {
            "verdict": verdict.value,
            "reasons": list(reasons),
            "integrity": int(integrity),
            "shock": shock,
        },
        "engine": _safe_engine(event, payload.get("engine")),
        "event_seq": event.seq,
        "metrics": _safe_metrics(event, payload.get("metrics")),
    }


def reconstruct_runtime(ledger: Ledger) -> ReconstructedRuntime:
    """Reconstruct restart-sensitive state after verifying the full hash chain.

    Already reversed commits remain in history and in source submission counts,
    but are excluded from root spend and the active engine spend log.
    """

    valid, reason = ledger.verify_chain()
    if not valid:
        raise _error(None, f"hash-chain verification failed: {reason}")
    events = ledger.events()
    reversed_sequences = _reversed_sequences(events)

    engine = Engine()
    root_spend: dict[str, float] = {}
    spend_log: list[SpendRecord] = []
    retracted_roots: set[str] = set()
    source_counts: dict[str, int] = {}
    responses: dict[str, dict[str, Any]] = {}

    for event in events:
        payload = event.payload
        if event.event_type is EventType.RETRACTION:
            retracted_roots.add(
                _nonempty_string(event, payload, "root_experiment_id")
            )

        verdict = _INGESTION_VERDICTS.get(event.event_type)
        if verdict is not None:
            source_id = _nonempty_string(event, payload, "source_id")
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            evidence_id, response = _safe_response(event, verdict)
            if evidence_id in responses:
                raise _error(event, f"duplicate evidence_id {evidence_id!r}")
            responses[evidence_id] = response

        if event.event_type in _APPLYING and event.seq not in reversed_sequences:
            record = _spend_record(event)
            root_id = record["root_experiment_id"]
            total = root_spend.get(root_id, 0.0) + record["root_cost"]
            if total > ROOT_BUDGET + 1e-12:
                raise _error(event, f"active spend for root {root_id!r} exceeds budget")
            root_spend[root_id] = min(ROOT_BUDGET, total)
            spend_log.append(record)

    with engine._lock:
        engine._spent = root_spend
        engine.spend_log = spend_log

    return ReconstructedRuntime(
        engine=engine,
        retracted_roots=retracted_roots,
        source_counts=source_counts,
        idempotent_responses=responses,
    )
