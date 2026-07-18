"""Escalation lane (G07): deterministic triggers + metamorphic reparse.

Fast lane (EVIDENCE_FIREWALL_V2.md section 5): one compile and one embedding
scheduled CONCURRENTLY, local validation, monitor, verdict. No generated
explanation on the hot path.

Escalation lane: risk-triggered only. Deterministic trigger evaluation
(thresholds below) decides when to spend extra budget; the only model-shaped
escalation implemented here is a metamorphic reparse -- compile a
canonicalized variant of the same input and compare the load-bearing
interpretation fields. ANY interpretation drift (or a failed reparse) forces
the verdict down to ESCROW. Deterministic code chooses the verdict; the
model never does.

Possible-worlds analysis is a stretch goal with zero load-bearing security
value (spec section 10) and is deliberately NOT implemented here.

This module never imports core.monitor at module level: the monitor is a
parameter, duck-typed against the G05 seam
``monitor.evaluate(ir, normalized_text, state, *, ood_distance,
duplicate_signal, prior_roots) -> MonitorEvaluation`` where
``evaluation.monitor`` is a §9 MonitorResult and ``evaluation.engine`` is an
EngineBreakdown or None.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.contracts import Compiler, Embedder, MonitorResult, PipelineResult
from core.types import EvidenceIR, Integrity, Relation, Verdict
from llm.compiler import CompileError
from llm.normalize import InputTooLarge, normalize

__all__ = [
    "OOD_THRESHOLD",
    "DUP_THRESHOLD",
    "LOW_TRUST",
    "ENTRENCHED",
    "BURST_THRESHOLD",
    "SHOCK_HIGH",
    "EscalationDecision",
    "evaluate_triggers",
    "metamorphic_check",
    "canonicalize",
    "run_pipeline",
]

# ---------------------------------------------------------------------------
# Trigger thresholds (documented constants -- the deterministic escalation
# policy; see spec section 5, escalation lane)
# ---------------------------------------------------------------------------

#: OOD distance (1 - max cosine vs claim index) at/above which input is
#: near-OOD and escalates.
OOD_THRESHOLD: float = 0.65

#: Duplicate signal (max cosine vs prior evidence) at/above which the input
#: looks like a duplicate flood and escalates.
DUP_THRESHOLD: float = 0.95

#: Source trust strictly below this escalates (new sources start at 0.3, so
#: every unseen source escalates until it earns trust).
LOW_TRUST: float = 0.35

#: Contradicting a claim whose confidence is at/above this escalates
#: (entrenched-claim contradiction).
ENTRENCHED: float = 0.85

#: More than this many submissions from one source in the window escalates
#: (source burst).
BURST_THRESHOLD: int = 3

#: Shock score at/above this escalates. Matches G05's SHOCK_HIGH boundary.
SHOCK_HIGH: float = 2.0


@dataclass
class EscalationDecision:
    """Deterministic escalation verdict with machine-readable reasons."""

    escalate: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_triggers(
    *,
    shock: float,
    ood_distance: float,
    duplicate_signal: float,
    source_trust: float,
    claim_confidence: float | None,
    relation: str,
    witness_flags: list[str],
    unmodeled: bool,
    source_burst_count: int,
    missing_provenance: bool = False,
    ambiguous_claim_match: bool = False,
    weak_witnesses: bool = False,
) -> EscalationDecision:
    """Evaluate every escalation trigger; purely deterministic.

    Reasons are stable machine-readable names, with ``:value`` suffixes for
    measured triggers, so the ledger and UI never need to parse prose.
    """
    reasons: list[str] = []

    if shock >= SHOCK_HIGH:
        reasons.append(f"high_shock:{shock:.2f}")
    if ood_distance >= OOD_THRESHOLD:
        reasons.append(f"near_ood:{ood_distance:.2f}")
    if duplicate_signal >= DUP_THRESHOLD:
        reasons.append(f"duplicate:{duplicate_signal:.2f}")
    if source_trust < LOW_TRUST:
        reasons.append(f"low_trust:{source_trust:.2f}")
    if (
        relation == Relation.CONTRADICTS.value
        and claim_confidence is not None
        and claim_confidence >= ENTRENCHED
    ):
        reasons.append(f"entrenched_contradiction:{claim_confidence:.2f}")
    if missing_provenance:
        reasons.append("missing_provenance")
    if ambiguous_claim_match:
        reasons.append("ambiguous_claim_match")
    if weak_witnesses:
        reasons.append("weak_witness")
    # Weak-witness seam: any hygiene flag from normalization (e.g.
    # "possible_base64", "zero_width_stripped") escalates. Flags are advisory
    # signals, never a rejection by themselves.
    for flag in witness_flags:
        reasons.append(f"hygiene_flag:{flag}")
    if unmodeled:
        reasons.append("unmodeled_claim")
    if source_burst_count > BURST_THRESHOLD:
        reasons.append(f"source_burst:{source_burst_count}")

    return EscalationDecision(escalate=bool(reasons), reasons=reasons)


# ---------------------------------------------------------------------------
# Metamorphic reparse
# ---------------------------------------------------------------------------

#: The load-bearing interpretation fields compared across parses.
_DRIFT_FIELDS = ("target_claim", "relation", "effect_direction")


def metamorphic_check(primary: EvidenceIR, variant: EvidenceIR) -> list[str]:
    """Compare the load-bearing interpretation of two parses of one input.

    Any difference in target claim, relation, or effect direction between
    the primary parse and the canonicalized-variant parse is interpretation
    drift -- the extraction is brittle under a meaning-preserving transform
    and must not commit. Returns machine-readable drift reasons.
    """
    drift: list[str] = []
    for field_name in _DRIFT_FIELDS:
        if getattr(primary, field_name).value != getattr(variant, field_name).value:
            drift.append(f"interpretation_drift:{field_name}")
    return drift


_WS_RUN_RE = re.compile(r"\s+")


def canonicalize(text: str) -> str:
    """Deterministic metamorphic variant: collapse whitespace runs.

    Whitespace-collapse only (runs of spaces/tabs/newlines become one space;
    ends stripped). We deliberately do NOT lowercase: lowercasing inside an
    ``<<IR>>`` fixture block would corrupt IDs and enum values, and the goal
    is a meaning-preserving transform. JSON between the markers still parses
    because inter-token whitespace is insignificant in JSON.

    Note: the deterministic FakeCompiler is insensitive to this transform by
    construction (it extracts the same marker block), so fixture reparses are
    stable; a real model may NOT be insensitive to it -- that instability is
    exactly the interpretation drift this lane exists to catch.
    """
    return _WS_RUN_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Pipeline entrypoint (the seam G08 consumes)
# ---------------------------------------------------------------------------

_USAGE_KEYS = ("input_tokens", "output_tokens", "thinking_tokens", "cached_tokens", "total_tokens")

_PROVENANCE_GAP_REASONS = frozenset(
    {
        "provenance_unverified",
        "provenance:experiment_unregistered",
        "provenance:replication_target_missing",
        "provenance:replication_target_unregistered",
        "provenance:replication_target_not_prior",
        "provenance:replication_unverified",
    }
)


def _reject_result(reasons: list[str], t0: float, gemini_calls: int) -> PipelineResult:
    monitor = MonitorResult(
        verdict=Verdict.REJECT,
        reasons=reasons,
        integrity=Integrity.L0_RAW,
        shock=0.0,
    )
    metrics: dict[str, float | int] = {
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
        "gemini_calls": gemini_calls,
        "escalated": 0,
        "cache_hit_fraction": 0.0,
    }
    return PipelineResult(monitor=monitor, engine=None, event_seq=None, metrics=metrics)


def _flatten_usage(metrics: dict[str, float | int], usage: dict[str, Any] | None) -> None:
    """Fold compiler token usage into metrics; compute cache-hit fraction."""
    if not usage:
        metrics["cache_hit_fraction"] = 0.0
        return
    for key in _USAGE_KEYS:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = value
    total = usage.get("total_tokens")
    cached = usage.get("cached_tokens")
    if isinstance(total, (int, float)) and total > 0 and isinstance(cached, (int, float)):
        metrics["cache_hit_fraction"] = cached / total
    else:
        metrics["cache_hit_fraction"] = 0.0


def _merge_usage(*records: dict[str, Any] | None) -> dict[str, float | int] | None:
    """Sum token categories across compile, embed, and escalation calls."""
    merged = {key: 0 for key in _USAGE_KEYS}
    seen = False
    for record in records:
        if not record:
            continue
        for key in _USAGE_KEYS:
            value = record.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] += value
                seen = True
    return merged if seen else None


async def run_pipeline(
    raw_text: str,
    *,
    compiler: Compiler,
    embedder: Embedder,
    monitor: Any,
    state: Any,
    engine: Any,
    claim_index: Any = None,
    evidence_index: Any = None,
    prior_roots: frozenset[str] = frozenset(),
    prior_experiments: frozenset[str] = frozenset(),
    source_burst_count: int = 0,
    escalation_compiler: Compiler | None = None,
    missing_provenance: bool = False,
    ambiguous_claim_match: bool = False,
    weak_witnesses: bool = False,
    raw_bf: float | None = None,
    provenance: Any = None,
    earned_integrity: Integrity | None = None,
) -> PipelineResult:
    """Fast lane + risk-triggered escalation lane. Deterministic control flow.

    Flow: normalize -> (compile || embed, concurrently) -> OOD/duplicate
    signals -> monitor.evaluate -> trigger evaluation -> optional metamorphic
    reparse -> PipelineResult. ``event_seq`` is always None here: G08 owns
    ledger appends. Callers may add trusted-registry ambiguity and
    weak-witness signals. Trusted provenance context, an externally derived
    Bayes factor, and earned integrity are forwarded to monitors that support
    them; omitted values preserve the original G05 call seam.
    """
    t0 = time.perf_counter()
    gemini_calls = 0

    # 1. Hygiene normalization (size limit is a hard deterministic reject).
    try:
        normalized = normalize(raw_text)
    except InputTooLarge:
        return _reject_result(["input_too_large"], t0, gemini_calls)

    # 2. Fast lane parallelism: schedule extraction and embedding
    #    concurrently (spec section 5: "One embedding/OOD lookup in parallel
    #    with extraction").
    gemini_calls += 2  # one compile + one embed attempted
    compile_res, embed_res = await asyncio.gather(
        compiler.compile(raw_text),
        embedder.embed(normalized.text),
        return_exceptions=True,
    )
    if isinstance(compile_res, CompileError):
        return _reject_result([f"compile_error:{compile_res.reason}"], t0, gemini_calls)
    if isinstance(compile_res, Exception):
        return _reject_result(["compile_error:provider_failure"], t0, gemini_calls)
    if isinstance(compile_res, BaseException):
        raise compile_res
    if isinstance(embed_res, Exception):
        return _reject_result(["embedding_error:provider_failure"], t0, gemini_calls)
    if isinstance(embed_res, BaseException):
        raise embed_res
    ir: EvidenceIR = compile_res
    vec: list[float] = embed_res
    usage_records: list[dict[str, Any] | None] = [
        getattr(compiler, "last_usage", None),
        getattr(embedder, "last_usage", None),
    ]

    # 3. OOD / duplicate signals (0.0 when no index is provided).
    ood_distance = claim_index.ood_distance(vec) if claim_index is not None else 0.0
    duplicate_signal = evidence_index.max_similarity(vec) if evidence_index is not None else 0.0

    # 4. Deterministic reference monitor (G05 seam; duck-typed).
    monitor_signals: dict[str, Any] = {
        "ood_distance": ood_distance,
        "duplicate_signal": duplicate_signal,
        "prior_roots": prior_roots,
    }
    # Preserve the original G05 monitor seam for the common case.  The
    # experiment registry context is an optional extension and is only
    # forwarded when a caller actually supplies it.
    # An explicitly empty set is security-relevant: it means no experiment
    # has committed yet and must not fall back to legacy root-only checks.
    monitor_signals["prior_experiments"] = prior_experiments
    if raw_bf is not None:
        monitor_signals["raw_bf"] = raw_bf
    if provenance is not None:
        monitor_signals["provenance"] = provenance
    if earned_integrity is not None:
        monitor_signals["earned_integrity"] = earned_integrity
    evaluation = monitor.evaluate(
        ir,
        normalized.text,
        state,
        **monitor_signals,
    )
    verdict: Verdict = evaluation.monitor["verdict"]
    reasons: list[str] = list(evaluation.monitor["reasons"])

    # 5. Escalation triggers (deterministic policy).
    claim = state.claims.get(ir.target_claim.value) if state is not None else None
    provenance_missing = missing_provenance or any(
        reason in _PROVENANCE_GAP_REASONS for reason in reasons
    )
    decision = evaluate_triggers(
        shock=evaluation.monitor["shock"],
        ood_distance=ood_distance,
        duplicate_signal=duplicate_signal,
        source_trust=engine.trust(ir.source_id),
        claim_confidence=claim.confidence if claim is not None else None,
        relation=str(ir.relation.value),
        witness_flags=list(normalized.flags),
        unmodeled=claim is None,
        source_burst_count=source_burst_count,
        missing_provenance=provenance_missing,
        ambiguous_claim_match=ambiguous_claim_match,
        weak_witnesses=weak_witnesses,
    )

    escalated = 0
    if decision.escalate:
        escalated = 1
        reasons.extend(decision.reasons)
        # Metamorphic reparse: recompile a canonicalized variant and compare
        # the load-bearing interpretation. A failed reparse counts as drift.
        reparse_compiler = (
            escalation_compiler if escalation_compiler is not None else compiler
        )
        gemini_calls += 1
        try:
            variant = await reparse_compiler.compile(canonicalize(raw_text))
            usage_records.append(getattr(reparse_compiler, "last_usage", None))
            drift = metamorphic_check(ir, variant)
        except Exception:
            drift = ["reparse_failed"]
        if drift:
            reasons.extend(drift)
            # Deterministic override: a brittle parse must not commit.
            # Deterministic code chooses the verdict; never the model.
            if verdict in (Verdict.COMMIT, Verdict.PROVISIONAL):
                verdict = Verdict.ESCROW

    # 6. Duplicate-signal bookkeeping: everything that was not rejected
    #    becomes part of the prior-evidence store.
    if verdict != Verdict.REJECT and evidence_index is not None:
        evidence_index.add(vec)

    monitor_result = MonitorResult(
        verdict=verdict,
        reasons=reasons,
        integrity=evaluation.monitor["integrity"],
        shock=evaluation.monitor["shock"],
    )
    metrics: dict[str, float | int] = {
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
        "gemini_calls": gemini_calls,
        "escalated": escalated,
    }
    _flatten_usage(metrics, _merge_usage(*usage_records))

    return PipelineResult(
        monitor=monitor_result,
        engine=evaluation.engine,
        event_seq=None,  # G08 owns ledger appends
        metrics=metrics,
    )
