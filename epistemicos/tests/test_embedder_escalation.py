"""Provider-free contract tests for G07 embedding and escalation seams."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from core.escalate import (
    BURST_THRESHOLD,
    DUP_THRESHOLD,
    ENTRENCHED,
    LOW_TRUST,
    OOD_THRESHOLD,
    SHOCK_HIGH,
    canonicalize,
    evaluate_triggers,
    metamorphic_check,
    run_pipeline,
)
from core.types import (
    Claim,
    EffectDirection,
    EvidenceIR,
    Integrity,
    Relation,
    Verdict,
)
from llm.compiler import CompileError
from llm.embedder import ClaimIndex, EvidenceIndex, FakeEmbedder, cosine


class StaticCompiler:
    def __init__(
        self,
        evidence: EvidenceIR | None = None,
        *,
        error: BaseException | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.evidence = evidence
        self.error = error
        self.calls: list[str] = []
        self.last_usage = usage

    async def compile(self, raw_text: str) -> EvidenceIR:
        self.calls.append(raw_text)
        if self.error is not None:
            raise self.error
        assert self.evidence is not None
        return self.evidence.model_copy(deep=True)


class StaticEmbedder:
    def __init__(
        self,
        vector: list[float] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.vector = vector or [1.0, 0.0]
        self.error = error
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return list(self.vector)


@dataclass
class MonitorEvaluation:
    monitor: dict[str, Any]
    engine: dict[str, Any] | None


class RecordingMonitor:
    def __init__(
        self,
        *,
        verdict: Verdict = Verdict.COMMIT,
        shock: float = 0.0,
        reasons: list[str] | None = None,
    ) -> None:
        self.verdict = verdict
        self.shock = shock
        self.reasons = reasons or []
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        ir: EvidenceIR,
        normalized_text: str,
        state: Any,
        **signals: Any,
    ) -> MonitorEvaluation:
        self.calls.append(
            {
                "ir": ir,
                "normalized_text": normalized_text,
                "state": state,
                **signals,
            }
        )
        return MonitorEvaluation(
            monitor={
                "verdict": self.verdict,
                "reasons": list(self.reasons),
                "integrity": Integrity.L2_VERIFIED,
                "shock": self.shock,
            },
            engine={
                "prior": 0.5,
                "raw_bf": 2.0,
                "bounded_delta": 0.1,
                "root_spent": 0.01,
                "posterior": 0.6,
                "integrity": Integrity.L2_VERIFIED,
            },
        )


class TrustEngine:
    def __init__(self, trust: float = 0.8) -> None:
        self.value = trust

    def trust(self, source_id: str) -> float:
        return self.value


def state_for(evidence: EvidenceIR, confidence: float = 0.5) -> SimpleNamespace:
    claim_id = evidence.target_claim.value
    return SimpleNamespace(
        claims={claim_id: Claim(id=claim_id, text="modeled claim", confidence=confidence)}
    )


def variant_with(
    evidence: EvidenceIR,
    field_name: str,
    value: str | Relation | EffectDirection,
) -> EvidenceIR:
    witnessed = getattr(evidence, field_name).model_copy(update={"value": value})
    return evidence.model_copy(update={field_name: witnessed})


# -- embedder geometry ------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_embedder_is_deterministic_unit_length_and_copy_safe() -> None:
    embedder = FakeEmbedder(dim=12, canned={"fixed": [3.0, 4.0]})

    first = await embedder.embed("same text")
    second = await embedder.embed("same text")
    different = await embedder.embed("different text")
    canned = await embedder.embed("fixed")
    canned.append(99.0)

    assert first == second
    assert first != different
    assert len(first) == 12
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert await embedder.embed("fixed") == [3.0, 4.0]


def test_fake_embedder_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dim must be >= 1"):
        FakeEmbedder(dim=0)


def test_cosine_handles_standard_zero_and_nonfinite_vectors() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([math.nan, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])


@pytest.mark.asyncio
async def test_claim_index_reports_nearest_claim_and_ood_distance() -> None:
    embedder = FakeEmbedder(
        dim=2,
        canned={"alpha": [1.0, 0.0], "beta": [0.0, 1.0]},
    )
    index = await ClaimIndex.build({"C1": "alpha", "C2": "beta"}, embedder)
    query = [0.8, 0.2]

    claim_id, similarity = index.nearest(query)
    assert claim_id == "C1"
    assert similarity == pytest.approx(cosine(query, [1.0, 0.0]))
    assert index.ood_distance(query) == pytest.approx(1.0 - similarity)
    assert index.nearest([0.0, 0.0]) == ("C1", 0.0)
    assert index.ood_distance([0.0, 0.0]) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_empty_claim_index_is_fully_ood() -> None:
    index = await ClaimIndex.build({}, FakeEmbedder(dim=2))

    assert index.nearest([1.0, 0.0]) == (None, 0.0)
    assert index.ood_distance([1.0, 0.0]) == pytest.approx(1.0)


def test_claim_index_rejects_query_dimension_mismatch() -> None:
    index = ClaimIndex(["C1"], np.asarray([[1.0, 0.0]]))

    with pytest.raises(ValueError):
        index.nearest([1.0])


def test_evidence_index_detects_duplicates_and_handles_zero() -> None:
    index = EvidenceIndex()
    assert len(index) == 0
    assert index.max_similarity([1.0, 0.0]) == 0.0

    index.add([3.0, 0.0])
    assert len(index) == 1
    assert index.max_similarity([1.0, 0.0]) == pytest.approx(1.0)
    assert index.max_similarity([0.0, 1.0]) == pytest.approx(0.0)
    assert index.max_similarity([0.0, 0.0]) == 0.0


# -- deterministic escalation policy --------------------------------------


def test_every_escalation_trigger_emits_a_machine_readable_reason() -> None:
    decision = evaluate_triggers(
        shock=SHOCK_HIGH,
        ood_distance=OOD_THRESHOLD,
        duplicate_signal=DUP_THRESHOLD,
        source_trust=math.nextafter(LOW_TRUST, 0.0),
        claim_confidence=ENTRENCHED,
        relation=Relation.CONTRADICTS.value,
        witness_flags=["zero_width_stripped", "possible_base64"],
        unmodeled=True,
        source_burst_count=BURST_THRESHOLD + 1,
    )

    assert decision.escalate is True
    assert {reason.split(":", 1)[0] for reason in decision.reasons} == {
        "high_shock",
        "near_ood",
        "duplicate",
        "low_trust",
        "entrenched_contradiction",
        "hygiene_flag",
        "unmodeled_claim",
        "source_burst",
    }
    assert decision.reasons.count("hygiene_flag:zero_width_stripped") == 1
    assert decision.reasons.count("hygiene_flag:possible_base64") == 1


def test_trigger_boundaries_below_policy_do_not_escalate() -> None:
    decision = evaluate_triggers(
        shock=math.nextafter(SHOCK_HIGH, 0.0),
        ood_distance=math.nextafter(OOD_THRESHOLD, 0.0),
        duplicate_signal=math.nextafter(DUP_THRESHOLD, 0.0),
        source_trust=LOW_TRUST,
        claim_confidence=math.nextafter(ENTRENCHED, 0.0),
        relation=Relation.CONTRADICTS.value,
        witness_flags=[],
        unmodeled=False,
        source_burst_count=BURST_THRESHOLD,
    )

    assert decision.escalate is False
    assert decision.reasons == []


def test_canonicalization_is_deterministic_idempotent_and_whitespace_only() -> None:
    raw = "  Alpha\t beta\r\n  <<IR>> { \"C17\": 1 } <</IR>>  "
    expected = 'Alpha beta <<IR>> { "C17": 1 } <</IR>>'

    assert canonicalize(raw) == expected
    assert canonicalize(canonicalize(raw)) == expected
    assert "C17" in canonicalize(raw)


@pytest.mark.parametrize(
    ("field_name", "changed_value", "reason"),
    [
        ("target_claim", "C404", "interpretation_drift:target_claim"),
        ("relation", Relation.SUPPORTS, "interpretation_drift:relation"),
        (
            "effect_direction",
            EffectDirection.POSITIVE,
            "interpretation_drift:effect_direction",
        ),
    ],
)
def test_metamorphic_check_detects_each_load_bearing_drift(
    sample_evidence: EvidenceIR,
    field_name: str,
    changed_value: str | Relation | EffectDirection,
    reason: str,
) -> None:
    variant = variant_with(sample_evidence, field_name, changed_value)

    assert metamorphic_check(sample_evidence, variant) == [reason]
    assert metamorphic_check(sample_evidence, sample_evidence.model_copy(deep=True)) == []


# -- concrete run_pipeline seam -------------------------------------------


@pytest.mark.asyncio
async def test_compile_and_embed_are_started_concurrently(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    arrivals: set[str] = set()
    both_started = asyncio.Event()

    async def arrive(name: str) -> None:
        arrivals.add(name)
        if len(arrivals) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.5)

    class ConcurrentCompiler(StaticCompiler):
        async def compile(self, raw_text: str) -> EvidenceIR:
            await arrive("compile")
            return await super().compile(raw_text)

    class ConcurrentEmbedder(StaticEmbedder):
        async def embed(self, text: str) -> list[float]:
            await arrive("embed")
            return await super().embed(text)

    result = await run_pipeline(
        sample_text,
        compiler=ConcurrentCompiler(sample_evidence),
        embedder=ConcurrentEmbedder(),
        monitor=RecordingMonitor(),
        state=state_for(sample_evidence),
        engine=TrustEngine(),
    )

    assert arrivals == {"compile", "embed"}
    assert result["monitor"]["verdict"] is Verdict.COMMIT


@pytest.mark.asyncio
async def test_compile_error_rejects_without_calling_monitor(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    monitor = RecordingMonitor()
    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(error=CompileError("strict schema failed")),
        embedder=StaticEmbedder(),
        monitor=monitor,
        state=state_for(sample_evidence),
        engine=TrustEngine(),
    )

    assert result["monitor"]["verdict"] is Verdict.REJECT
    assert result["monitor"]["integrity"] is Integrity.L0_RAW
    assert result["monitor"]["reasons"] == ["compile_error:strict schema failed"]
    assert result["engine"] is None
    assert result["event_seq"] is None
    assert result["metrics"]["gemini_calls"] == 2
    assert monitor.calls == []


@pytest.mark.asyncio
async def test_unexpected_compiler_provider_error_fails_closed(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(error=RuntimeError("provider unavailable")),
        embedder=StaticEmbedder(),
        monitor=RecordingMonitor(),
        state=state_for(sample_evidence),
        engine=TrustEngine(),
    )

    assert result["monitor"]["verdict"] is Verdict.REJECT
    assert result["monitor"]["integrity"] is Integrity.L0_RAW
    assert result["engine"] is None


@pytest.mark.asyncio
async def test_embedding_provider_error_fails_closed(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(sample_evidence),
        embedder=StaticEmbedder(error=RuntimeError("embedding provider unavailable")),
        monitor=RecordingMonitor(),
        state=state_for(sample_evidence),
        engine=TrustEngine(),
    )

    assert result["monitor"]["verdict"] is Verdict.REJECT
    assert result["monitor"]["integrity"] is Integrity.L0_RAW
    assert result["engine"] is None


@pytest.mark.asyncio
async def test_metrics_have_stable_shape_and_flatten_usage(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "thinking_tokens": 5,
        "cached_tokens": 25,
        "total_tokens": 125,
    }
    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(sample_evidence, usage=usage),
        embedder=StaticEmbedder(),
        monitor=RecordingMonitor(),
        state=state_for(sample_evidence),
        engine=TrustEngine(),
    )

    metrics = result["metrics"]
    assert set(metrics) == {
        "latency_ms",
        "gemini_calls",
        "escalated",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
        "cached_tokens",
        "total_tokens",
        "cache_hit_fraction",
    }
    assert isinstance(metrics["latency_ms"], float)
    assert metrics["latency_ms"] >= 0.0
    assert metrics["gemini_calls"] == 2
    assert metrics["escalated"] == 0
    assert metrics["cache_hit_fraction"] == pytest.approx(0.2)
    assert result["event_seq"] is None


@pytest.mark.asyncio
async def test_interpretation_drift_forces_commit_to_escrow(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    drifted = variant_with(sample_evidence, "target_claim", "C404")
    evidence_index = EvidenceIndex()
    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(sample_evidence),
        escalation_compiler=StaticCompiler(drifted),
        embedder=StaticEmbedder(),
        monitor=RecordingMonitor(verdict=Verdict.COMMIT),
        state=state_for(sample_evidence),
        engine=TrustEngine(trust=LOW_TRUST - 0.1),
        evidence_index=evidence_index,
    )

    assert result["monitor"]["verdict"] is Verdict.ESCROW
    assert "interpretation_drift:target_claim" in result["monitor"]["reasons"]
    assert any(reason.startswith("low_trust:") for reason in result["monitor"]["reasons"])
    assert result["metrics"]["escalated"] == 1
    assert result["metrics"]["gemini_calls"] == 3
    assert len(evidence_index) == 1


@pytest.mark.asyncio
async def test_failed_metamorphic_compile_forces_commit_to_escrow(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(sample_evidence),
        escalation_compiler=StaticCompiler(error=CompileError("variant invalid")),
        embedder=StaticEmbedder(),
        monitor=RecordingMonitor(verdict=Verdict.COMMIT, shock=SHOCK_HIGH),
        state=state_for(sample_evidence),
        engine=TrustEngine(),
    )

    assert result["monitor"]["verdict"] is Verdict.ESCROW
    assert "reparse_failed" in result["monitor"]["reasons"]


@pytest.mark.asyncio
async def test_escalation_provider_error_forces_escrow(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(sample_evidence),
        escalation_compiler=StaticCompiler(error=RuntimeError("provider unavailable")),
        embedder=StaticEmbedder(),
        monitor=RecordingMonitor(verdict=Verdict.COMMIT, shock=SHOCK_HIGH),
        state=state_for(sample_evidence),
        engine=TrustEngine(),
    )

    assert result["monitor"]["verdict"] is Verdict.ESCROW
    assert "reparse_failed" in result["monitor"]["reasons"]
