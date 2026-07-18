"""Focused completion tests for the G07 pipeline handoff."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.escalate import evaluate_triggers, run_pipeline
from core.engine import Engine
from core.monitor import ReferenceMonitor
from core.types import Claim, EvidenceIR, Integrity, Relation, Verdict
from llm.embedder import EMBED_MODEL_ENV, GeminiEmbedder


class StaticCompiler:
    def __init__(self, evidence: EvidenceIR) -> None:
        self.evidence = evidence

    async def compile(self, raw_text: str) -> EvidenceIR:
        return self.evidence.model_copy(deep=True)


class StaticEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class StaticMonitor:
    def __init__(
        self,
        *,
        reasons: list[str] | None = None,
        integrity: Integrity = Integrity.L2_VERIFIED,
    ) -> None:
        self.reasons = list(reasons or [])
        self.integrity = integrity
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        ir: EvidenceIR,
        normalized_text: str,
        state: Any,
        **signals: Any,
    ) -> SimpleNamespace:
        self.calls.append(dict(signals))
        return SimpleNamespace(
            monitor={
                "verdict": Verdict.COMMIT,
                "reasons": list(self.reasons),
                "integrity": self.integrity,
                "shock": 0.0,
            },
            engine=None,
        )


class TrustedEngine:
    def trust(self, source_id: str) -> float:
        return 0.8


def test_gemini_embedder_configuration_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv(EMBED_MODEL_ENV, "test-embedding-model")

    embedder = GeminiEmbedder()

    assert embedder.model == "test-embedding-model"
    assert embedder._client is None


@pytest.mark.asyncio
async def test_gemini_embedder_uses_async_provider_seam() -> None:
    embed_content = AsyncMock(
        return_value=SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.25, -0.75])]
        )
    )
    embedder = GeminiEmbedder(model="explicit-model")
    embedder._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    )

    result = await embedder.embed("normalized evidence")

    assert result == [0.25, -0.75]
    embed_content.assert_awaited_once_with(
        model="explicit-model",
        contents="normalized evidence",
    )


def test_named_spec_triggers_have_distinct_reasons() -> None:
    decision = evaluate_triggers(
        shock=0.0,
        ood_distance=0.0,
        duplicate_signal=0.0,
        source_trust=0.8,
        claim_confidence=0.5,
        relation=Relation.SUPPORTS.value,
        witness_flags=[],
        unmodeled=False,
        source_burst_count=0,
        missing_provenance=True,
        ambiguous_claim_match=True,
        weak_witnesses=True,
    )

    assert decision.escalate is True
    assert decision.reasons == [
        "missing_provenance",
        "ambiguous_claim_match",
        "weak_witness",
    ]


@pytest.mark.asyncio
async def test_pipeline_forwards_optional_trusted_monitor_context(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    claim_id = sample_evidence.target_claim.value
    state = SimpleNamespace(
        claims={claim_id: Claim(id=claim_id, text="modeled claim", confidence=0.5)}
    )
    provenance = object()
    monitor = StaticMonitor()

    await run_pipeline(
        sample_text,
        compiler=StaticCompiler(sample_evidence),
        embedder=StaticEmbedder(),
        monitor=monitor,
        state=state,
        engine=TrustedEngine(),
        raw_bf=1.25,
        provenance=provenance,
        earned_integrity=Integrity.L2_VERIFIED,
    )

    assert monitor.calls == [
        {
            "ood_distance": 0.0,
            "duplicate_signal": 0.0,
            "prior_roots": frozenset(),
            "prior_experiments": frozenset(),
            "raw_bf": 1.25,
            "provenance": provenance,
            "earned_integrity": Integrity.L2_VERIFIED,
        }
    ]


@pytest.mark.asyncio
async def test_pipeline_surfaces_inferred_and_explicit_escalation_reasons(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    evidence = sample_evidence
    claim_id = evidence.target_claim.value
    state = SimpleNamespace(
        claims={claim_id: Claim(id=claim_id, text="modeled claim", confidence=0.5)}
    )

    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(evidence),
        embedder=StaticEmbedder(),
        monitor=StaticMonitor(),
        state=state,
        engine=TrustedEngine(),
        missing_provenance=True,
        ambiguous_claim_match=True,
        weak_witnesses=True,
    )

    assert result["metrics"]["escalated"] == 1
    assert result["monitor"]["verdict"] is Verdict.COMMIT
    assert result["monitor"]["reasons"] == [
        "missing_provenance",
        "ambiguous_claim_match",
        "weak_witness",
    ]


@pytest.mark.asyncio
async def test_pipeline_infers_missing_provenance_from_monitor_reason(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    claim_id = sample_evidence.target_claim.value
    state = SimpleNamespace(
        claims={claim_id: Claim(id=claim_id, text="modeled claim", confidence=0.5)}
    )
    monitor = StaticMonitor(
        reasons=["provenance_unverified", "integrity:parsed_only"],
        integrity=Integrity.L1_PARSED,
    )

    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(sample_evidence),
        embedder=StaticEmbedder(),
        monitor=monitor,
        state=state,
        engine=TrustedEngine(),
    )

    assert result["metrics"]["escalated"] == 1
    assert result["monitor"]["reasons"] == [
        "provenance_unverified",
        "integrity:parsed_only",
        "missing_provenance",
    ]


@pytest.mark.asyncio
async def test_pipeline_integrates_with_reference_monitor(
    sample_text: str,
    sample_evidence: EvidenceIR,
) -> None:
    claim_id = sample_evidence.target_claim.value
    state = SimpleNamespace(
        claims={claim_id: Claim(id=claim_id, text="modeled claim", confidence=0.5)}
    )
    engine = Engine()

    result = await run_pipeline(
        sample_text,
        compiler=StaticCompiler(sample_evidence),
        embedder=StaticEmbedder(),
        monitor=ReferenceMonitor(engine),
        state=state,
        engine=engine,
    )

    assert result["monitor"]["verdict"] is Verdict.PROVISIONAL
    assert "missing_provenance" in result["monitor"]["reasons"]
    assert any(reason.startswith("low_trust:") for reason in result["monitor"]["reasons"])
    assert result["metrics"]["escalated"] == 1
