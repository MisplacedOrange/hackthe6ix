"""Shared coordination interfaces (§9 of EVIDENCE_FIREWALL_V2.md).

These names are the coordination seam between goals. Keep them stable; if a
change is required, update the spec's §9 first and record it in the goal's
handoff note.

Deterministic code owns pipeline control, schema acceptance, witness
verification, integrity promotion, confidence arithmetic, invariants, shock
scoring, provenance budgets, trust ledgers, cascade caps, commit/escrow/
reject, append-only logging, and reversal. No model may assist those
decisions on the hot path.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

from core.types import EvidenceIR, Integrity, Verdict


class Compiler(Protocol):
    """Compiles untrusted raw text into a typed EvidenceIR (L0 -> L1 seam).

    Implementations: llm/compiler.py (Gemini) and a deterministic fake for
    tests. Output is ONLY EvidenceIR; validation errors must surface as
    rejected/L0 events, never be retried with a looser schema.
    """

    async def compile(self, raw_text: str) -> EvidenceIR: ...


class Embedder(Protocol):
    """Embeds normalized text for OOD / duplicate detection (G07)."""

    async def embed(self, text: str) -> list[float]: ...


class MonitorResult(TypedDict):
    verdict: Verdict
    reasons: list[str]
    integrity: Integrity
    shock: float


class EngineBreakdown(TypedDict):
    prior: float
    raw_bf: float
    bounded_delta: float
    root_spent: float
    posterior: float
    integrity: Integrity


class PipelineResult(TypedDict):
    monitor: MonitorResult
    engine: EngineBreakdown | None
    event_seq: int | None
    metrics: dict[str, float | int]
