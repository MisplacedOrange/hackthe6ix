"""Lattice types and EvidenceIR (G02).

The integrity lattice is the core security primitive: all incoming text is
low-integrity by default. Text can never declare its own integrity level --
`EvidenceIR` is a strict model (`extra="forbid"`) with no integrity field, so
a compiled observation cannot self-promote. Promotion happens only in
deterministic code (see core/monitor.py and core/engine.py).
"""

from __future__ import annotations

import math
from enum import IntEnum, StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Sentinel for references that are not (yet) modeled in the graph.
UNMODELED = "UNMODELED"


class Integrity(IntEnum):
    """Earned integrity level. Ordering is meaningful: L0 < L1 < L2 < L3."""

    L0_RAW = 0
    L1_PARSED = 1
    L2_VERIFIED = 2
    L3_REPLICATED = 3


class Relation(StrEnum):
    """Ontology of evidence-to-claim relations."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REPLICATES = "replicates"


class EffectDirection(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NULL = "null"


class Verdict(StrEnum):
    """Deterministic routing decision for a candidate update."""

    COMMIT = "commit"
    PROVISIONAL = "provisional"
    ESCROW = "escrow"
    REJECT = "reject"


class ClaimState(StrEnum):
    """Four-valued claim state."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    CONTESTED = "CONTESTED"
    UNKNOWN = "UNKNOWN"


class Span(BaseModel):
    """A half-open Unicode code-point span in the normalized input text.

    Python string indexing counts Unicode code points, so ``[start:end]`` is
    the canonical witness operation. Strict integers prevent booleans,
    numeric strings, and fractional offsets from being coerced into spans.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Span":
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be > start ({self.start})")
        return self

    def slice(self, text: str) -> str:
        return text[self.start : self.end]

    def in_bounds(self, text: str) -> bool:
        return self.end <= len(text) and len(self.slice(text).strip()) > 0


T = TypeVar("T")


class Witnessed(BaseModel, Generic[T]):
    """A load-bearing value plus the span of normalized input that supports it.

    Witnesses are an audit trail, not a proof: a model can cite a real span
    and still mis-extract. The point is inspectability.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    value: T
    support_span: Span


class EvidenceIR(BaseModel):
    """The only thing a compiler (LLM) may produce.

    Deliberately has NO integrity, confidence, verdict, or policy field:
    text cannot self-promote, and the model never chooses its own verdict.

    The ID fields are retained for serialized compatibility, but they are
    untrusted provenance *proposals* extracted from attacker-controlled text.
    Deterministic pipeline code must resolve them against the trusted
    :class:`EvidenceSubmission` envelope and provenance registry before any
    integrity promotion or committed influence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    #: Untrusted proposal; never authoritative without registry resolution.
    source_id: str = Field(min_length=1)
    #: Untrusted proposal; compare with the trusted submission envelope.
    experiment_id: str = Field(min_length=1)
    #: Untrusted lineage proposal; resolve through the provenance registry.
    root_experiment_id: str = Field(min_length=1)

    target_claim: Witnessed[str]
    relation: Witnessed[Relation]
    effect_direction: Witnessed[EffectDirection]

    effect_size: Witnessed[float] | None = None
    sample_size: Witnessed[int] | None = None

    #: experiment_id this evidence claims to replicate, or None.
    claimed_replication_of: str | None = None

    @model_validator(mode="after")
    def _validate_extracted_values(self) -> "EvidenceIR":
        if not self.target_claim.value.strip():
            raise ValueError("target_claim value must be non-empty")
        if self.effect_size is not None and not math.isfinite(self.effect_size.value):
            raise ValueError("effect_size value must be finite")
        if self.sample_size is not None and (
            type(self.sample_size.value) is not int or self.sample_size.value <= 0
        ):
            raise ValueError("sample_size value must be a strict positive integer")
        return self


class EvidenceSubmission(BaseModel):
    """Trusted ingestion envelope paired with untrusted compiler output.

    ``raw_text`` remains L0 attacker-controlled content. The four metadata
    fields are supplied by the authenticated ingestion boundary, not copied
    from model output. G08 must compare these values with the proposals in an
    ``EvidenceIR`` and use the deterministic provenance registry as the
    authority when they disagree.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    raw_text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    root_experiment_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class Claim(BaseModel):
    """A node in the belief graph. Mutated only by the deterministic reducer."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    supporting: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)

    @property
    def state(self) -> ClaimState:
        has_support = len(self.supporting) > 0
        has_contra = len(self.contradicting) > 0
        if has_support and has_contra:
            return ClaimState.CONTESTED
        if has_support:
            return ClaimState.SUPPORTED
        if has_contra:
            return ClaimState.CONTRADICTED
        return ClaimState.UNKNOWN
