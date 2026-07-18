"""Shared fixtures. All fixtures here are provider-free and deterministic."""

from __future__ import annotations

import pytest

from core.types import (
    EffectDirection,
    EvidenceIR,
    Relation,
    Span,
    Witnessed,
)

#: Canonical normalized text the sample spans below point into.
SAMPLE_TEXT = (
    "We conducted a randomized controlled trial (n=482) of compound X. "
    "The treatment group showed a significant reduction in tumor growth "
    "relative to control, contradicting claim C17 that compound X has no "
    "in-vivo effect. Effect size was -0.42 (95% CI -0.61 to -0.23)."
)


def make_evidence(
    *,
    source_id: str = "lab-alpha",
    experiment_id: str = "EXP-001",
    root_experiment_id: str = "ROOT-001",
    claim_id: str = "C17",
    relation: Relation = Relation.CONTRADICTS,
    direction: EffectDirection = EffectDirection.NEGATIVE,
) -> EvidenceIR:
    """Hand-written, schema-valid EvidenceIR pointing at SAMPLE_TEXT spans."""
    return EvidenceIR(
        source_id=source_id,
        experiment_id=experiment_id,
        root_experiment_id=root_experiment_id,
        target_claim=Witnessed[str](value=claim_id, support_span=Span(start=168, end=177)),
        relation=Witnessed[Relation](value=relation, support_span=Span(start=154, end=215)),
        effect_direction=Witnessed[EffectDirection](
            value=direction, support_span=Span(start=95, end=132)
        ),
        effect_size=Witnessed[float](value=-0.42, support_span=Span(start=217, end=262)),
        sample_size=Witnessed[int](value=482, support_span=Span(start=44, end=49)),
    )


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture
def sample_evidence() -> EvidenceIR:
    return make_evidence()
