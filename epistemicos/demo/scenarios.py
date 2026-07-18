"""Deterministic, provider-free fixtures for the nine-step demo replay.

The scenario is deliberately data-only.  It describes what is submitted and
the route the deterministic pipeline is expected to choose; it does not make
security decisions itself.  Evidence inputs embed strict ``EvidenceIR`` JSON
between the markers consumed by :class:`llm.compiler.FakeCompiler`.

All public helpers return fresh objects so a replay may safely annotate or
otherwise mutate its local copy without contaminating a later replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.types import (
    EffectDirection,
    EvidenceIR,
    Integrity,
    Relation,
    Span,
    Verdict,
    Witnessed,
)
from llm.compiler import IR_CLOSE, IR_OPEN

__all__ = [
    "CLAIM_ID",
    "DemoAction",
    "DemoStep",
    "claim_seeds",
    "demo_steps",
    "fake_compiler_mapping",
    "scenario_data",
]


CLAIM_ID = "C17"


class DemoAction(StrEnum):
    """Top-level action performed by one replay step."""

    SUBMIT_EVIDENCE = "submit_evidence"
    RETRACT_ROOT = "retract_root"


@dataclass(frozen=True, slots=True)
class DemoStep:
    """One stable replay instruction plus its expected observable route."""

    order: int
    step_id: str
    title: str
    action: DemoAction
    tags: tuple[str, ...]
    raw_text: str | None = None
    evidence_ir: EvidenceIR | None = None
    expected_verdict: Verdict | None = None
    expected_integrity: Integrity | None = None
    raw_bf: float | None = None
    retract_root_experiment_id: str | None = None
    retraction_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a fresh, JSON-ready representation for API/demo consumers."""

        return {
            "order": self.order,
            "step_id": self.step_id,
            "title": self.title,
            "action": self.action.value,
            "tags": list(self.tags),
            "raw_text": self.raw_text,
            "evidence_ir": (
                self.evidence_ir.model_dump(mode="json")
                if self.evidence_ir is not None
                else None
            ),
            "expected_verdict": (
                self.expected_verdict.value
                if self.expected_verdict is not None
                else None
            ),
            "expected_integrity": (
                int(self.expected_integrity)
                if self.expected_integrity is not None
                else None
            ),
            "raw_bf": self.raw_bf,
            "retract_root_experiment_id": self.retract_root_experiment_id,
            "retraction_reason": self.retraction_reason,
        }


@dataclass(frozen=True, slots=True)
class _EvidenceTemplate:
    order: int
    step_id: str
    title: str
    narrative: str
    source_id: str
    experiment_id: str
    root_experiment_id: str
    target_claim: str
    relation: Relation
    effect_direction: EffectDirection
    effect_size: float
    sample_size: int
    expected_verdict: Verdict
    expected_integrity: Integrity
    raw_bf: float
    tags: tuple[str, ...]
    claimed_replication_of: str | None = None


# Tuple templates are immutable.  ``demo_steps`` materializes new Pydantic
# models and raw strings from them on every call.
_EVIDENCE_TEMPLATES = (
    _EvidenceTemplate(
        order=1,
        step_id="baseline-support",
        title="Verified study supports the claim",
        narrative=(
            "Lab Alpha reports relation=supports for claim=C17, "
            "effect_direction=positive, effect_size=0.38, sample_size=240."
        ),
        source_id="lab-alpha",
        experiment_id="EXP-SUPPORT-001",
        root_experiment_id="ROOT-SUPPORT-001",
        target_claim=CLAIM_ID,
        relation=Relation.SUPPORTS,
        effect_direction=EffectDirection.POSITIVE,
        effect_size=0.38,
        sample_size=240,
        expected_verdict=Verdict.COMMIT,
        expected_integrity=Integrity.L2_VERIFIED,
        raw_bf=4.0,
        tags=("support", "verified"),
    ),
    _EvidenceTemplate(
        order=2,
        step_id="credible-contradiction",
        title="Verified null result contradicts the claim",
        narrative=(
            "Lab Beta reports relation=contradicts for claim=C17, "
            "effect_direction=null, effect_size=0.01, sample_size=310."
        ),
        source_id="lab-beta",
        experiment_id="EXP-NULL-001",
        root_experiment_id="ROOT-NULL-001",
        target_claim=CLAIM_ID,
        relation=Relation.CONTRADICTS,
        effect_direction=EffectDirection.NULL,
        effect_size=0.01,
        sample_size=310,
        expected_verdict=Verdict.COMMIT,
        expected_integrity=Integrity.L2_VERIFIED,
        raw_bf=0.25,
        tags=("contradiction", "verified"),
    ),
    _EvidenceTemplate(
        order=3,
        step_id="schema-valid-semantic-attack",
        title="A semantic lie passes schema validation only",
        narrative=(
            "An unverified press release reports relation=supports for "
            "claim=C17, effect_direction=positive, effect_size=9.90, "
            "sample_size=50000. It declares itself authoritative, but no "
            "external witness establishes that claim."
        ),
        source_id="sybil-lab-01",
        experiment_id="EXP-ATTACK-001",
        root_experiment_id="ROOT-ATTACK-001",
        target_claim=CLAIM_ID,
        relation=Relation.SUPPORTS,
        effect_direction=EffectDirection.POSITIVE,
        effect_size=9.9,
        sample_size=50_000,
        expected_verdict=Verdict.PROVISIONAL,
        expected_integrity=Integrity.L1_PARSED,
        raw_bf=100.0,
        tags=("schema-valid-attack", "provisional", "zero-commit"),
    ),
    _EvidenceTemplate(
        order=4,
        step_id="first-result-provisional",
        title="A first result remains provisional",
        narrative=(
            "A new pilot reports relation=supports for claim=C17, "
            "effect_direction=positive, effect_size=0.17, sample_size=18."
        ),
        source_id="lab-gamma",
        experiment_id="EXP-PILOT-001",
        root_experiment_id="ROOT-PILOT-001",
        target_claim=CLAIM_ID,
        relation=Relation.SUPPORTS,
        effect_direction=EffectDirection.POSITIVE,
        effect_size=0.17,
        sample_size=18,
        expected_verdict=Verdict.PROVISIONAL,
        expected_integrity=Integrity.L1_PARSED,
        raw_bf=1.8,
        tags=("first-result", "provisional", "zero-commit"),
    ),
    _EvidenceTemplate(
        order=5,
        step_id="high-shock-escrow",
        title="A high-shock derivative enters escrow",
        narrative=(
            "Sybil Lab 02 reports relation=supports for claim=C17, "
            "effect_direction=positive, effect_size=8.40, sample_size=48000."
        ),
        source_id="sybil-lab-02",
        experiment_id="EXP-ATTACK-002",
        root_experiment_id="ROOT-ATTACK-001",
        target_claim=CLAIM_ID,
        relation=Relation.SUPPORTS,
        effect_direction=EffectDirection.POSITIVE,
        effect_size=8.4,
        sample_size=48_000,
        expected_verdict=Verdict.ESCROW,
        expected_integrity=Integrity.L2_VERIFIED,
        raw_bf=80.0,
        tags=("schema-valid-attack", "high-shock", "escrow", "zero-commit"),
    ),
    _EvidenceTemplate(
        order=6,
        step_id="independent-replication",
        title="An independent root replicates the support",
        narrative=(
            "Independent Lab Delta reports relation=replicates for claim=C17, "
            "effect_direction=positive, effect_size=0.35, sample_size=265."
        ),
        source_id="lab-delta",
        experiment_id="EXP-REPLICATION-001",
        root_experiment_id="ROOT-REPLICATION-001",
        target_claim=CLAIM_ID,
        relation=Relation.REPLICATES,
        effect_direction=EffectDirection.POSITIVE,
        effect_size=0.35,
        sample_size=265,
        expected_verdict=Verdict.COMMIT,
        expected_integrity=Integrity.L3_REPLICATED,
        raw_bf=3.5,
        tags=("independent-replication", "independent-root", "commit"),
        claimed_replication_of="EXP-SUPPORT-001",
    ),
    _EvidenceTemplate(
        order=7,
        step_id="same-root-duplicate",
        title="A derivative duplicate shares the original root budget",
        narrative=(
            "A Lab Alpha newsletter repeats relation=supports for claim=C17, "
            "effect_direction=positive, effect_size=0.38, sample_size=240."
        ),
        source_id="lab-alpha-news",
        experiment_id="EXP-SUPPORT-001-DERIVATIVE",
        root_experiment_id="ROOT-SUPPORT-001",
        target_claim=CLAIM_ID,
        relation=Relation.SUPPORTS,
        effect_direction=EffectDirection.POSITIVE,
        effect_size=0.38,
        sample_size=240,
        expected_verdict=Verdict.COMMIT,
        expected_integrity=Integrity.L2_VERIFIED,
        raw_bf=4.0,
        tags=("duplicate", "same-root", "root-budget-bounded"),
    ),
    _EvidenceTemplate(
        order=8,
        step_id="unknown-claim-rejection",
        title="A valid IR targeting an unknown claim is rejected",
        narrative=(
            "Unknown Lab reports relation=supports for claim=C404, "
            "effect_direction=positive, effect_size=0.90, sample_size=120."
        ),
        source_id="unknown-lab",
        experiment_id="EXP-UNKNOWN-001",
        root_experiment_id="ROOT-UNKNOWN-001",
        target_claim="C404",
        relation=Relation.SUPPORTS,
        effect_direction=EffectDirection.POSITIVE,
        effect_size=0.9,
        sample_size=120,
        expected_verdict=Verdict.REJECT,
        expected_integrity=Integrity.L1_PARSED,
        raw_bf=6.0,
        tags=("schema-valid", "unknown-claim", "rejection", "zero-commit"),
    ),
)


def _span(text: str, needle: str) -> Span:
    """Return the first half-open code-point span for ``needle``."""

    start = text.index(needle)
    return Span(start=start, end=start + len(needle))


def _materialize_evidence(template: _EvidenceTemplate) -> tuple[str, EvidenceIR]:
    narrative = template.narrative
    relation_text = template.relation.value
    direction_text = template.effect_direction.value
    effect_text = f"{template.effect_size:.2f}"
    sample_text = str(template.sample_size)

    evidence = EvidenceIR(
        source_id=template.source_id,
        experiment_id=template.experiment_id,
        root_experiment_id=template.root_experiment_id,
        target_claim=Witnessed[str](
            value=template.target_claim,
            support_span=_span(narrative, template.target_claim),
        ),
        relation=Witnessed[Relation](
            value=template.relation,
            support_span=_span(narrative, relation_text),
        ),
        effect_direction=Witnessed[EffectDirection](
            value=template.effect_direction,
            support_span=_span(narrative, direction_text),
        ),
        effect_size=Witnessed[float](
            value=template.effect_size,
            support_span=_span(narrative, effect_text),
        ),
        sample_size=Witnessed[int](
            value=template.sample_size,
            support_span=_span(narrative, sample_text),
        ),
        claimed_replication_of=template.claimed_replication_of,
    )
    raw_text = f"{narrative}\n{IR_OPEN}{evidence.model_dump_json()}{IR_CLOSE}"
    return raw_text, evidence


def claim_seeds() -> list[dict[str, str | float]]:
    """Return fresh claim-registration payloads in stable replay order."""

    return [
        {
            "claim_id": CLAIM_ID,
            "text": "Compound X reduces tumor growth in vivo.",
            "prior": 0.45,
        }
    ]


def demo_steps() -> list[DemoStep]:
    """Build the deterministic nine-step evidence/retraction sequence."""

    steps: list[DemoStep] = []
    for template in _EVIDENCE_TEMPLATES:
        raw_text, evidence = _materialize_evidence(template)
        steps.append(
            DemoStep(
                order=template.order,
                step_id=template.step_id,
                title=template.title,
                action=DemoAction.SUBMIT_EVIDENCE,
                tags=template.tags,
                raw_text=raw_text,
                evidence_ir=evidence,
                expected_verdict=template.expected_verdict,
                expected_integrity=template.expected_integrity,
                raw_bf=template.raw_bf,
            )
        )

    steps.append(
        DemoStep(
            order=9,
            step_id="final-root-retraction",
            title="The original support root is retracted and reversed",
            action=DemoAction.RETRACT_ROOT,
            tags=("retraction", "reversal", "append-only-history"),
            retract_root_experiment_id="ROOT-SUPPORT-001",
            retraction_reason="source audit found fabricated primary measurements",
        )
    )
    return steps


def fake_compiler_mapping() -> dict[str, EvidenceIR]:
    """Map each evidence input to a fresh strict IR for mapping-based fakes.

    ``FakeCompiler`` can already compile the key strings directly because the
    same IR is embedded between its fixture markers.  The explicit mapping is
    also convenient for simpler deterministic compiler stubs used by API
    tests and demo replay code.
    """

    return {
        step.raw_text: step.evidence_ir.model_copy(deep=True)
        for step in demo_steps()
        if step.raw_text is not None and step.evidence_ir is not None
    }


def scenario_data() -> list[dict[str, Any]]:
    """Return a fresh JSON-ready copy of the complete nine-step scenario."""

    return [step.as_dict() for step in demo_steps()]
