"""Regression tests for experiment-specific and outcome-bound L3 promotion."""

from __future__ import annotations

from core.engine import Engine
from core.ledger import GraphState
from core.monitor import ProvenanceRegistry, ReferenceMonitor
from core.types import (
    Claim,
    EffectDirection,
    EvidenceIR,
    Integrity,
    Relation,
    Span,
    Verdict,
    Witnessed,
)


def _span(text: str, value: str) -> Span:
    start = text.index(value)
    return Span(start=start, end=start + len(value))


def _replication(direction: EffectDirection = EffectDirection.POSITIVE) -> tuple[str, EvidenceIR]:
    text = f"Independent study replicates claim C17 with {direction.value} direction."
    return text, EvidenceIR(
        source_id="lab-replication",
        experiment_id="EXP-REPLICATION",
        root_experiment_id="ROOT-REPLICATION",
        target_claim=Witnessed[str](
            value="C17",
            support_span=_span(text, "C17"),
        ),
        relation=Witnessed[Relation](
            value=Relation.REPLICATES,
            support_span=_span(text, "replicates"),
        ),
        effect_direction=Witnessed[EffectDirection](
            value=direction,
            support_span=_span(text, direction.value),
        ),
        claimed_replication_of="EXP-TARGET",
    )


def _state() -> GraphState:
    return GraphState(
        claims={"C17": Claim(id="C17", text="Compound X has an effect", confidence=0.5)}
    )


def _registry(
    *,
    target_direction: EffectDirection | None = EffectDirection.POSITIVE,
    replication_direction: EffectDirection | None = EffectDirection.POSITIVE,
    replication_relation: Relation | None = Relation.REPLICATES,
) -> ProvenanceRegistry:
    registry = ProvenanceRegistry()
    registry.register_experiment(
        "EXP-TARGET",
        source_id="lab-target",
        root_experiment_id="ROOT-TARGET",
        claim_ids={"C17"},
        outcome_relation=(Relation.SUPPORTS if target_direction is not None else None),
        effect_direction=target_direction,
    )
    registry.register_experiment(
        "EXP-REPLICATION",
        source_id="lab-replication",
        root_experiment_id="ROOT-REPLICATION",
        claim_ids={"C17"},
        replicates_experiment_id="EXP-TARGET",
        independent=True,
        outcome_relation=replication_relation,
        effect_direction=replication_direction,
    )
    return registry


def test_same_root_derivative_does_not_unlock_uncommitted_target_experiment() -> None:
    text, evidence = _replication()

    result = ReferenceMonitor(Engine(), _registry()).evaluate(
        evidence,
        text,
        _state(),
        prior_roots=frozenset({"ROOT-TARGET"}),
        # A derivative from the target root committed, but EXP-TARGET did not.
        prior_experiments=frozenset({"EXP-TARGET-DERIVATIVE"}),
    )

    assert result.monitor["integrity"] is Integrity.L2_VERIFIED
    assert "provenance:replication_target_not_prior" in result.monitor["reasons"]
    assert "provenance:independent_replication" not in result.monitor["reasons"]


def test_specific_committed_target_and_consistent_outcome_unlock_l3() -> None:
    text, evidence = _replication()

    result = ReferenceMonitor(Engine(), _registry()).evaluate(
        evidence,
        text,
        _state(),
        prior_roots=frozenset({"ROOT-TARGET"}),
        prior_experiments=frozenset({"EXP-TARGET"}),
    )

    assert result.monitor["integrity"] is Integrity.L3_REPLICATED
    assert result.monitor["verdict"] is Verdict.COMMIT
    assert "provenance:independent_replication" in result.monitor["reasons"]


def test_missing_trusted_outcome_metadata_conservatively_caps_at_l2() -> None:
    text, evidence = _replication()

    result = ReferenceMonitor(
        Engine(),
        _registry(target_direction=None, replication_direction=None, replication_relation=None),
    ).evaluate(
        evidence,
        text,
        _state(),
        prior_roots=frozenset({"ROOT-TARGET"}),
        prior_experiments=frozenset({"EXP-TARGET"}),
    )

    assert result.monitor["integrity"] is Integrity.L2_VERIFIED
    assert "provenance:replication_outcome_unverified" in result.monitor["reasons"]


def test_negative_replication_of_positive_target_cannot_earn_l3() -> None:
    text, evidence = _replication(EffectDirection.NEGATIVE)

    result = ReferenceMonitor(
        Engine(),
        _registry(replication_direction=EffectDirection.NEGATIVE),
    ).evaluate(
        evidence,
        text,
        _state(),
        prior_roots=frozenset({"ROOT-TARGET"}),
        prior_experiments=frozenset({"EXP-TARGET"}),
    )

    assert result.monitor["integrity"] is Integrity.L2_VERIFIED
    assert "provenance:replication_outcome_mismatch" in result.monitor["reasons"]
    assert "provenance:independent_replication" not in result.monitor["reasons"]


def test_replication_relation_must_match_trusted_current_outcome() -> None:
    text, evidence = _replication()

    result = ReferenceMonitor(
        Engine(),
        _registry(replication_relation=Relation.SUPPORTS),
    ).evaluate(
        evidence,
        text,
        _state(),
        prior_roots=frozenset({"ROOT-TARGET"}),
        prior_experiments=frozenset({"EXP-TARGET"}),
    )

    assert result.monitor["verdict"] is Verdict.REJECT
    assert result.monitor["reasons"] == ["provenance:relation_mismatch"]
    assert result.engine is None
