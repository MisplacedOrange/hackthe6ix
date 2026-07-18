"""Focused completion tests for the G05 monitor/shadow handoff."""

from __future__ import annotations

from core.engine import ROOT_BUDGET, Engine
from core.ledger import GraphState
from core.monitor import (
    ProvenanceRegistry,
    ReferenceMonitor,
    TrustedProvenance,
)
from core.shadow import execute_shadow
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
from tests.conftest import SAMPLE_TEXT, make_evidence


def _state() -> GraphState:
    return GraphState(
        claims={"C17": Claim(id="C17", text="Compound X has an effect", confidence=0.5)}
    )


def _registry() -> ProvenanceRegistry:
    registry = ProvenanceRegistry()
    registry.register_experiment(
        "EXP-001",
        source_id="lab-alpha",
        root_experiment_id="ROOT-001",
        claim_ids={"C17"},
    )
    return registry


def test_evaluate_defaults_and_shadow_graph_are_isolated() -> None:
    live_engine = Engine()
    live_state = _state()

    evaluation = ReferenceMonitor(live_engine, _registry()).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        live_state,
    )

    assert evaluation.monitor["verdict"] is Verdict.COMMIT
    assert evaluation.engine is not None
    assert evaluation.shock is evaluation.shock_report
    assert evaluation.shock is not None
    assert evaluation.shock.affected_nodes == 1
    assert live_state.claims["C17"].confidence == 0.5
    assert live_engine.spent_for_root("ROOT-001") == 0.0
    assert evaluation.engine["posterior"] != live_state.claims["C17"].confidence

    shadow = execute_shadow(
        live_engine,
        claim_id="C17",
        root_experiment_id="ROOT-001",
        source_id="lab-alpha",
        prior=0.5,
        raw_bf=0.5,
        integrity=Integrity.L2_VERIFIED,
        relation=Relation.CONTRADICTS,
        ood_distance=0.0,
        duplicate_signal=0.0,
        state=live_state,
    )
    assert shadow.state is not live_state
    assert shadow.state.claims["C17"] is not live_state.claims["C17"]
    assert shadow.state.claims["C17"].confidence == shadow.engine["posterior"]
    assert live_state.claims["C17"].confidence == 0.5


def test_trusted_request_provenance_can_verify_without_registry_mutation() -> None:
    monitor = ReferenceMonitor(Engine())
    trusted = TrustedProvenance(
        source_id="lab-alpha",
        experiment_id="EXP-001",
        root_experiment_id="ROOT-001",
        verified=True,
    )

    evaluation = monitor.evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        _state(),
        raw_bf=0.5,
        provenance=trusted,
        earned_integrity=Integrity.L2_VERIFIED,
    )

    assert evaluation.monitor["integrity"] is Integrity.L2_VERIFIED
    assert evaluation.monitor["verdict"] is Verdict.COMMIT
    assert evaluation.engine is not None
    assert evaluation.engine["raw_bf"] == 0.5


def test_integrity_override_cannot_exceed_external_witness() -> None:
    result = ReferenceMonitor(Engine()).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        _state(),
        provenance=TrustedProvenance(
            source_id="lab-alpha",
            experiment_id="EXP-001",
            root_experiment_id="ROOT-001",
            verified=False,
        ),
        earned_integrity=Integrity.L3_REPLICATED,
    )

    assert result.monitor["verdict"] is Verdict.REJECT
    assert result.monitor["integrity"] is Integrity.L1_PARSED
    assert "integrity:exceeds_provenance_witness" in result.monitor["reasons"]
    assert result.engine is None


def test_exhausted_root_budget_rejects_before_another_shadow_proposal() -> None:
    live_engine = Engine()
    live_engine.propose(
        "C17",
        "ROOT-001",
        "lab-alpha",
        0.5,
        1e100,
        Integrity.L3_REPLICATED,
    )
    assert live_engine.spent_for_root("ROOT-001") == ROOT_BUDGET

    result = ReferenceMonitor(live_engine, _registry()).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        _state(),
    )

    assert result.monitor["verdict"] is Verdict.REJECT
    assert "budget:root_exhausted:ROOT-001" in result.monitor["reasons"]
    assert result.engine is None


def test_known_claim_requires_its_exact_identifier_in_witness_span() -> None:
    evidence = make_evidence().model_copy(
        update={
            "target_claim": Witnessed[str](
                value="C17",
                support_span=Span(start=0, end=12),
            )
        }
    )

    result = ReferenceMonitor(Engine(), _registry()).evaluate(
        evidence,
        SAMPLE_TEXT,
        _state(),
    )

    assert result.monitor["verdict"] is Verdict.REJECT
    assert result.monitor["reasons"] == ["witness:target_claim:value_mismatch"]
    assert result.engine is None


def test_explicit_unmodeled_claim_is_inert_and_provisional() -> None:
    text = "UNMODELED supports positive"
    evidence = EvidenceIR(
        source_id="source",
        experiment_id="UNMODELED",
        root_experiment_id="UNMODELED",
        target_claim=Witnessed[str](
            value="UNMODELED",
            support_span=Span(start=0, end=9),
        ),
        relation=Witnessed[Relation](
            value=Relation.SUPPORTS,
            support_span=Span(start=10, end=18),
        ),
        effect_direction=Witnessed[EffectDirection](
            value=EffectDirection.POSITIVE,
            support_span=Span(start=19, end=27),
        ),
    )

    result = ReferenceMonitor(Engine()).evaluate(evidence, text, GraphState())

    assert result.monitor["verdict"] is Verdict.PROVISIONAL
    assert result.monitor["integrity"] is Integrity.L1_PARSED
    assert result.engine is None
    assert "claim:unmodeled" in result.monitor["reasons"]


def test_external_bayes_factor_must_match_relation_direction() -> None:
    result = ReferenceMonitor(Engine(), _registry()).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        _state(),
        raw_bf=2.0,
    )

    assert result.monitor["verdict"] is Verdict.REJECT
    assert "engine:raw_bf_invalid" in result.monitor["reasons"]
    assert result.engine is None
