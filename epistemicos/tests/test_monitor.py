"""G05 contracts for deterministic monitoring and shadow execution."""

from __future__ import annotations

import math

import pytest

from core.engine import Engine
from core.ledger import GraphState
from core.monitor import ProvenanceRegistry, ReferenceMonitor
from core.shadow import SHOCK_HIGH, SHOCK_MEDIUM
from core.types import (
    Claim,
    EffectDirection,
    Integrity,
    Relation,
    Span,
    Verdict,
    Witnessed,
)
from tests.conftest import SAMPLE_TEXT, make_evidence


def state(*, confidence: float = 0.5) -> GraphState:
    return GraphState(
        claims={"C17": Claim(id="C17", text="Compound X has an effect", confidence=confidence)}
    )


def verified_registry() -> ProvenanceRegistry:
    registry = ProvenanceRegistry()
    registry.register_experiment(
        "EXP-001",
        source_id="lab-alpha",
        root_experiment_id="ROOT-001",
        claim_ids={"C17"},
    )
    return registry


def test_verified_low_shock_commits_only_in_shadow() -> None:
    live = Engine()
    monitor = ReferenceMonitor(live, verified_registry())

    evaluation = monitor.evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )

    assert evaluation.monitor["verdict"] is Verdict.COMMIT
    assert evaluation.monitor["integrity"] is Integrity.L2_VERIFIED
    assert evaluation.monitor["shock"] < SHOCK_MEDIUM
    assert evaluation.engine is not None
    assert evaluation.engine["raw_bf"] < 1.0
    assert evaluation.engine["bounded_delta"] < 0.0
    assert evaluation.engine["root_spent"] > 0.0
    assert live.spent_for_root("ROOT-001") == 0.0
    assert live.spend_log == []


@pytest.mark.parametrize(
    ("ood_distance", "expected"),
    [(0.4, Verdict.PROVISIONAL), (1.0, Verdict.ESCROW)],
)
def test_medium_and_high_finite_shock_route_deterministically(
    ood_distance: float, expected: Verdict
) -> None:
    evaluation = ReferenceMonitor(Engine(), verified_registry()).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        state(),
        ood_distance=ood_distance,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert evaluation.monitor["verdict"] is expected
    assert math.isfinite(evaluation.monitor["shock"])
    if expected is Verdict.PROVISIONAL:
        assert SHOCK_MEDIUM <= evaluation.monitor["shock"] < SHOCK_HIGH
        assert "shock:medium" in evaluation.monitor["reasons"]
    else:
        assert evaluation.monitor["shock"] >= SHOCK_HIGH
        assert "shock:high" in evaluation.monitor["reasons"]


class FailIfClonedEngine(Engine):
    def clone(self):
        raise AssertionError("invalid evidence reached the engine")


def test_out_of_bounds_witness_rejects_before_engine() -> None:
    evidence = make_evidence().model_copy(
        update={
            "target_claim": Witnessed[str](
                value="C17", support_span=Span(start=0, end=len(SAMPLE_TEXT) + 1)
            )
        }
    )
    evaluation = ReferenceMonitor(FailIfClonedEngine(), verified_registry()).evaluate(
        evidence,
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert evaluation.monitor["verdict"] is Verdict.REJECT
    assert evaluation.engine is None
    assert "witness:target_claim:out_of_bounds" in evaluation.monitor["reasons"]


def test_numeric_value_must_appear_in_its_witness_span() -> None:
    evidence = make_evidence().model_copy(
        update={
            "effect_size": Witnessed[float](
                value=-0.42,
                support_span=Span(start=0, end=12),
            )
        }
    )
    evaluation = ReferenceMonitor(FailIfClonedEngine(), verified_registry()).evaluate(
        evidence,
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert evaluation.monitor["verdict"] is Verdict.REJECT
    assert evaluation.engine is None
    assert "witness:effect_size:numeric_value_missing" in evaluation.monitor["reasons"]


def test_unrelated_fields_cannot_reuse_a_witness_span() -> None:
    evidence = make_evidence()
    reused = evidence.model_copy(
        update={
            "effect_direction": Witnessed[EffectDirection](
                value=EffectDirection.NEGATIVE,
                support_span=evidence.relation.support_span,
            )
        }
    )
    evaluation = ReferenceMonitor(FailIfClonedEngine(), verified_registry()).evaluate(
        reused,
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert evaluation.monitor["verdict"] is Verdict.REJECT
    assert "witness:span_reused:relation:effect_direction" in evaluation.monitor["reasons"]


def test_unknown_claim_and_registered_scope_violation_reject() -> None:
    unknown = make_evidence(claim_id="C404")
    unknown_result = ReferenceMonitor(Engine(), verified_registry()).evaluate(
        unknown,
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert unknown_result.monitor["verdict"] is Verdict.REJECT
    assert "claim:unknown" in unknown_result.monitor["reasons"]

    registry = ProvenanceRegistry()
    registry.register_experiment(
        "EXP-001",
        source_id="lab-alpha",
        root_experiment_id="ROOT-001",
        claim_ids={"C99"},
    )
    scoped = ReferenceMonitor(Engine(), registry).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert scoped.monitor["verdict"] is Verdict.REJECT
    assert "provenance:claim_scope_violation" in scoped.monitor["reasons"]


def test_unregistered_provenance_remains_parsed_and_provisional() -> None:
    live = Engine()
    evaluation = ReferenceMonitor(live, ProvenanceRegistry()).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert evaluation.monitor["integrity"] is Integrity.L1_PARSED
    assert evaluation.monitor["verdict"] is Verdict.PROVISIONAL
    assert "provenance:experiment_unregistered" in evaluation.monitor["reasons"]
    assert evaluation.engine is not None
    assert evaluation.engine["bounded_delta"] == 0.0
    assert live.spend_log == []


def test_registered_mapping_mismatch_rejects_as_spoof() -> None:
    spoofed = make_evidence(root_experiment_id="ROOT-ATTACKER-FRESH")
    evaluation = ReferenceMonitor(FailIfClonedEngine(), verified_registry()).evaluate(
        spoofed,
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert evaluation.monitor["verdict"] is Verdict.REJECT
    assert "provenance:root_mismatch" in evaluation.monitor["reasons"]
    assert evaluation.engine is None


def replication_registry(*, independent: bool = True) -> ProvenanceRegistry:
    registry = ProvenanceRegistry()
    registry.register_experiment(
        "EXP-ORIGINAL",
        source_id="lab-original",
        root_experiment_id="ROOT-ORIGINAL",
        claim_ids={"C17"},
        outcome_relation=Relation.SUPPORTS,
        effect_direction=EffectDirection.POSITIVE,
    )
    registry.register_experiment(
        "EXP-REPLICATION",
        source_id="lab-independent",
        root_experiment_id="ROOT-REPLICATION",
        claim_ids={"C17"},
        replicates_experiment_id="EXP-ORIGINAL",
        independent=independent,
        outcome_relation=Relation.REPLICATES,
        effect_direction=EffectDirection.POSITIVE,
    )
    return registry


def replication_evidence():
    text = (
        "Independent experiment EXP-REPLICATION replicates claim C17 with "
        "positive direction, effect size 0.35, and sample size 265."
    )
    evidence = make_evidence(
        source_id="lab-independent",
        experiment_id="EXP-REPLICATION",
        root_experiment_id="ROOT-REPLICATION",
        relation=Relation.REPLICATES,
        direction=EffectDirection.POSITIVE,
    ).model_copy(
        update={
            "target_claim": Witnessed[str](
                value="C17", support_span=Span(start=text.index("C17"), end=text.index("C17") + 3)
            ),
            "relation": Witnessed[Relation](
                value=Relation.REPLICATES,
                support_span=Span(
                    start=text.index("replicates"),
                    end=text.index("replicates") + len("replicates"),
                ),
            ),
            "effect_direction": Witnessed[EffectDirection](
                value=EffectDirection.POSITIVE,
                support_span=Span(
                    start=text.index("positive"), end=text.index("positive") + len("positive")
                ),
            ),
            "effect_size": Witnessed[float](
                value=0.35,
                support_span=Span(start=text.index("0.35"), end=text.index("0.35") + 4),
            ),
            "sample_size": Witnessed[int](
                value=265,
                support_span=Span(start=text.index("265"), end=text.index("265") + 3),
            ),
            "claimed_replication_of": "EXP-ORIGINAL",
        }
    )
    return text, evidence


def test_registry_confirmed_independent_replication_earns_l3() -> None:
    text, evidence = replication_evidence()
    evaluation = ReferenceMonitor(Engine(), replication_registry()).evaluate(
        evidence,
        text,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset({"ROOT-ORIGINAL"}),
    )
    assert evaluation.monitor["integrity"] is Integrity.L3_REPLICATED
    assert evaluation.monitor["verdict"] is Verdict.COMMIT
    assert evaluation.engine is not None
    assert evaluation.engine["bounded_delta"] > 0.0
    assert "provenance:independent_replication" in evaluation.monitor["reasons"]


@pytest.mark.parametrize("registered", [False, True])
def test_attacker_declared_replication_cannot_self_promote(registered: bool) -> None:
    text, evidence = replication_evidence()
    registry = replication_registry(independent=False) if registered else ProvenanceRegistry()
    evaluation = ReferenceMonitor(Engine(), registry).evaluate(
        evidence,
        text,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset({"ROOT-ORIGINAL"}),
    )
    assert evaluation.monitor["integrity"] is not Integrity.L3_REPLICATED
    assert "provenance:independent_replication" not in evaluation.monitor["reasons"]


@pytest.mark.parametrize(
    ("ood_distance", "duplicate_signal"),
    [(math.nan, 0.0), (math.inf, 0.0), (0.0, math.nan), (0.0, -math.inf), (True, 0.0)],
)
def test_nonfinite_or_boolean_shock_inputs_reject_before_engine(
    ood_distance, duplicate_signal
) -> None:
    evaluation = ReferenceMonitor(FailIfClonedEngine(), verified_registry()).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        state(),
        ood_distance=ood_distance,
        duplicate_signal=duplicate_signal,
        prior_roots=frozenset(),
    )
    assert evaluation.monitor["verdict"] is Verdict.REJECT
    assert evaluation.engine is None
    assert "shock:nonfinite_signal" in evaluation.monitor["reasons"]


class NonfiniteShadowClone:
    def propose(self, *args, **kwargs):
        return {
            "prior": 0.5,
            "raw_bf": 2.0,
            "bounded_delta": math.nan,
            "root_spent": 0.0,
            "posterior": math.nan,
            "integrity": Integrity.L2_VERIFIED,
        }


class NonfiniteShadowEngine:
    def clone(self):
        return NonfiniteShadowClone()


def test_nonfinite_shadow_report_rejects() -> None:
    evaluation = ReferenceMonitor(NonfiniteShadowEngine(), verified_registry()).evaluate(
        make_evidence(),
        SAMPLE_TEXT,
        state(),
        ood_distance=0.0,
        duplicate_signal=0.0,
        prior_roots=frozenset(),
    )
    assert evaluation.monitor["verdict"] is Verdict.REJECT
    assert "shadow:nonfinite" in evaluation.monitor["reasons"]


def test_registry_rejects_conflicting_experiment_redefinition() -> None:
    registry = verified_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register_experiment(
            "EXP-001",
            source_id="attacker",
            root_experiment_id="ROOT-FRESH",
            claim_ids={"C17"},
        )
