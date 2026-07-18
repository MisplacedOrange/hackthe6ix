"""Provider-free contract tests for the deterministic demo scenario."""

from __future__ import annotations

import pytest

from core.types import Relation, Verdict, Witnessed
from demo.scenarios import (
    DemoAction,
    claim_seeds,
    demo_steps,
    fake_compiler_mapping,
    scenario_data,
)
from llm.compiler import FakeCompiler


EXPECTED_STEP_IDS = [
    "baseline-support",
    "credible-contradiction",
    "schema-valid-semantic-attack",
    "first-result-provisional",
    "high-shock-escrow",
    "independent-replication",
    "same-root-duplicate",
    "unknown-claim-rejection",
    "final-root-retraction",
]


def test_scenario_has_exactly_nine_stable_ordered_steps() -> None:
    steps = demo_steps()

    assert len(steps) == 9
    assert [step.order for step in steps] == list(range(1, 10))
    assert [step.step_id for step in steps] == EXPECTED_STEP_IDS
    assert all(step.action is DemoAction.SUBMIT_EVIDENCE for step in steps[:8])
    assert steps[8].action is DemoAction.RETRACT_ROOT


def test_public_helpers_return_fresh_deep_data() -> None:
    first_steps = demo_steps()
    second_steps = demo_steps()
    assert first_steps is not second_steps
    assert all(left is not right for left, right in zip(first_steps, second_steps))
    assert all(
        left.evidence_ir is not right.evidence_ir
        for left, right in zip(first_steps[:8], second_steps[:8])
    )

    first_claims = claim_seeds()
    first_claims[0]["text"] = "mutated"
    assert claim_seeds()[0]["text"] == "Compound X reduces tumor growth in vivo."

    first_data = scenario_data()
    first_data[0]["tags"].append("mutated")
    first_data[0]["evidence_ir"]["source_id"] = "mutated"
    second_data = scenario_data()
    assert "mutated" not in second_data[0]["tags"]
    assert second_data[0]["evidence_ir"]["source_id"] == "lab-alpha"

    first_mapping = fake_compiler_mapping()
    second_mapping = fake_compiler_mapping()
    assert first_mapping is not second_mapping
    assert first_mapping.keys() == second_mapping.keys()
    assert all(
        first_mapping[raw_text] is not second_mapping[raw_text]
        for raw_text in first_mapping
    )


@pytest.mark.asyncio
async def test_all_eight_evidence_inputs_round_trip_through_fake_compiler() -> None:
    compiler = FakeCompiler()
    evidence_steps = demo_steps()[:8]

    assert len(evidence_steps) == 8
    for step in evidence_steps:
        assert step.raw_text is not None
        assert step.evidence_ir is not None
        assert await compiler.compile(step.raw_text) == step.evidence_ir


def test_all_witness_spans_are_in_bounds_nonempty_and_not_reused() -> None:
    for step in demo_steps()[:8]:
        assert step.raw_text is not None
        assert step.evidence_ir is not None
        evidence = step.evidence_ir
        witnessed_values: tuple[Witnessed[object] | None, ...] = (
            evidence.target_claim,
            evidence.relation,
            evidence.effect_direction,
            evidence.effect_size,
            evidence.sample_size,
        )
        witnesses = [witness for witness in witnessed_values if witness is not None]
        coordinates = [
            (witness.support_span.start, witness.support_span.end)
            for witness in witnesses
        ]

        assert all(witness.support_span.in_bounds(step.raw_text) for witness in witnesses)
        assert all(witness.support_span.slice(step.raw_text).strip() for witness in witnesses)
        assert len(coordinates) == len(set(coordinates))


def test_scenario_covers_required_security_routes() -> None:
    steps = demo_steps()
    evidence_steps = steps[:8]
    verdicts = {step.expected_verdict for step in evidence_steps}
    tags = {tag for step in steps for tag in step.tags}
    relations = {
        step.evidence_ir.relation.value
        for step in evidence_steps
        if step.evidence_ir is not None
    }

    assert verdicts == {
        Verdict.COMMIT,
        Verdict.PROVISIONAL,
        Verdict.ESCROW,
        Verdict.REJECT,
    }
    assert {Relation.SUPPORTS, Relation.CONTRADICTS, Relation.REPLICATES} <= relations
    assert {
        "schema-valid-attack",
        "provisional",
        "escrow",
        "independent-replication",
        "duplicate",
        "rejection",
        "retraction",
    } <= tags


def test_same_root_duplicate_links_to_original_lineage() -> None:
    baseline = demo_steps()[0].evidence_ir
    duplicate = demo_steps()[6].evidence_ir
    assert baseline is not None and duplicate is not None

    assert duplicate.root_experiment_id == baseline.root_experiment_id
    assert duplicate.experiment_id != baseline.experiment_id
    assert duplicate.source_id != baseline.source_id


def test_independent_replication_uses_a_distinct_root() -> None:
    baseline = demo_steps()[0].evidence_ir
    replication = demo_steps()[5].evidence_ir
    assert baseline is not None and replication is not None

    assert replication.relation.value is Relation.REPLICATES
    assert replication.claimed_replication_of == baseline.experiment_id
    assert replication.root_experiment_id != baseline.root_experiment_id
    assert replication.source_id != baseline.source_id


def test_final_retraction_targets_original_support_root() -> None:
    steps = demo_steps()
    baseline = steps[0].evidence_ir
    retraction = steps[-1]
    assert baseline is not None

    assert retraction.action is DemoAction.RETRACT_ROOT
    assert retraction.retract_root_experiment_id == baseline.root_experiment_id
    assert retraction.retraction_reason
    assert retraction.raw_text is None
    assert retraction.evidence_ir is None
    assert retraction.expected_verdict is None
