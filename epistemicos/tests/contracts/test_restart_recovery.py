"""File-backed EvidenceFirewall restart-recovery contracts.

These tests intentionally exercise the service twice against one SQLite
ledger. Durable reconstruction relies on the ingestion payload contract:
``evidence_id`` is the idempotency key, committed events carry non-negative
``root_cost`` and ``root_experiment_id``, every ingestion event carries its
``source_id``, and RETRACTION carries the durable root tombstone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.main import EvidenceFirewall
from core.engine import ROOT_BUDGET
from core.escalate import BURST_THRESHOLD
from core.ledger import EventType
from core.types import EvidenceSubmission, Verdict
from demo.scenarios import demo_steps


def _submission(idempotency_key: str, *, step_index: int = 0) -> EvidenceSubmission:
    step = demo_steps()[step_index]
    assert step.raw_text is not None and step.evidence_ir is not None
    ir = step.evidence_ir
    return EvidenceSubmission(
        raw_text=step.raw_text,
        source_id=ir.source_id,
        experiment_id=ir.experiment_id,
        root_experiment_id=ir.root_experiment_id,
        idempotency_key=idempotency_key,
    )


def _committed_root_cost(firewall: EvidenceFirewall, root_id: str) -> float:
    return sum(
        float(event.payload["root_cost"])
        for event in firewall.ledger.events()
        if event.event_type is EventType.EVIDENCE_COMMITTED
        and event.payload.get("root_experiment_id") == root_id
    )


@pytest.mark.asyncio
async def test_same_root_budget_remains_cumulative_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "root-budget.db"
    first = EvidenceFirewall(db_path=db_path, seed_demo=True)
    try:
        initial = await first.submit(_submission("restart-root-01"))
        assert initial["monitor"]["verdict"] == Verdict.COMMIT.value
        root_id = _submission("unused").root_experiment_id
        spent_before = first.engine.spent_for_root(root_id)
        assert 0.0 < spent_before <= ROOT_BUDGET
    finally:
        first.close()

    restarted = EvidenceFirewall(db_path=db_path, seed_demo=True)
    try:
        assert restarted.engine.spent_for_root(root_id) == pytest.approx(spent_before)
        for index in range(2, 9):
            await restarted.submit(_submission(f"restart-root-{index:02d}"))
            assert restarted.engine.spent_for_root(root_id) <= ROOT_BUDGET + 1e-12

        assert restarted.engine.spent_for_root(root_id) == pytest.approx(ROOT_BUDGET)
        assert _committed_root_cost(restarted, root_id) <= ROOT_BUDGET + 1e-12
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_committed_idempotency_replay_does_not_append_after_restart(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "idempotency.db"
    submission = _submission("restart-idempotent-commit")
    first = EvidenceFirewall(db_path=db_path, seed_demo=True)
    try:
        original = await first.submit(submission)
        assert original["monitor"]["verdict"] == Verdict.COMMIT.value
        original_event_count = len(first.ledger.events())
    finally:
        first.close()

    restarted = EvidenceFirewall(db_path=db_path, seed_demo=True)
    try:
        before = len(restarted.ledger.events())
        replay = await restarted.submit(submission)
        after = len(restarted.ledger.events())

        assert before == original_event_count
        assert replay == original
        assert after == before
        assert sum(
            event.payload.get("evidence_id") == submission.idempotency_key
            for event in restarted.ledger.events()
        ) == 1
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_retracted_root_remains_rejected_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "retracted-root.db"
    original = _submission("restart-retract-original")
    first = EvidenceFirewall(db_path=db_path, seed_demo=True)
    try:
        committed = await first.submit(original)
        assert committed["monitor"]["verdict"] == Verdict.COMMIT.value
        retraction = await first.retract(original.experiment_id)
        assert retraction["root_experiment_id"] == original.root_experiment_id
        assert retraction["reversed_events"] == 1
    finally:
        first.close()

    restarted = EvidenceFirewall(db_path=db_path, seed_demo=True)
    try:
        assert original.root_experiment_id in restarted._retracted_roots
        assert restarted.engine.spent_for_root(original.root_experiment_id) == 0.0

        retry = original.model_copy(
            update={"idempotency_key": "restart-retract-new-request"}
        )
        result = await restarted.submit(retry)
        assert result["monitor"]["verdict"] == Verdict.REJECT.value
        assert "provenance:root_retracted" in result["monitor"]["reasons"]
        assert result["engine"] is None
        assert restarted.engine.spent_for_root(original.root_experiment_id) == 0.0
        assert restarted.ledger.events()[-1].event_type is EventType.EVIDENCE_REJECTED
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_source_burst_and_seeded_trust_do_not_reset_after_restart(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "source-state.db"
    source_id = _submission("unused").source_id
    first = EvidenceFirewall(db_path=db_path, seed_demo=True)
    try:
        trust_before = first.engine.trust(source_id)
        for index in range(1, BURST_THRESHOLD + 1):
            await first.submit(_submission(f"restart-source-{index:02d}"))
        assert first._source_counts[source_id] == BURST_THRESHOLD
    finally:
        first.close()

    restarted = EvidenceFirewall(db_path=db_path, seed_demo=True)
    try:
        assert restarted._source_counts[source_id] == BURST_THRESHOLD
        assert restarted.engine.trust(source_id) == pytest.approx(trust_before)

        result = await restarted.submit(_submission("restart-source-after"))
        assert restarted._source_counts[source_id] == BURST_THRESHOLD + 1
        assert any(
            reason == f"source_burst:{BURST_THRESHOLD + 1}"
            for reason in result["monitor"]["reasons"]
        )
        assert result["metrics"]["escalated"] == 1
    finally:
        restarted.close()
