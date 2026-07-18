"""Focused restart contracts for deterministic runtime reconstruction."""

from __future__ import annotations

import json

import pytest

from core.engine import ROOT_BUDGET, Engine
from core.ledger import EventType, Ledger, build_reversals
from core.runtime import RuntimeReconstructionError, reconstruct_runtime
from core.types import Integrity


def _append_commit(
    ledger: Ledger,
    engine: Engine,
    *,
    evidence_id: str,
    source_id: str,
    experiment_id: str,
    root_id: str,
    prior: float,
) -> float:
    breakdown = engine.propose(
        "C17",
        root_id,
        source_id,
        prior,
        100.0,
        Integrity.L2_VERIFIED,
    )
    spend = engine.spend_log[-1]
    ledger.append(
        EventType.EVIDENCE_COMMITTED,
        {
            "evidence_id": evidence_id,
            "source_id": source_id,
            "experiment_id": experiment_id,
            "root_experiment_id": root_id,
            "claim_id": "C17",
            "relation": "supports",
            "integrity": int(Integrity.L2_VERIFIED),
            "verdict": "commit",
            "reasons": ["shock:low"],
            "shock": 0.1,
            "metrics": {
                "latency_ms": 1.0,
                "gemini_calls": 0,
                "escalated": 0,
            },
            "engine": dict(breakdown),
            "delta": breakdown["bounded_delta"],
            "root_cost": spend["root_cost"],
            "logit_delta": spend["logit_delta"],
            "event_kl": spend["kl"],
        },
    )
    return breakdown["posterior"]


def test_restart_preserves_active_root_budget_and_spend_log(tmp_path) -> None:
    path = tmp_path / "runtime.db"
    original = Engine()
    with Ledger(path) as ledger:
        _append_commit(
            ledger,
            original,
            evidence_id="EV-1",
            source_id="lab-alpha",
            experiment_id="EXP-1",
            root_id="ROOT-1",
            prior=0.5,
        )
        original_spent = original.spent_for_root("ROOT-1")

    with Ledger(path) as reopened:
        runtime = reconstruct_runtime(reopened)

    assert runtime.engine.spent_for_root("ROOT-1") == original_spent
    assert runtime.engine.spend_log == original.spend_log
    assert runtime.source_counts == {"lab-alpha": 1}
    assert runtime.idempotent_responses["EV-1"]["event_seq"] == 1

    runtime.engine.propose(
        "C17",
        "ROOT-1",
        "lab-alpha",
        0.6,
        100.0,
        Integrity.L2_VERIFIED,
    )
    assert runtime.engine.spent_for_root("ROOT-1") == pytest.approx(ROOT_BUDGET)


def test_restart_remembers_retraction_and_excludes_reversed_commits(tmp_path) -> None:
    path = tmp_path / "retracted.db"
    original = Engine()
    with Ledger(path) as ledger:
        prior = _append_commit(
            ledger,
            original,
            evidence_id="EV-1",
            source_id="lab-alpha",
            experiment_id="EXP-1",
            root_id="ROOT-1",
            prior=0.5,
        )
        _append_commit(
            ledger,
            original,
            evidence_id="EV-2",
            source_id="lab-alpha",
            experiment_id="EXP-2",
            root_id="ROOT-1",
            prior=prior,
        )
        events = ledger.events()
        ledger.append(
            EventType.RETRACTION,
            {"root_experiment_id": "ROOT-1", "reason": "fabricated"},
        )
        for event_type, payload in build_reversals(events, "ROOT-1"):
            original_event = next(
                event for event in events if event.seq == payload["reverses_seq"]
            )
            ledger.append(
                event_type,
                {
                    **payload,
                    "root_experiment_id": "ROOT-1",
                    "root_cost": -original_event.payload["root_cost"],
                },
            )

    with Ledger(path) as reopened:
        runtime = reconstruct_runtime(reopened)

    assert runtime.retracted_roots == {"ROOT-1"}
    assert runtime.engine.spent_for_root("ROOT-1") == 0.0
    assert runtime.engine.spend_log == []
    assert runtime.source_counts == {"lab-alpha": 2}
    assert set(runtime.idempotent_responses) == {"EV-1", "EV-2"}
    assert "raw_text" not in json.dumps(runtime.idempotent_responses)


def test_reconstruction_rejects_a_tampered_hash_chain(tmp_path) -> None:
    path = tmp_path / "tampered.db"
    with Ledger(path) as ledger:
        ledger.append(
            EventType.RETRACTION,
            {"root_experiment_id": "ROOT-1", "reason": "fabricated"},
        )
        ledger._conn.execute(
            "UPDATE events SET payload = ? WHERE seq = 1",
            ('{"root_experiment_id":"ROOT-ATTACKER"}',),
        )
        ledger._conn.commit()

        with pytest.raises(RuntimeReconstructionError, match="hash-chain"):
            reconstruct_runtime(ledger)
