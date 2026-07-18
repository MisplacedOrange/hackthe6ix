"""G03 acceptance tests: hash chain, reducer determinism, escrow, retraction.

All tests are provider-free and offline.
"""

from __future__ import annotations

import sqlite3

import pytest

from core.ledger import (
    GENESIS_HASH,
    EventType,
    Ledger,
    build_reversals,
    reduce,
)
from core.types import ClaimState


def register(ledger: Ledger, claim_id: str = "C17", prior: float = 0.5) -> None:
    ledger.append(
        EventType.CLAIM_REGISTERED,
        {"claim_id": claim_id, "text": "compound X has no in-vivo effect", "prior": prior},
    )


def evidence_payload(
    *,
    claim_id: str = "C17",
    evidence_id: str = "EV-1",
    experiment_id: str = "EXP-001",
    root_experiment_id: str = "ROOT-001",
    relation: str = "contradicts",
    delta: float = -0.1,
    integrity: int = 2,
) -> dict:
    return {
        "claim_id": claim_id,
        "evidence_id": evidence_id,
        "experiment_id": experiment_id,
        "root_experiment_id": root_experiment_id,
        "relation": relation,
        "delta": delta,
        "integrity": integrity,
    }


# -- basic chain mechanics ---------------------------------------------------


def test_append_assigns_monotonic_seq_and_links_hashes():
    ledger = Ledger(":memory:")
    register(ledger)
    e2 = ledger.append(EventType.EVIDENCE_COMMITTED, evidence_payload(delta=0.1, relation="supports"))
    events = ledger.events()
    assert [e.seq for e in events] == [1, 2]
    assert events[0].prev_hash == GENESIS_HASH
    assert events[1].prev_hash == events[0].hash
    assert e2.hash == events[1].hash
    ok, reason = ledger.verify_chain()
    assert ok and reason is None


def test_empty_ledger_verifies():
    ledger = Ledger(":memory:")
    assert ledger.verify_chain() == (True, None)
    assert ledger.state().claims == {}


# -- tamper detection --------------------------------------------------------


def test_tampered_payload_breaks_chain(tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db)
    register(ledger)
    ledger.append(EventType.EVIDENCE_COMMITTED, evidence_payload(delta=0.2, relation="supports"))
    ledger.close()

    raw = sqlite3.connect(db)
    raw.execute(
        "UPDATE events SET payload = ? WHERE seq = 2",
        ('{"claim_id":"C17","delta":0.9,"evidence_id":"EV-1"}',),
    )
    raw.commit()
    raw.close()

    reopened = Ledger(db)
    ok, reason = reopened.verify_chain()
    assert ok is False
    assert reason is not None and "seq 2" in reason
    reopened.close()


def test_tampered_hash_breaks_chain(tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db)
    register(ledger)
    ledger.append(EventType.EVIDENCE_COMMITTED, evidence_payload(delta=0.2, relation="supports"))
    ledger.append(EventType.EVIDENCE_ESCROWED, evidence_payload(evidence_id="EV-2"))
    ledger.close()

    raw = sqlite3.connect(db)
    raw.execute("UPDATE events SET hash = ? WHERE seq = 1", ("f" * 64,))
    raw.commit()
    raw.close()

    reopened = Ledger(db)
    ok, reason = reopened.verify_chain()
    assert ok is False
    assert reason is not None
    # seq 1's stored hash no longer matches its recomputed hash.
    assert "seq 1" in reason
    reopened.close()


def test_tampered_event_type_breaks_chain(tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db)
    register(ledger)
    ledger.append(EventType.EVIDENCE_ESCROWED, evidence_payload())
    ledger.close()

    raw = sqlite3.connect(db)
    # Attacker tries to upgrade an escrowed event into a committed one.
    raw.execute("UPDATE events SET event_type = 'EVIDENCE_COMMITTED' WHERE seq = 2")
    raw.commit()
    raw.close()

    reopened = Ledger(db)
    ok, reason = reopened.verify_chain()
    assert ok is False
    assert reason is not None and "seq 2" in reason
    reopened.close()


def test_deleted_row_breaks_chain(tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db)
    register(ledger)
    ledger.append(EventType.EVIDENCE_COMMITTED, evidence_payload(delta=0.2, relation="supports"))
    ledger.append(EventType.RETRACTION, {"root_experiment_id": "ROOT-001", "reason": "fraud"})
    ledger.close()

    raw = sqlite3.connect(db)
    raw.execute("DELETE FROM events WHERE seq = 2")
    raw.commit()
    raw.close()

    reopened = Ledger(db)
    ok, reason = reopened.verify_chain()
    assert ok is False
    assert reason is not None
    reopened.close()


# -- reducer determinism -----------------------------------------------------


def test_reduce_from_scratch_is_deterministic():
    ledger = Ledger(":memory:")
    register(ledger, prior=0.5)
    ledger.append(EventType.EVIDENCE_COMMITTED, evidence_payload(delta=0.15, relation="supports"))
    ledger.append(EventType.EVIDENCE_ESCROWED, evidence_payload(evidence_id="EV-2", delta=-0.3))
    ledger.append(EventType.ESCROW_RELEASED, evidence_payload(evidence_id="EV-2", delta=-0.3))
    events = ledger.events()

    first = reduce(events)
    second = reduce(events)
    assert first.model_dump() == second.model_dump()
    assert first.model_dump() == ledger.state().model_dump()


def test_same_events_in_two_ledgers_reduce_to_identical_state(tmp_path):
    """Timestamps differ across ledgers but state must not."""
    file_ledger = Ledger(tmp_path / "a.db")
    mem_ledger = Ledger(":memory:")
    for ledger in (file_ledger, mem_ledger):
        register(ledger, prior=0.4)
        ledger.append(EventType.EVIDENCE_COMMITTED, evidence_payload(delta=0.2, relation="supports"))
        ledger.append(EventType.EVIDENCE_PROVISIONAL, evidence_payload(evidence_id="EV-9"))

    assert file_ledger.state().model_dump() == mem_ledger.state().model_dump()
    # Identical content means identical hash chains (ts is not in the preimage).
    assert [e.hash for e in file_ledger.events()] == [e.hash for e in mem_ledger.events()]
    file_ledger.close()
    mem_ledger.close()


# -- escrow / provisional containment ----------------------------------------


def test_uncommitted_events_do_not_move_belief():
    ledger = Ledger(":memory:")
    register(ledger, prior=0.5)
    ledger.append(EventType.EVIDENCE_PROVISIONAL, evidence_payload(evidence_id="EV-1", delta=0.3))
    ledger.append(EventType.EVIDENCE_ESCROWED, evidence_payload(evidence_id="EV-2", delta=-0.4))
    ledger.append(EventType.EVIDENCE_REJECTED, evidence_payload(evidence_id="EV-3", delta=0.9))
    ledger.append(EventType.ESCROW_REJECTED, evidence_payload(evidence_id="EV-2", delta=-0.4))

    claim = ledger.state().claims["C17"]
    assert claim.confidence == pytest.approx(0.5)
    assert claim.supporting == []
    assert claim.contradicting == []
    assert claim.state == ClaimState.UNKNOWN


def test_escrow_release_applies_the_delta():
    ledger = Ledger(":memory:")
    register(ledger, prior=0.5)
    ledger.append(EventType.EVIDENCE_ESCROWED, evidence_payload(evidence_id="EV-2", delta=-0.2))
    before = ledger.state().claims["C17"].confidence
    assert before == pytest.approx(0.5)

    ledger.append(EventType.ESCROW_RELEASED, evidence_payload(evidence_id="EV-2", delta=-0.2))
    claim = ledger.state().claims["C17"]
    assert claim.confidence == pytest.approx(0.3)
    assert claim.contradicting == ["EV-2"]
    assert claim.state == ClaimState.CONTRADICTED


def test_committed_evidence_buckets_by_delta_sign_and_clamps():
    ledger = Ledger(":memory:")
    register(ledger, prior=0.9)
    ledger.append(
        EventType.EVIDENCE_COMMITTED,
        evidence_payload(evidence_id="EV-S", relation="replicates", delta=0.4),
    )
    claim = ledger.state().claims["C17"]
    assert claim.confidence == pytest.approx(1.0)  # clamped to [0, 1]
    assert claim.supporting == ["EV-S"]  # replicates counts as supports
    ledger.append(
        EventType.EVIDENCE_COMMITTED,
        evidence_payload(evidence_id="EV-C", relation="contradicts", delta=-0.25),
    )
    claim = ledger.state().claims["C17"]
    assert claim.contradicting == ["EV-C"]
    assert claim.state == ClaimState.CONTESTED


# -- retraction and reversal --------------------------------------------------


def test_retraction_flow_restores_prior_state_and_preserves_history(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    register(ledger, prior=0.5)
    pre_commit = ledger.state().claims["C17"].confidence

    ledger.append(
        EventType.EVIDENCE_COMMITTED,
        evidence_payload(evidence_id="EV-1", relation="supports", delta=0.2, root_experiment_id="ROOT-BAD"),
    )
    ledger.append(
        EventType.ESCROW_RELEASED,
        evidence_payload(evidence_id="EV-2", relation="contradicts", delta=-0.1, root_experiment_id="ROOT-BAD"),
    )
    # An unrelated root that must NOT be reversed.
    ledger.append(
        EventType.EVIDENCE_COMMITTED,
        evidence_payload(evidence_id="EV-OK", relation="supports", delta=0.05, root_experiment_id="ROOT-GOOD"),
    )
    mid = ledger.state().claims["C17"]
    assert mid.confidence == pytest.approx(0.65)

    ledger.append(EventType.RETRACTION, {"root_experiment_id": "ROOT-BAD", "reason": "confirmed fabrication"})
    reversals = build_reversals(ledger.events(), "ROOT-BAD")
    assert len(reversals) == 2
    # Most recent first: EV-2's release is reversed before EV-1's commit.
    assert reversals[0][1]["evidence_id"] == "EV-2"
    assert reversals[0][1]["delta"] == pytest.approx(0.1)
    assert reversals[1][1]["evidence_id"] == "EV-1"
    assert reversals[1][1]["delta"] == pytest.approx(-0.2)
    for event_type, payload in reversals:
        ledger.append(event_type, payload)

    claim = ledger.state().claims["C17"]
    # Confidence back to pre-commit value plus only the unrelated root's delta.
    assert claim.confidence == pytest.approx(pre_commit + 0.05)
    assert "EV-1" not in claim.supporting
    assert "EV-2" not in claim.contradicting
    assert claim.supporting == ["EV-OK"]

    # Append-only: original commit events are still in the ledger.
    types = [e.event_type for e in ledger.events()]
    assert types.count(EventType.EVIDENCE_COMMITTED) == 2
    assert types.count(EventType.ESCROW_RELEASED) == 1
    assert types.count(EventType.REVERSAL) == 2
    ok, reason = ledger.verify_chain()
    assert ok and reason is None
    ledger.close()


def test_build_reversals_skips_already_reversed_events():
    ledger = Ledger(":memory:")
    register(ledger)
    ledger.append(EventType.EVIDENCE_COMMITTED, evidence_payload(evidence_id="EV-1", delta=0.2, relation="supports"))
    for event_type, payload in build_reversals(ledger.events(), "ROOT-001"):
        ledger.append(event_type, payload)
    assert build_reversals(ledger.events(), "ROOT-001") == []
    assert ledger.state().claims["C17"].confidence == pytest.approx(0.5)


def test_build_reversals_ignores_non_committed_events():
    ledger = Ledger(":memory:")
    register(ledger)
    ledger.append(EventType.EVIDENCE_ESCROWED, evidence_payload(evidence_id="EV-1", delta=0.2))
    ledger.append(EventType.EVIDENCE_PROVISIONAL, evidence_payload(evidence_id="EV-2", delta=0.1))
    assert build_reversals(ledger.events(), "ROOT-001") == []


# -- atomic batch append ------------------------------------------------------


def test_append_batch_atomically_links_retraction_and_reversals(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    register(ledger, prior=0.5)
    ledger.append(
        EventType.EVIDENCE_COMMITTED,
        evidence_payload(
            evidence_id="EV-1",
            delta=0.2,
            relation="supports",
            root_experiment_id="ROOT-BAD",
        ),
    )
    prior_tip = ledger.events()[-1]
    reversals = build_reversals(ledger.events(), "ROOT-BAD")

    appended = ledger.append_batch(
        [
            (
                EventType.RETRACTION,
                {"root_experiment_id": "ROOT-BAD", "reason": "audit failure"},
            ),
            *reversals,
        ]
    )

    assert [event.seq for event in appended] == [3, 4]
    assert [event.event_type for event in appended] == [
        EventType.RETRACTION,
        EventType.REVERSAL,
    ]
    assert appended[0].prev_hash == prior_tip.hash
    assert appended[1].prev_hash == appended[0].hash
    assert ledger.events()[-2:] == appended
    assert ledger.state().claims["C17"].confidence == pytest.approx(0.5)
    assert ledger.verify_chain() == (True, None)
    ledger.close()


def test_append_batch_empty_input_is_noop():
    ledger = Ledger(":memory:")
    register(ledger)
    before = ledger.events()

    assert ledger.append_batch([]) == []
    assert ledger.events() == before
    assert ledger.verify_chain() == (True, None)


def test_append_batch_rolls_back_every_row_and_chain_tip_on_sql_failure(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    register(ledger)
    ledger.append(
        EventType.EVIDENCE_COMMITTED,
        evidence_payload(delta=0.2, relation="supports", root_experiment_id="ROOT-BAD"),
    )
    before = ledger.events()
    prior_tip = before[-1]
    reversal = build_reversals(before, "ROOT-BAD")[0]
    ledger._conn.execute(
        """
        CREATE TEMP TRIGGER force_reversal_failure
        BEFORE INSERT ON events
        WHEN NEW.event_type = 'REVERSAL'
        BEGIN
            SELECT RAISE(ABORT, 'forced batch failure');
        END
        """
    )
    ledger._conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="forced batch failure"):
        ledger.append_batch(
            [
                (
                    EventType.RETRACTION,
                    {"root_experiment_id": "ROOT-BAD", "reason": "audit failure"},
                ),
                reversal,
            ]
        )

    assert ledger.events() == before
    assert ledger.verify_chain() == (True, None)

    # A later successful append must reuse the uncommitted sequence and tip,
    # proving the in-memory chain cache rolled back with SQLite.
    ledger._conn.execute("DROP TRIGGER force_reversal_failure")
    event = ledger.append(
        EventType.RETRACTION,
        {"root_experiment_id": "ROOT-BAD", "reason": "retry"},
    )
    assert event.seq == 3
    assert event.prev_hash == prior_tip.hash
    assert ledger.verify_chain() == (True, None)
    ledger.close()


def test_append_batch_validation_failure_writes_nothing():
    ledger = Ledger(":memory:")
    register(ledger)
    before = ledger.events()

    with pytest.raises(ValueError):
        ledger.append_batch(
            [
                (EventType.RETRACTION, {"root_experiment_id": "ROOT-1"}),
                ("NOT_AN_EVENT_TYPE", {}),
            ]
        )

    assert ledger.events() == before
    assert ledger.verify_chain() == (True, None)


def test_append_batch_allocates_from_persisted_tip_across_ledger_instances(tmp_path):
    path = tmp_path / "ledger.db"
    first = Ledger(path)
    register(first)
    stale = Ledger(path)

    batch = first.append_batch(
        [
            (EventType.EVIDENCE_PROVISIONAL, evidence_payload(evidence_id="EV-1")),
            (EventType.EVIDENCE_ESCROWED, evidence_payload(evidence_id="EV-2")),
        ]
    )
    appended_from_stale_instance = stale.append(
        EventType.EVIDENCE_REJECTED,
        {"evidence_id": "EV-3", "reasons": ["test"]},
    )

    assert [event.seq for event in batch] == [2, 3]
    assert appended_from_stale_instance.seq == 4
    assert appended_from_stale_instance.prev_hash == batch[-1].hash
    assert stale.verify_chain() == (True, None)
    first.close()
    stale.close()


# -- persistence ---------------------------------------------------------------


def test_file_ledger_survives_reopen_and_chain_resumes(tmp_path):
    db = tmp_path / "ledger.db"
    ledger = Ledger(db)
    register(ledger, prior=0.5)
    ledger.append(EventType.EVIDENCE_COMMITTED, evidence_payload(delta=0.2, relation="supports"))
    last_hash = ledger.events()[-1].hash
    ledger.close()

    reopened = Ledger(db)
    assert reopened.verify_chain() == (True, None)
    # New appends must chain off the persisted tip, not restart at genesis.
    event = reopened.append(EventType.RETRACTION, {"root_experiment_id": "ROOT-001", "reason": "audit"})
    assert event.seq == 3
    assert event.prev_hash == last_hash
    assert reopened.verify_chain() == (True, None)
    assert reopened.state().claims["C17"].confidence == pytest.approx(0.7)
    reopened.close()
