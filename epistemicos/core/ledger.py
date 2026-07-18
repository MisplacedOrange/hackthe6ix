"""Append-only, hash-chained SQLite event ledger and deterministic reducer (G03).

Temporal containment (spec §4.4):
- The event log is append-only and hash-chained; tampering is detectable.
- Retraction appends REVERSAL events; history is never deleted or rewritten.
- Graph state is never stored -- it is always *reduced* from the event stream,
  so replay-from-scratch is the single source of truth.

Hash chain:
    hash = sha256(prev_hash + seq + event_type + canonical_payload)
    genesis prev_hash = "0" * 64

Timestamps are deliberately NOT part of the hash preimage, so replay
verification is deterministic regardless of wall-clock capture.

Payload conventions (handoff contract for G08's event stream):
    CLAIM_REGISTERED     {"claim_id", "text", "prior"}
    EVIDENCE_COMMITTED   {"claim_id", "evidence_id", "experiment_id",
                          "root_experiment_id", "relation", "delta", "integrity"}
    EVIDENCE_PROVISIONAL same fields as COMMITTED (recorded, no belief change)
    EVIDENCE_ESCROWED    same fields as COMMITTED (recorded, no belief change)
    EVIDENCE_REJECTED    free-form; typically {"claim_id"?, "evidence_id"?, "reasons"}
    ESCROW_RELEASED      same fields as COMMITTED (applies like a commit)
    ESCROW_REJECTED      free-form (no belief change)
    RETRACTION           {"root_experiment_id", "reason"}
    REVERSAL             {"claim_id", "evidence_id", "delta", "reverses_seq"}
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import weakref
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from core.types import Claim

#: prev_hash of the first event in the chain.
GENESIS_HASH = "0" * 64


class EventType(StrEnum):
    """Every kind of transition the ledger may record."""

    CLAIM_REGISTERED = "CLAIM_REGISTERED"
    EVIDENCE_COMMITTED = "EVIDENCE_COMMITTED"
    EVIDENCE_PROVISIONAL = "EVIDENCE_PROVISIONAL"
    EVIDENCE_ESCROWED = "EVIDENCE_ESCROWED"
    EVIDENCE_REJECTED = "EVIDENCE_REJECTED"
    ESCROW_RELEASED = "ESCROW_RELEASED"
    ESCROW_REJECTED = "ESCROW_REJECTED"
    RETRACTION = "RETRACTION"
    REVERSAL = "REVERSAL"


#: Event types whose delta actually mutates committed belief.
_APPLYING = frozenset({EventType.EVIDENCE_COMMITTED, EventType.ESCROW_RELEASED})


class Event(BaseModel):
    """One immutable, hash-chained ledger entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seq: int = Field(ge=1)
    ts: float
    event_type: EventType
    payload: dict[str, Any]
    prev_hash: str = Field(min_length=64, max_length=64)
    hash: str = Field(min_length=64, max_length=64)


class GraphState(BaseModel):
    """Belief-graph state, always derived by `reduce` -- never stored."""

    model_config = ConfigDict(extra="forbid")

    claims: dict[str, Claim] = Field(default_factory=dict)


def canonical_json(payload: dict[str, Any]) -> str:
    """Canonical JSON used for hashing. Must be byte-stable across replays."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(prev_hash: str, seq: int, event_type: str, canonical_payload: str) -> str:
    preimage = f"{prev_hash}{seq}{event_type}{canonical_payload}"
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


class Ledger:
    """Append-only SQLite event ledger (WAL mode) with hash chaining.

    Supports file-backed databases and ":memory:". A single connection is
    held for the ledger's lifetime (required for :memory: semantics).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        self._finalizer = weakref.finalize(self, self._conn.close)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq        INTEGER PRIMARY KEY,
                ts         REAL    NOT NULL,
                event_type TEXT    NOT NULL,
                payload    TEXT    NOT NULL,
                prev_hash  TEXT    NOT NULL,
                hash       TEXT    NOT NULL
            )
            """
        )
        # Resume the chain if the database already has history.
        row = self._conn.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        self._last_seq: int = row[0] if row else 0
        self._last_hash: str = row[1] if row else GENESIS_HASH

    # -- write path --------------------------------------------------------

    def append(self, event_type: EventType | str, payload: dict[str, Any]) -> Event:
        """Atomically append one event and return its immutable envelope."""
        return self.append_batch([(event_type, payload)])[0]

    def append_batch(
        self,
        events: Iterable[tuple[EventType | str, dict[str, Any]]],
    ) -> list[Event]:
        """Append a hash-linked event batch in one SQLite transaction.

        The iterable is fully materialized and validated before any write.
        Sequence allocation reads the persisted chain tip under
        ``BEGIN IMMEDIATE``, allowing multiple ``Ledger`` instances to append
        without relying on a stale in-memory sequence.  If any insert or
        commit fails, every row in the batch is rolled back and the cached
        chain tip is restored.  An empty batch is a no-op.

        This is the write seam for compound transitions such as one
        ``RETRACTION`` followed by all of its ``REVERSAL`` events.
        """
        prepared: list[tuple[EventType, str, dict[str, Any]]] = []
        for event_type, payload in events:
            etype = EventType(event_type)
            canonical = canonical_json(payload)
            # The returned Event payload must be the same immutable snapshot
            # represented by the bytes stored and hashed in SQLite, not a
            # caller-owned dictionary that may later change.
            payload_snapshot = json.loads(canonical)
            prepared.append((etype, canonical, payload_snapshot))

        if not prepared:
            return []

        with self._lock:
            start_seq = self._last_seq
            start_hash = self._last_hash
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                start_seq = row[0] if row else 0
                start_hash = row[1] if row else GENESIS_HASH

                next_seq = start_seq
                prev_hash = start_hash
                appended: list[Event] = []
                for etype, canonical, payload_snapshot in prepared:
                    next_seq += 1
                    digest = compute_hash(
                        prev_hash,
                        next_seq,
                        etype.value,
                        canonical,
                    )
                    ts = time.time()
                    event = Event(
                        seq=next_seq,
                        ts=ts,
                        event_type=etype,
                        payload=payload_snapshot,
                        prev_hash=prev_hash,
                        hash=digest,
                    )
                    self._conn.execute(
                        "INSERT INTO events "
                        "(seq, ts, event_type, payload, prev_hash, hash) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            event.seq,
                            event.ts,
                            event.event_type.value,
                            canonical,
                            event.prev_hash,
                            event.hash,
                        ),
                    )
                    appended.append(event)
                    prev_hash = digest

                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                self._last_seq = start_seq
                self._last_hash = start_hash
                raise

            self._last_seq = next_seq
            self._last_hash = prev_hash
            return appended

    # -- read path ---------------------------------------------------------

    def events(self) -> list[Event]:
        """All events ordered by seq. Payloads are parsed from stored JSON."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, ts, event_type, payload, prev_hash, hash "
                "FROM events ORDER BY seq ASC"
            ).fetchall()
        return [
            Event(
                seq=seq,
                ts=ts,
                event_type=EventType(etype),
                payload=json.loads(payload),
                prev_hash=prev_hash,
                hash=digest,
            )
            for seq, ts, etype, payload, prev_hash, digest in rows
        ]

    def verify_chain(self) -> tuple[bool, str | None]:
        """Recompute the full hash chain from stored rows.

        Returns (True, None) if intact, else (False, reason) for the first
        broken link.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, event_type, payload, prev_hash, hash "
                "FROM events ORDER BY seq ASC"
            ).fetchall()
        expected_prev = GENESIS_HASH
        expected_seq = 1
        for seq, etype, payload, prev_hash, digest in rows:
            if seq != expected_seq:
                return False, f"seq {seq}: expected contiguous seq {expected_seq}"
            if etype not in EventType._value2member_map_:
                return False, f"seq {seq}: unknown event_type {etype!r}"
            if prev_hash != expected_prev:
                return False, f"seq {seq}: prev_hash does not match hash of seq {seq - 1}"
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return False, f"seq {seq}: stored payload is not valid JSON"
            recomputed = compute_hash(prev_hash, seq, etype, canonical_json(parsed))
            if recomputed != digest:
                return False, f"seq {seq}: stored hash does not match recomputed hash"
            expected_prev = digest
            expected_seq += 1
        return True, None

    def state(self) -> GraphState:
        """Convenience: reduce the full event stream from scratch."""
        return reduce(self.events())

    def close(self) -> None:
        with self._lock:
            if self._finalizer.alive:
                self._finalizer()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# -- reducer ---------------------------------------------------------------


def _apply_delta(state: GraphState, payload: dict[str, Any]) -> None:
    claim = state.claims.get(payload.get("claim_id", ""))
    if claim is None:
        return
    delta = float(payload["delta"])
    claim.confidence = _clamp01(claim.confidence + delta)
    evidence_id = payload.get("evidence_id")
    if evidence_id is not None:
        # replicates counts as supports; sign of delta decides the bucket.
        if delta >= 0:
            claim.supporting.append(evidence_id)
        else:
            claim.contradicting.append(evidence_id)


def reduce(events: Iterable[Event]) -> GraphState:
    """Pure, deterministic fold of an event stream into a GraphState.

    Escrowed/provisional/rejected evidence is recorded in the ledger but
    MUST NOT move committed belief; only EVIDENCE_COMMITTED, ESCROW_RELEASED,
    and REVERSAL touch claim confidence or evidence lists.
    """
    state = GraphState()
    for event in events:
        payload = event.payload
        if event.event_type is EventType.CLAIM_REGISTERED:
            state.claims[payload["claim_id"]] = Claim(
                id=payload["claim_id"],
                text=payload["text"],
                confidence=_clamp01(float(payload.get("prior", 0.5))),
            )
        elif event.event_type in _APPLYING:
            _apply_delta(state, payload)
        elif event.event_type is EventType.REVERSAL:
            claim = state.claims.get(payload.get("claim_id", ""))
            if claim is None:
                continue
            claim.confidence = _clamp01(claim.confidence + float(payload["delta"]))
            evidence_id = payload.get("evidence_id")
            if evidence_id is not None:
                claim.supporting = [e for e in claim.supporting if e != evidence_id]
                claim.contradicting = [e for e in claim.contradicting if e != evidence_id]
        # EVIDENCE_PROVISIONAL, EVIDENCE_ESCROWED, EVIDENCE_REJECTED,
        # ESCROW_REJECTED, RETRACTION: recorded, zero committed influence.
    return state


# -- reversal generation ----------------------------------------------------


def build_reversals(
    events: Iterable[Event], root_experiment_id: str
) -> list[tuple[EventType, dict[str, Any]]]:
    """Emit REVERSAL payloads negating every committed delta under a root.

    Scans EVIDENCE_COMMITTED and ESCROW_RELEASED events whose payload
    descends from `root_experiment_id` and returns (event_type, payload)
    pairs, most recent first, each exactly negating a prior committed delta.
    Events already reversed (matched by `reverses_seq`) are skipped, so
    calling this twice never double-reverses. History is never deleted.
    """
    event_list = list(events)
    already_reversed = {
        e.payload.get("reverses_seq")
        for e in event_list
        if e.event_type is EventType.REVERSAL
    }
    reversals: list[tuple[EventType, dict[str, Any]]] = []
    for event in reversed(event_list):
        if event.event_type not in _APPLYING:
            continue
        if event.payload.get("root_experiment_id") != root_experiment_id:
            continue
        if event.seq in already_reversed:
            continue
        reversals.append(
            (
                EventType.REVERSAL,
                {
                    "claim_id": event.payload["claim_id"],
                    "evidence_id": event.payload.get("evidence_id"),
                    "delta": -float(event.payload["delta"]),
                    "reverses_seq": event.seq,
                },
            )
        )
    return reversals
