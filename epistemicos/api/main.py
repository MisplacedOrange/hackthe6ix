"""FastAPI surface for the provider-free Evidence Firewall demo (G08)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

from core.engine import Engine, bernoulli_kl
from core.escalate import run_pipeline
from core.ledger import Event, EventType, Ledger, build_reversals
from core.monitor import ProvenanceRegistry, ReferenceMonitor
from core.runtime import reconstruct_runtime
from core.types import EvidenceIR, EvidenceSubmission, Integrity, Relation, Verdict
from demo.metrics import MetricsCollector
from demo.scenarios import claim_seeds, demo_steps
from llm.compiler import CompileError, FakeCompiler, GeminiCompiler
from llm.embedder import ClaimIndex, EvidenceIndex, FakeEmbedder, GeminiEmbedder
from llm.normalize import InputTooLarge, normalize


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return jsonable_encoder(value)


def event_to_sse(event: Event) -> dict[str, str]:
    """Return one stable SSE message with structured JSON data."""
    return {
        "id": str(event.seq),
        "event": event.event_type.value,
        "data": json.dumps(event.model_dump(mode="json"), sort_keys=True),
    }


class _EnvelopeCompiler:
    """Bind compiler provenance proposals to one trusted request envelope."""

    def __init__(self, compiler: Any, submission: EvidenceSubmission) -> None:
        self._compiler = compiler
        self._submission = submission
        self.evidence: EvidenceIR | None = None

    @property
    def last_usage(self) -> dict[str, Any] | None:
        return getattr(self._compiler, "last_usage", None)

    async def compile(self, raw_text: str) -> EvidenceIR:
        evidence = await self._compiler.compile(raw_text)
        expected = self._submission
        proposed = (
            evidence.source_id,
            evidence.experiment_id,
            evidence.root_experiment_id,
        )
        trusted = (
            expected.source_id,
            expected.experiment_id,
            expected.root_experiment_id,
        )
        if proposed != trusted:
            raise CompileError("provenance envelope mismatch")
        self.evidence = evidence
        return evidence


class EvidenceFirewall:
    """Serialized monitor/engine/ledger coordinator.

    The lock is the demo-scale atomic mutation boundary. Provider work,
    deterministic shadowing, budget charge, and ledger append for one request
    cannot interleave with another request. The SQLite ledger remains the
    durable audit source; raw L0 text is never stored in ledger payloads.
    """

    def __init__(
        self,
        *,
        db_path: str | Path = ":memory:",
        compiler: Any | None = None,
        escalation_compiler: Any | None = None,
        embedder: Any | None = None,
        seed_demo: bool = False,
    ) -> None:
        self.ledger = Ledger(db_path)
        existing_events = self.ledger.events()
        restored = reconstruct_runtime(self.ledger) if existing_events else None
        self.engine = restored.engine if restored is not None else Engine()
        self.compiler = compiler or FakeCompiler()
        self.escalation_compiler = escalation_compiler or self.compiler
        self.embedder = embedder or FakeEmbedder()
        self.registry = ProvenanceRegistry()
        self.monitor = ReferenceMonitor(self.engine, self.registry)
        self.evidence_index = EvidenceIndex()
        self.claim_index: ClaimIndex | None = None
        self.metrics = MetricsCollector()
        self._lock = asyncio.Lock()
        self._responses = (
            restored.idempotent_responses if restored is not None else {}
        )
        self._source_counts = restored.source_counts if restored is not None else {}
        self._retracted_roots = (
            restored.retracted_roots if restored is not None else set()
        )
        if restored is not None:
            self._restore_metrics(existing_events)
        if seed_demo:
            self._seed_demo()

    def close(self) -> None:
        """Release the SQLite connection owned by this service."""
        self.ledger.close()

    def _restore_metrics(self, events: list[Event]) -> None:
        """Rebuild observable aggregates without changing security state."""
        ingestion = {
            EventType.EVIDENCE_COMMITTED,
            EventType.EVIDENCE_PROVISIONAL,
            EventType.EVIDENCE_ESCROWED,
            EventType.EVIDENCE_REJECTED,
        }
        by_seq = {event.seq: event for event in events}
        reversed_ids: list[str] = []
        for event in events:
            if event.event_type in ingestion:
                evidence_id = event.payload.get("evidence_id")
                committed = (
                    [str(evidence_id)]
                    if event.event_type is EventType.EVIDENCE_COMMITTED and evidence_id
                    else []
                )
                self.metrics.ingest(
                    event.payload["metrics"],
                    verdict=event.payload["verdict"],
                    committed_identifiers=committed,
                )
            elif event.event_type is EventType.REVERSAL:
                if event.payload.get("evidence_id"):
                    reversed_ids.append(str(event.payload["evidence_id"]))
        if reversed_ids:
            self.metrics.record_reversals(reversed_ids)

        for root_id in sorted(self._retracted_roots):
            expected: list[str] = []
            actual: list[str] = []
            for event in events:
                if (
                    event.event_type
                    in {EventType.EVIDENCE_COMMITTED, EventType.ESCROW_RELEASED}
                    and event.payload.get("root_experiment_id") == root_id
                    and event.payload.get("evidence_id")
                ):
                    expected.append(str(event.payload["evidence_id"]))
                if (
                    event.event_type is EventType.REVERSAL
                    and event.payload.get("root_experiment_id") == root_id
                    and event.payload.get("evidence_id")
                    and event.payload.get("reverses_seq") in by_seq
                ):
                    actual.append(str(event.payload["evidence_id"]))
            self.metrics.record_retraction(expected, actual)

    def _seed_demo(self) -> None:
        if not self.ledger.events():
            for claim in claim_seeds():
                self.ledger.append(EventType.CLAIM_REGISTERED, claim)

        # Registry records are trusted configuration, not inferred at runtime
        # from compiler output. Register primary records before replications.
        steps = [step for step in demo_steps() if step.evidence_ir is not None]
        trusted_sources: set[str] = set()
        for step in steps:
            ir = step.evidence_ir
            assert ir is not None
            if step.expected_integrity is None or step.expected_integrity < Integrity.L2_VERIFIED:
                continue
            if ir.relation.value is Relation.REPLICATES:
                continue
            self.registry.register_experiment(
                ir.experiment_id,
                source_id=ir.source_id,
                root_experiment_id=ir.root_experiment_id,
                claim_ids={ir.target_claim.value},
                outcome_relation=ir.relation.value,
                effect_direction=ir.effect_direction.value,
            )
            trusted_sources.add(ir.source_id)
        for step in steps:
            ir = step.evidence_ir
            assert ir is not None
            if ir.relation.value is not Relation.REPLICATES:
                continue
            self.registry.register_experiment(
                ir.experiment_id,
                source_id=ir.source_id,
                root_experiment_id=ir.root_experiment_id,
                claim_ids={ir.target_claim.value},
                replicates_experiment_id=ir.claimed_replication_of,
                independent=True,
                outcome_relation=ir.relation.value,
                effect_direction=ir.effect_direction.value,
            )
            trusted_sources.add(ir.source_id)

        # The seeded registry represents pre-audited demo provenance rather
        # than a first-seen internet source. One survived-audit trust step
        # keeps ordinary verified inputs on the observable fast lane while
        # new/unregistered sources still trigger escalation.
        for source_id in sorted(trusted_sources):
            self.engine.record_escrow_survival(source_id)

    async def prepare(self) -> None:
        """Precompute the claim matrix once for live embedding providers."""
        if self.claim_index is not None or isinstance(self.embedder, FakeEmbedder):
            return
        state = self._verified_state()
        claims = {claim.id: claim.text for claim in state.claims.values()}
        if claims:
            self.claim_index = await ClaimIndex.build(claims, self.embedder)

    def _verified_state(self):
        valid, reason = self.ledger.verify_chain()
        if not valid:
            raise RuntimeError(f"ledger verification failed: {reason}")
        return self.ledger.state()

    def _prior_roots(self) -> frozenset[str]:
        applying = {EventType.EVIDENCE_COMMITTED, EventType.ESCROW_RELEASED}
        return frozenset(
            str(event.payload["root_experiment_id"])
            for event in self.ledger.events()
            if event.event_type in applying
            and event.payload.get("root_experiment_id") not in self._retracted_roots
        )

    def _prior_experiments(self) -> frozenset[str]:
        """Specific committed experiments eligible as replication targets."""
        applying = {EventType.EVIDENCE_COMMITTED, EventType.ESCROW_RELEASED}
        return frozenset(
            str(event.payload["experiment_id"])
            for event in self.ledger.events()
            if event.event_type in applying
            and event.payload.get("experiment_id")
            and event.payload.get("root_experiment_id") not in self._retracted_roots
        )

    @staticmethod
    def _event_type(verdict: Verdict) -> EventType:
        return {
            Verdict.COMMIT: EventType.EVIDENCE_COMMITTED,
            Verdict.PROVISIONAL: EventType.EVIDENCE_PROVISIONAL,
            Verdict.ESCROW: EventType.EVIDENCE_ESCROWED,
            Verdict.REJECT: EventType.EVIDENCE_REJECTED,
        }[verdict]

    async def submit(self, submission: EvidenceSubmission) -> dict[str, Any]:
        async with self._lock:
            cached = self._responses.get(submission.idempotency_key)
            if cached is not None:
                return json.loads(json.dumps(cached))

            state = self._verified_state()
            await self.prepare()
            wrapped = _EnvelopeCompiler(self.compiler, submission)
            escalation_wrapped = _EnvelopeCompiler(
                self.escalation_compiler, submission
            )
            burst_count = self._source_counts.get(submission.source_id, 0) + 1
            pipeline = await run_pipeline(
                submission.raw_text,
                compiler=wrapped,
                embedder=self.embedder,
                monitor=self.monitor,
                state=state,
                engine=self.engine,
                claim_index=self.claim_index,
                evidence_index=self.evidence_index,
                prior_roots=self._prior_roots(),
                prior_experiments=self._prior_experiments(),
                source_burst_count=burst_count,
                escalation_compiler=escalation_wrapped,
            )
            verdict = Verdict(pipeline["monitor"]["verdict"])
            evidence = wrapped.evidence
            engine_result = pipeline["engine"]
            live_before = self.engine.clone()

            # ``run_pipeline`` owns the size-limit rejection. Recompute only
            # the safe audit context for accepted-size inputs; oversized raw
            # L0 data is neither hashed into a payload nor stored.
            try:
                normalized = normalize(submission.raw_text)
            except InputTooLarge:
                normalized = None

            if submission.root_experiment_id in self._retracted_roots:
                verdict = Verdict.REJECT
                pipeline["monitor"]["verdict"] = verdict
                pipeline["monitor"]["integrity"] = Integrity.L0_RAW
                pipeline["monitor"]["reasons"].append("provenance:root_retracted")
                engine_result = None

            if verdict is Verdict.COMMIT and evidence is not None and engine_result is not None:
                live = self.engine.propose(
                    evidence.target_claim.value,
                    submission.root_experiment_id,
                    submission.source_id,
                    state.claims[evidence.target_claim.value].confidence,
                    float(engine_result["raw_bf"]),
                    Integrity(pipeline["monitor"]["integrity"]),
                )
                engine_result = live

            reasons = list(pipeline["monitor"]["reasons"])
            integrity = Integrity(pipeline["monitor"]["integrity"])
            evidence_id = submission.idempotency_key
            payload: dict[str, Any] = {
                "evidence_id": evidence_id,
                "source_id": submission.source_id,
                "experiment_id": submission.experiment_id,
                "root_experiment_id": submission.root_experiment_id,
                "claim_id": evidence.target_claim.value if evidence is not None else None,
                "relation": evidence.relation.value.value if evidence is not None else None,
                "integrity": int(integrity),
                "integrity_name": integrity.name,
                "verdict": verdict.value,
                "reasons": reasons,
                "shock": float(pipeline["monitor"]["shock"]),
                "normalization_version": (
                    normalized.normalization_version if normalized is not None else None
                ),
                "normalized_sha256": (
                    normalized.text_sha256 if normalized is not None else None
                ),
                "metrics": dict(pipeline["metrics"]),
            }
            if engine_result is not None:
                payload["engine"] = _json_ready(engine_result)
                payload["delta"] = (
                    float(engine_result["bounded_delta"])
                    if verdict is Verdict.COMMIT
                    else 0.0
                )
                if verdict is Verdict.COMMIT and self.engine.spend_log:
                    spend = self.engine.spend_log[-1]
                    payload.update(
                        {
                            "root_cost": spend["root_cost"],
                            "logit_delta": spend["logit_delta"],
                            "event_kl": spend["kl"],
                        }
                    )

            try:
                event = self.ledger.append(self._event_type(verdict), payload)
            except Exception:
                self.engine = live_before
                self.monitor.engine = self.engine
                raise

            self._source_counts[submission.source_id] = burst_count
            pipeline["monitor"]["verdict"] = verdict
            pipeline["engine"] = engine_result
            pipeline["event_seq"] = event.seq
            committed_ids = [evidence_id] if verdict is Verdict.COMMIT else []
            self.metrics.ingest(
                pipeline["metrics"],
                verdict=verdict,
                committed_identifiers=committed_ids,
            )
            response = _json_ready(pipeline)
            self._responses[submission.idempotency_key] = response
            return json.loads(json.dumps(response))

    async def retract(self, experiment_id: str) -> dict[str, Any]:
        async with self._lock:
            events = self.ledger.events()
            matching = [
                event
                for event in events
                if event.payload.get("experiment_id") == experiment_id
            ]
            if not matching:
                raise KeyError(experiment_id)
            root_id = str(matching[0].payload["root_experiment_id"])
            if root_id in self._retracted_roots:
                return {"root_experiment_id": root_id, "reversed_events": 0}

            originals = {event.seq: event for event in events}
            reversals = build_reversals(events, root_id)
            expected_ids = [
                str(originals[int(reversal["reverses_seq"])].payload["evidence_id"])
                for _, reversal in reversals
                if originals[int(reversal["reverses_seq"])].payload.get("evidence_id")
            ]
            batch: list[tuple[EventType, dict[str, Any]]] = [
                (
                    EventType.RETRACTION,
                    {
                        "experiment_id": experiment_id,
                        "root_experiment_id": root_id,
                        "reason": "demo retraction request",
                    },
                )
            ]
            refunds: list[tuple[Event, float]] = []
            reversed_ids: list[str] = []
            for event_type, reversal in reversals:
                original = originals[int(reversal["reverses_seq"])]
                root_cost = float(original.payload.get("root_cost", 0.0))
                reversal.update(
                    {
                        "root_experiment_id": root_id,
                        "experiment_id": original.payload.get("experiment_id"),
                        "root_cost": -root_cost,
                    }
                )
                batch.append((event_type, reversal))
                refunds.append((original, root_cost))
                if original.payload.get("evidence_id"):
                    reversed_ids.append(str(original.payload["evidence_id"]))

            # The durable tombstone and every reversal are one compound
            # transition: either the whole batch is hash-linked and committed,
            # or SQLite rolls it all back. Only then mirror the refunds into
            # the in-memory engine.
            self.ledger.append_batch(batch)
            for original, root_cost in refunds:
                self.engine.revert(
                    str(original.payload["claim_id"]),
                    root_id,
                    float(original.payload["delta"]),
                    root_cost,
                )
            self.metrics.record_retraction(expected_ids, reversed_ids)
            self._retracted_roots.add(root_id)
            self.metrics.record_reversals(reversed_ids)
            valid, reason = self.ledger.verify_chain()
            if not valid:
                raise RuntimeError(f"ledger verification failed after retraction: {reason}")
            return {
                "root_experiment_id": root_id,
                "reversed_events": len(reversals),
                "reversed_identifiers": reversed_ids,
            }

    def claims(self) -> dict[str, Any]:
        state = self._verified_state()
        events = self.ledger.events()
        claims: list[dict[str, Any]] = []
        for claim in state.claims.values():
            item = claim.model_dump(mode="json")
            item["state"] = claim.state.value
            item["timeline"] = [
                {
                    "seq": event.seq,
                    "event_type": event.event_type.value,
                    "delta": event.payload.get("delta", 0.0),
                    "verdict": event.payload.get("verdict"),
                }
                for event in events
                if event.payload.get("claim_id") == claim.id
            ]
            claims.append(item)
        return {"claims": claims, "chain_valid": True}

    def unauthorized_transition_count(self) -> int:
        """Count committed events that violate the deterministic commit contract."""
        count = 0
        for event in self.ledger.events():
            if event.event_type is not EventType.EVIDENCE_COMMITTED:
                continue
            payload = event.payload
            engine = payload.get("engine")
            authorized = (
                payload.get("verdict") == Verdict.COMMIT.value
                and isinstance(payload.get("integrity"), int)
                and int(payload["integrity"]) >= int(Integrity.L2_VERIFIED)
                and isinstance(engine, dict)
                and payload.get("delta") == engine.get("bounded_delta")
            )
            if not authorized:
                count += 1
        return count

    async def replay_demo(self) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for step in demo_steps():
            if step.raw_text is not None and step.evidence_ir is not None:
                ir = step.evidence_ir
                result = await self.submit(
                    EvidenceSubmission(
                        raw_text=step.raw_text,
                        source_id=ir.source_id,
                        experiment_id=ir.experiment_id,
                        root_experiment_id=ir.root_experiment_id,
                        idempotency_key=f"demo-{step.step_id}",
                    )
                )
            else:
                result = await self.retract("EXP-SUPPORT-001")
            results.append({"step_id": step.step_id, "result": result})
        return {"steps": results}


def create_app(
    *,
    db_path: str | Path = ":memory:",
    compiler: Any | None = None,
    escalation_compiler: Any | None = None,
    embedder: Any | None = None,
    seed_demo: bool = False,
) -> FastAPI:
    service = EvidenceFirewall(
        db_path=db_path,
        compiler=compiler,
        escalation_compiler=escalation_compiler,
        embedder=embedder,
        seed_demo=seed_demo,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await service.prepare()
        yield
        service.close()

    app = FastAPI(
        title="The Evidence Firewall",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.firewall = service

    @app.post("/evidence")
    async def submit_evidence(submission: EvidenceSubmission) -> dict[str, Any]:
        return await service.submit(submission)

    @app.get("/claims")
    async def get_claims() -> dict[str, Any]:
        return service.claims()

    @app.get("/events")
    async def get_events(after: int = 0) -> dict[str, Any]:
        events = [event for event in service.ledger.events() if event.seq > after]
        return {"events": [event.model_dump(mode="json") for event in events]}

    @app.get("/stream")
    async def stream(request: Request) -> EventSourceResponse:
        header = request.headers.get("last-event-id", "0")
        try:
            after = max(0, int(header))
        except ValueError:
            after = 0

        async def event_generator():
            cursor = after
            while not await request.is_disconnected():
                emitted = False
                for event in service.ledger.events():
                    if event.seq > cursor:
                        cursor = event.seq
                        emitted = True
                        yield event_to_sse(event)
                if not emitted:
                    await asyncio.sleep(0.2)

        return EventSourceResponse(event_generator())

    @app.post("/retract/{experiment_id}")
    async def retract(experiment_id: str) -> dict[str, Any]:
        try:
            return await service.retract(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="experiment not found") from exc

    @app.get("/explain/{event_seq}")
    async def explain(event_seq: int) -> dict[str, Any]:
        event = next((item for item in service.ledger.events() if item.seq == event_seq), None)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return {
            "event_seq": event.seq,
            "event_type": event.event_type.value,
            "generated": False,
            "summary": "Deterministic ledger decision; no raw L0 text was provided to an explainer.",
            "reasons": event.payload.get("reasons", []),
            "engine": event.payload.get("engine"),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        snapshot = service.metrics.snapshot()
        latency = snapshot["fast_lane_latency_ms"]
        return {
            **snapshot,
            "provider_mode": (
                "fake" if isinstance(service.compiler, FakeCompiler) else "gemini"
            ),
            "unauthorized_transition_count": service.unauthorized_transition_count(),
            "p50_latency_ms": latency["p50"],
            "p95_latency_ms": latency["p95"],
        }

    @app.get("/demo/scenario")
    async def demo_scenario() -> dict[str, Any]:
        return {"steps": [step.as_dict() for step in demo_steps()]}

    @app.post("/demo/replay")
    async def replay_demo() -> dict[str, Any]:
        return await service.replay_demo()

    return app


def create_app_from_env() -> FastAPI:
    """Build the shipped app from explicit, safe environment configuration."""
    load_dotenv()
    provider_mode = os.environ.get("EPISTEMICOS_PROVIDER_MODE", "fake").casefold()
    db_path = os.environ.get("EPISTEMICOS_DB_PATH", "epistemicos.db")
    seed_demo = os.environ.get("EPISTEMICOS_SEED_DEMO", "0").casefold() not in {
        "0",
        "false",
        "no",
    }
    if provider_mode == "fake":
        return create_app(db_path=db_path, seed_demo=seed_demo)
    if provider_mode == "gemini":
        fast = GeminiCompiler()
        escalation = GeminiCompiler(
            model=os.environ.get("EPISTEMICOS_ESCALATION_MODEL", "gemini-3.5-flash")
        )
        return create_app(
            db_path=db_path,
            compiler=fast,
            escalation_compiler=escalation,
            embedder=GeminiEmbedder(),
            seed_demo=seed_demo,
        )
    raise RuntimeError(
        "EPISTEMICOS_PROVIDER_MODE must be either 'fake' or 'gemini'"
    )


app = create_app_from_env()
