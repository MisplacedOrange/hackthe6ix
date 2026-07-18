"""G10 end-to-end security assertions over the seeded red-team corpus."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx
import pytest

from api.main import create_app
from core.engine import EPS, ROOT_BUDGET, bernoulli_kl
from core.escalate import BURST_THRESHOLD
from core.ledger import EventType
from core.types import Integrity
from llm.compiler import FakeCompiler
from llm.embedder import FakeEmbedder
from redteam.generate import load_corpus
from redteam.report import run_security_report

CORPUS = load_corpus()
CASES: list[dict[str, Any]] = CORPUS["cases"]
T1_CASES = [case for case in CASES if case["expected_containment"]["threat_tier"] == "T1"]
STRUCTURED_CASES = [case for case in CASES if "<<IR>>" in case["raw_text"]]


def _cases(category: str) -> list[dict[str, Any]]:
    return [case for case in CASES if case["category"] == category]


def _new_app():
    app = create_app(seed_demo=False)
    firewall = app.state.firewall
    firewall.ledger.append(
        EventType.CLAIM_REGISTERED,
        {"claim_id": "C17", "text": "Test claim", "prior": 0.5},
    )
    assert firewall.registry.resolve("anything") is None
    assert isinstance(firewall.compiler, FakeCompiler)
    assert isinstance(firewall.embedder, FakeEmbedder)
    return app


def _payload(case: dict[str, Any]) -> dict[str, str]:
    return {"raw_text": case["raw_text"], **case["submission"]}


def _register(firewall: Any, cases: Iterable[dict[str, Any]]) -> None:
    for case in cases:
        envelope = case["submission"]
        firewall.registry.register_experiment(
            envelope["experiment_id"],
            source_id=envelope["source_id"],
            root_experiment_id=envelope["root_experiment_id"],
            claim_ids={"C17"},
        )


@pytest.fixture
def app():
    application = _new_app()
    yield application
    application.state.firewall.ledger.close()


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        yield test_client


@pytest.mark.parametrize("case", T1_CASES, ids=lambda case: case["id"])
async def test_t1_cases_reject_with_zero_committed_influence(
    app, client: httpx.AsyncClient, case: dict[str, Any]
) -> None:
    before = (await client.get("/claims")).json()["claims"][0]["confidence"]
    response = await client.post("/evidence", json=_payload(case))
    after = (await client.get("/claims")).json()["claims"][0]["confidence"]

    assert response.status_code == 200
    result = response.json()
    assert result["monitor"]["verdict"] == "reject"
    assert result["monitor"]["integrity"] == int(Integrity.L0_RAW)
    assert result["monitor"]["reasons"]
    assert result["engine"] is None
    assert after == before

    event = app.state.firewall.ledger.events()[-1]
    assert event.event_type is EventType.EVIDENCE_REJECTED
    assert event.payload["reasons"]
    assert float(event.payload.get("delta", 0.0)) == 0.0
    assert case["raw_text"] not in json.dumps(event.model_dump(mode="json"))


async def test_unregistered_structured_cases_never_gain_unauthorized_integrity(
    app, client: httpx.AsyncClient
) -> None:
    for case in STRUCTURED_CASES:
        result = (await client.post("/evidence", json=_payload(case))).json()
        assert result["monitor"]["integrity"] <= int(Integrity.L1_PARSED), case["id"]
        assert result["monitor"]["verdict"] != "commit", case["id"]
        assert result["engine"] is None or result["engine"]["bounded_delta"] == 0.0
    assert (await client.get("/claims")).json()["claims"][0]["confidence"] == 0.5


async def test_duplicate_flood_shares_one_global_root_budget(
    app, client: httpx.AsyncClient
) -> None:
    cases = _cases("duplicate_flood")
    firewall = app.state.firewall
    _register(firewall, cases)
    root_id = cases[0]["submission"]["root_experiment_id"]
    deltas: list[float] = []

    for case in cases:
        result = (await client.post("/evidence", json=_payload(case))).json()
        assert result["monitor"]["integrity"] == int(Integrity.L2_VERIFIED)
        assert firewall.engine.spent_for_root(root_id) <= ROOT_BUDGET + 1e-12
        if result["monitor"]["verdict"] == "commit":
            deltas.append(abs(float(result["engine"]["bounded_delta"])))

    records = firewall.engine.proposals_for_root(root_id)
    assert sum(float(record["root_cost"]) for record in records) <= ROOT_BUDGET + 1e-12
    assert firewall.engine.spent_for_root(root_id) == pytest.approx(ROOT_BUDGET)
    assert any(delta > 0.0 for delta in deltas)

    overflow = _payload(cases[-1])
    overflow["idempotency_key"] += "-overflow"
    overflow_result = (await client.post("/evidence", json=overflow)).json()
    if overflow_result["monitor"]["verdict"] == "commit":
        assert overflow_result["engine"]["bounded_delta"] == 0.0
    assert firewall.engine.spent_for_root(root_id) == pytest.approx(ROOT_BUDGET)


async def test_slow_drip_is_per_event_bounded_and_source_burst_escalates(
    app, client: httpx.AsyncClient
) -> None:
    cases = _cases("slow_drip")
    firewall = app.state.firewall
    _register(firewall, cases)

    for index, case in enumerate(cases, start=1):
        before = (await client.get("/claims")).json()["claims"][0]["confidence"]
        result = (await client.post("/evidence", json=_payload(case))).json()
        after = (await client.get("/claims")).json()["claims"][0]["confidence"]
        engine = result["engine"]
        assert engine is not None
        assert bernoulli_kl(float(engine["posterior"]), float(engine["prior"])) <= (
            EPS[Integrity.L2_VERIFIED] + 1e-12
        )
        assert abs(after - before) <= abs(float(engine["bounded_delta"])) + 1e-15
        if index > BURST_THRESHOLD:
            assert any(reason.startswith("source_burst:") for reason in result["monitor"]["reasons"])
            assert result["metrics"]["escalated"] == 1


async def test_fake_replication_never_reaches_l3(app, client: httpx.AsyncClient) -> None:
    case = _cases("fake_replication")[0]
    result = (await client.post("/evidence", json=_payload(case))).json()
    assert result["monitor"]["integrity"] < int(Integrity.L3_REPLICATED)
    assert result["monitor"]["verdict"] != "commit"
    assert any("replication" in reason for reason in result["monitor"]["reasons"])


async def test_schema_valid_semantic_lie_stays_inert_and_contained(
    client: httpx.AsyncClient,
) -> None:
    case = _cases("schema_valid_semantic_lie")[0]
    before = (await client.get("/claims")).json()["claims"][0]["confidence"]
    result = (await client.post("/evidence", json=_payload(case))).json()
    after = (await client.get("/claims")).json()["claims"][0]["confidence"]

    assert result["monitor"]["verdict"] in {"provisional", "escrow", "reject"}
    assert result["monitor"]["integrity"] <= int(Integrity.L1_PARSED)
    assert result["engine"] is None or result["engine"]["bounded_delta"] == 0.0
    assert after == before


async def test_every_rejection_has_machine_reasons(client: httpx.AsyncClient) -> None:
    for case in T1_CASES:
        result = (await client.post("/evidence", json=_payload(case))).json()
        reasons = result["monitor"]["reasons"]
        assert reasons and all(isinstance(reason, str) and reason.strip() for reason in reasons)
        assert all(" " not in reason.split(":", 1)[0] for reason in reasons)


async def test_hash_chain_verifies_then_detects_payload_tampering(
    app, client: httpx.AsyncClient
) -> None:
    await client.post("/evidence", json=_payload(T1_CASES[0]))
    ledger = app.state.firewall.ledger
    assert ledger.verify_chain() == (True, None)
    ledger._conn.execute("UPDATE events SET payload = ? WHERE seq = 2", ('{"tampered":true}',))
    ledger._conn.commit()
    valid, reason = ledger.verify_chain()
    assert valid is False
    assert reason and "hash" in reason


async def test_retraction_is_visible_exact_and_idempotent(
    app, client: httpx.AsyncClient
) -> None:
    case = _cases("duplicate_flood")[0]
    firewall = app.state.firewall
    _register(firewall, [case])
    committed = (await client.post("/evidence", json=_payload(case))).json()
    assert committed["monitor"]["verdict"] == "commit"
    assert (await client.get("/claims")).json()["claims"][0]["confidence"] > 0.5

    first = await client.post(f"/retract/{case['submission']['experiment_id']}")
    assert first.status_code == 200
    assert first.json()["reversed_events"] == 1
    assert (await client.get("/claims")).json()["claims"][0]["confidence"] == 0.5
    events = (await client.get("/events")).json()["events"]
    assert [event["event_type"] for event in events][-2:] == ["RETRACTION", "REVERSAL"]
    assert firewall.ledger.verify_chain() == (True, None)

    second = await client.post(f"/retract/{case['submission']['experiment_id']}")
    assert second.json()["reversed_events"] == 0


def test_default_pipeline_is_provider_free(app) -> None:
    firewall = app.state.firewall
    assert isinstance(firewall.compiler, FakeCompiler)
    assert isinstance(firewall.embedder, FakeEmbedder)


async def test_security_report_is_reproducible_and_json_ready() -> None:
    first = await run_security_report()
    second = await run_security_report()
    assert first == second
    json.dumps(first, sort_keys=True, allow_nan=False)
    assert first["seed"] == CORPUS["seed"]
    assert first["failure_counts"]["failed_cases"] == 0
    assert first["unauthorized_transition_count"] == 0
    assert first["representative_traces"]
    assert "not perfect injection detection" in first["disclaimer"]
    assert first["provider_mode"] == "fake"
