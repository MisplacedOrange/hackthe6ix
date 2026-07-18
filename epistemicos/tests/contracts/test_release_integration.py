"""Cross-goal release contracts for the completed G08-G11 handoff."""

from __future__ import annotations

import httpx
import pytest

from api.main import create_app, create_app_from_env
from core.types import EvidenceSubmission
from demo.scenarios import demo_steps
from llm.normalize import MAX_INPUT_CHARS
from llm.compiler import GeminiCompiler
from llm.embedder import GeminiEmbedder


@pytest.mark.asyncio
async def test_oversized_l0_input_fails_closed_without_raw_audit_storage() -> None:
    app = create_app(seed_demo=True)
    raw_text = "x" * (MAX_INPUT_CHARS + 1)
    payload = {
        "raw_text": raw_text,
        "source_id": "untrusted-source",
        "experiment_id": "EXP-OVERSIZED",
        "root_experiment_id": "ROOT-OVERSIZED",
        "idempotency_key": "oversized-input",
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        result = (await client.post("/evidence", json=payload)).json()
        events = (await client.get("/events")).json()["events"]

    assert result["monitor"]["verdict"] == "reject"
    assert result["monitor"]["integrity"] == 0
    assert result["monitor"]["reasons"] == ["input_too_large"]
    assert result["engine"] is None
    assert raw_text not in str(events)
    app.state.firewall.ledger.close()


@pytest.mark.asyncio
async def test_seeded_release_replay_exercises_every_route_and_real_fast_lane() -> None:
    app = create_app(seed_demo=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        replay = (await client.post("/demo/replay")).json()
        metrics = (await client.get("/metrics")).json()

    verdicts = [
        step["result"].get("monitor", {}).get("verdict")
        for step in replay["steps"]
    ]
    assert verdicts == [
        "commit",
        "commit",
        "provisional",
        "provisional",
        "escrow",
        "commit",
        "commit",
        "reject",
        None,
    ]
    assert metrics["count"] == 8
    assert metrics["p50_latency_ms"] > 0.0
    assert metrics["p95_latency_ms"] >= metrics["p50_latency_ms"]
    assert metrics["verdict_counts"] == {
        "commit": 4,
        "provisional": 2,
        "escrow": 1,
        "reject": 1,
    }
    assert metrics["reversal_completeness"]["reversed_count"] == 2
    app.state.firewall.ledger.close()


@pytest.mark.asyncio
async def test_file_backed_restart_preserves_budget_idempotency_and_retraction(
    tmp_path,
) -> None:
    db_path = tmp_path / "restart.db"
    baseline = demo_steps()[0]
    derivative = demo_steps()[6]
    assert baseline.raw_text and baseline.evidence_ir
    assert derivative.raw_text and derivative.evidence_ir

    first = create_app(db_path=db_path, seed_demo=True).state.firewall
    baseline_submission = EvidenceSubmission(
        raw_text=baseline.raw_text,
        source_id=baseline.evidence_ir.source_id,
        experiment_id=baseline.evidence_ir.experiment_id,
        root_experiment_id=baseline.evidence_ir.root_experiment_id,
        idempotency_key="restart-baseline",
    )
    original = await first.submit(baseline_submission)
    spent = first.engine.spent_for_root(baseline.evidence_ir.root_experiment_id)
    assert spent > 0.0
    first.close()

    second = create_app(db_path=db_path, seed_demo=True).state.firewall
    assert second.engine.spent_for_root(baseline.evidence_ir.root_experiment_id) == spent
    assert await second.submit(baseline_submission) == original
    await second.retract(baseline.evidence_ir.experiment_id)
    second.close()

    third = create_app(db_path=db_path, seed_demo=True).state.firewall
    assert baseline.evidence_ir.root_experiment_id in third._retracted_roots
    assert third.engine.spent_for_root(baseline.evidence_ir.root_experiment_id) == 0.0
    rejected = await third.submit(
        EvidenceSubmission(
            raw_text=derivative.raw_text,
            source_id=derivative.evidence_ir.source_id,
            experiment_id=derivative.evidence_ir.experiment_id,
            root_experiment_id=derivative.evidence_ir.root_experiment_id,
            idempotency_key="restart-derivative",
        )
    )
    assert rejected["monitor"]["verdict"] == "reject"
    assert "provenance:root_retracted" in rejected["monitor"]["reasons"]
    third.close()


def test_environment_factory_wires_distinct_live_provider_lanes(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EPISTEMICOS_PROVIDER_MODE", "gemini")
    monkeypatch.setenv("EPISTEMICOS_DB_PATH", str(tmp_path / "live-smoke.db"))
    monkeypatch.setenv("EPISTEMICOS_FAST_MODEL", "fast-test-model")
    monkeypatch.setenv("EPISTEMICOS_ESCALATION_MODEL", "escalation-test-model")
    app = create_app_from_env()
    firewall = app.state.firewall

    assert isinstance(firewall.compiler, GeminiCompiler)
    assert isinstance(firewall.escalation_compiler, GeminiCompiler)
    assert isinstance(firewall.embedder, GeminiEmbedder)
    assert firewall.compiler.model == "fast-test-model"
    assert firewall.escalation_compiler.model == "escalation-test-model"
    firewall.close()
