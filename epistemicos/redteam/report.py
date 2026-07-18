"""Provider-free G10 security report aggregation over the seeded corpus.

The report measures deterministic containment. It deliberately does not claim
that prompt injection or semantic fabrication can be detected perfectly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from api.main import create_app
from core.engine import EPS, ROOT_BUDGET, bernoulli_kl
from core.escalate import BURST_THRESHOLD
from core.ledger import EventType
from core.types import EvidenceSubmission, Integrity
from llm.compiler import FakeCompiler
from llm.embedder import FakeEmbedder
from redteam.generate import load_corpus

_REGISTERED_CATEGORIES = frozenset({"duplicate_flood", "slow_drip"})


def _new_firewall() -> Any:
    app = create_app(seed_demo=False)
    firewall = app.state.firewall
    firewall.ledger.append(
        EventType.CLAIM_REGISTERED,
        {"claim_id": "C17", "text": "Red-team report claim", "prior": 0.5},
    )
    return firewall


def _register_cases(firewall: Any, cases: list[dict[str, Any]]) -> None:
    for case in cases:
        envelope = case["submission"]
        firewall.registry.register_experiment(
            envelope["experiment_id"],
            source_id=envelope["source_id"],
            root_experiment_id=envelope["root_experiment_id"],
            claim_ids={"C17"},
        )


def _submission(case: dict[str, Any]) -> EvidenceSubmission:
    return EvidenceSubmission(raw_text=case["raw_text"], **case["submission"])


def _trace(
    case: dict[str, Any],
    result: dict[str, Any],
    event: Any,
    *,
    root_spent: float,
    authorized_provenance: bool,
) -> dict[str, Any]:
    engine = result.get("engine")
    event_kl = 0.0
    if engine is not None:
        event_kl = bernoulli_kl(float(engine["posterior"]), float(engine["prior"]))
    return {
        "case_id": case["id"],
        "category": case["category"],
        "verdict": result["monitor"]["verdict"],
        "integrity": int(result["monitor"]["integrity"]),
        "reasons": list(result["monitor"]["reasons"]),
        "committed_delta": float(event.payload.get("delta", 0.0)),
        "event_kl": event_kl,
        "root_spent": float(root_spent),
        "escalated": int(result["metrics"].get("escalated", 0)),
        "event_seq": int(result["event_seq"]),
        "authorized_provenance": authorized_provenance,
    }


def _failure_reasons(case: dict[str, Any], trace: dict[str, Any]) -> list[str]:
    category = case["category"]
    failures: list[str] = []
    verdict = str(trace["verdict"])
    integrity = int(trace["integrity"])
    delta = float(trace["committed_delta"])

    allowed = {item.lower() for item in case["expected_containment"]["allowed_verdicts"]}
    if verdict not in allowed:
        failures.append("verdict_outside_expected_set")
    if verdict == "reject" and not trace["reasons"]:
        failures.append("rejection_missing_reasons")

    if case["expected_containment"]["threat_tier"] == "T1":
        if verdict != "reject" or integrity != int(Integrity.L0_RAW) or delta != 0.0:
            failures.append("t1_influence_or_non_rejection")
    elif category == "fake_replication":
        if integrity >= int(Integrity.L3_REPLICATED) or verdict == "commit":
            failures.append("unverified_replication_promoted")
    elif category == "duplicate_flood":
        if integrity > int(Integrity.L2_VERIFIED):
            failures.append("duplicate_gained_unauthorized_integrity")
        if float(trace["root_spent"]) > ROOT_BUDGET + 1e-12:
            failures.append("root_budget_exceeded")
    elif category == "slow_drip":
        if float(trace["event_kl"]) > EPS[Integrity.L2_VERIFIED] + 1e-12:
            failures.append("per_event_kl_exceeded")
        sequence_index = int(case["expected_containment"]["limits"]["sequence_index"])
        if sequence_index > BURST_THRESHOLD and not any(
            str(reason).startswith("source_burst:") for reason in trace["reasons"]
        ):
            failures.append("source_burst_not_escalated")
    elif category == "schema_valid_semantic_lie":
        if verdict not in {"reject", "provisional", "escrow"}:
            failures.append("semantic_lie_committed")
        if integrity > int(Integrity.L1_PARSED) or delta != 0.0:
            failures.append("semantic_lie_gained_influence")

    if "<<IR>>" in case["raw_text"] and not trace["authorized_provenance"]:
        if integrity > int(Integrity.L1_PARSED):
            failures.append("unregistered_structure_gained_integrity")
    return failures


async def _run_phase(
    cases: list[dict[str, Any]], *, register_provenance: bool
) -> list[dict[str, Any]]:
    firewall = _new_firewall()
    if register_provenance:
        _register_cases(firewall, cases)
    traces: list[dict[str, Any]] = []
    try:
        if not isinstance(firewall.compiler, FakeCompiler) or not isinstance(
            firewall.embedder, FakeEmbedder
        ):
            raise RuntimeError("security report requires fake providers")
        for case in cases:
            result = await firewall.submit(_submission(case))
            event = firewall.ledger.events()[-1]
            root_id = case["submission"]["root_experiment_id"]
            trace = _trace(
                case,
                result,
                event,
                root_spent=firewall.engine.spent_for_root(root_id),
                authorized_provenance=register_provenance,
            )
            trace["failure_reasons"] = _failure_reasons(case, trace)
            trace["passed"] = not trace["failure_reasons"]
            traces.append(trace)
        valid, reason = firewall.ledger.verify_chain()
        if not valid:
            raise RuntimeError(f"report ledger verification failed: {reason}")
        return traces
    finally:
        firewall.ledger.close()


def _aggregate(corpus: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "failed": 0}
    )
    for trace in traces:
        bucket = by_category[trace["category"]]
        bucket["total"] += 1
        if not trace["passed"]:
            bucket["failed"] += 1

    unauthorized = sum(
        1
        for trace in traces
        if trace["verdict"] == "commit" and not trace["authorized_provenance"]
    )
    failed = sum(1 for trace in traces if not trace["passed"])

    representative: list[dict[str, Any]] = []
    represented: set[str] = set()
    for trace in traces:
        if trace["category"] not in represented or not trace["passed"]:
            representative.append(trace)
            represented.add(trace["category"])

    return {
        "corpus_version": corpus["version"],
        "seed": corpus["seed"],
        "provider_mode": "fake",
        "disclaimer": corpus["disclaimer"],
        "failure_counts": {
            "total_cases": len(traces),
            "passed_cases": len(traces) - failed,
            "failed_cases": failed,
            "by_category": dict(by_category),
        },
        "unauthorized_transition_count": unauthorized,
        "representative_traces": representative,
        "case_results": traces,
    }


async def run_security_report() -> dict[str, Any]:
    """Run all seeded cases through real firewall orchestration using fakes."""
    corpus = load_corpus()
    cases: list[dict[str, Any]] = corpus["cases"]
    order = {case["id"]: index for index, case in enumerate(cases)}

    unregistered = [
        case for case in cases if case["category"] not in _REGISTERED_CATEGORIES
    ]
    duplicate = [case for case in cases if case["category"] == "duplicate_flood"]
    slow_drip = [case for case in cases if case["category"] == "slow_drip"]

    traces = [
        *(await _run_phase(unregistered, register_provenance=False)),
        *(await _run_phase(duplicate, register_provenance=True)),
        *(await _run_phase(slow_drip, register_provenance=True)),
    ]
    traces.sort(key=lambda trace: order[trace["case_id"]])
    return _aggregate(corpus, traces)


__all__ = ["run_security_report"]
