"""Run the adversarial policy dataset against a candidate ingest function.

Each case starts from a fresh copy of the declared seed graph.  Items within a
case run in order, which lets the dataset exercise pending/retraction behavior
without allowing one unrelated attack to contaminate later results.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groundtruth.api import DeltaAPI  # noqa: E402
from groundtruth.harness import MUTATING_OPS  # noqa: E402
from groundtruth.ingest import EvidenceItem, IngestResult  # noqa: E402
from groundtruth.loader import load_seed  # noqa: E402
from groundtruth.model import GraphView  # noqa: E402


@dataclass
class ItemResult:
    evidence_id: str
    passed: bool
    checks: list[str] = field(default_factory=list)
    emitted_ops: list[str] = field(default_factory=list)
    applied_ops: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    ood: bool = False
    attempted_mutation: bool = False
    confidence_changes: dict[str, tuple[float, float]] = field(default_factory=dict)
    pending_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    error: str | None = None


@dataclass
class CaseResult:
    case_id: str
    title: str
    category: str
    passed: bool
    expected_outcome: str
    items: list[ItemResult] = field(default_factory=list)


def _load_ingest(path: Path):
    spec = importlib.util.spec_from_file_location("adversarial_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ingest = getattr(module, "ingest", None)
    if not callable(ingest):
        raise RuntimeError(f"{path} does not export a callable ingest(item, view)")
    return ingest


def _direction(before: float, after: float, tolerance: float = 1e-9) -> str:
    if after > before + tolerance:
        return "up"
    if after < before - tolerance:
        return "down"
    return "unchanged"


def _check_expectation(
    expectation: dict[str, Any],
    result: ItemResult,
    graph,
    before_confidence: dict[str, float],
) -> None:
    failures: list[str] = []

    if "ood" in expectation and result.ood != expectation["ood"]:
        failures.append(f"OOD expected {expectation['ood']}, got {result.ood}")

    expected_mutation = expectation.get("attempted_mutation")
    if expected_mutation is not None and result.attempted_mutation != expected_mutation:
        failures.append(
            "attempted mutation expected "
            f"{expected_mutation}, got {result.attempted_mutation}"
        )

    required_ops = set(expectation.get("required_ops", []))
    missing_ops = sorted(required_ops.difference(result.applied_ops))
    if missing_ops:
        failures.append(f"missing required applied op(s): {', '.join(missing_ops)}")

    forbidden_ops = set(expectation.get("forbidden_ops", []))
    emitted_forbidden = sorted(forbidden_ops.intersection(result.emitted_ops))
    if emitted_forbidden:
        failures.append(f"emitted forbidden op(s): {', '.join(emitted_forbidden)}")

    for conf_expectation in expectation.get("confidence", []):
        claim_id = conf_expectation["claim_id"]
        expected_direction = conf_expectation["direction"]
        before = before_confidence.get(claim_id)
        claim = graph.claims.get(claim_id)
        if before is None or claim is None:
            failures.append(f"cannot inspect confidence for missing claim {claim_id}")
            continue
        actual_direction = _direction(before, claim.confidence)
        if actual_direction != expected_direction:
            failures.append(
                f"{claim_id} confidence expected {expected_direction}, got "
                f"{actual_direction} ({before:.6f} -> {claim.confidence:.6f})"
            )

    if "pending_count" in expectation:
        expected_count = expectation["pending_count"]
        if len(graph.pending) != expected_count:
            failures.append(
                f"pending count expected {expected_count}, got {len(graph.pending)}"
            )

    for pending_id in expectation.get("pending_contains", []):
        if pending_id not in graph.pending:
            failures.append(f"expected pending item {pending_id!r} is absent")

    for prefix in expectation.get("pending_prefix", []):
        if not any(pending_id.startswith(prefix) for pending_id in graph.pending):
            failures.append(f"no pending item starts with expected prefix {prefix!r}")

    for prefix in expectation.get("pending_absent_prefix", []):
        if any(pending_id.startswith(prefix) for pending_id in graph.pending):
            failures.append(f"a pending item unexpectedly starts with prefix {prefix!r}")

    result.checks = failures
    result.passed = not failures and result.error is None


def run_case(
    case: dict[str, Any],
    ingest,
    seed_path: Path,
    profiles: dict[str, dict[str, Any]],
) -> CaseResult:
    graph = load_seed(str(seed_path))
    api = DeltaAPI(graph)
    item_results: list[ItemResult] = []

    for raw_item in case["items"]:
        evidence_id = raw_item["id"]
        api.set_active_evidence(evidence_id)
        before_confidence = {
            claim_id: claim.confidence for claim_id, claim in graph.claims.items()
        }
        provenance = dict(profiles.get(raw_item.get("profile", ""), {}))
        provenance.update(raw_item.get("provenance", {}))
        public_item = EvidenceItem(
            id=evidence_id,
            tag="",
            body=raw_item["body"],
            provenance=provenance,
            era=raw_item.get("era", ""),
        )
        item_result = ItemResult(evidence_id=evidence_id, passed=False)

        try:
            decision = ingest(public_item, GraphView(graph))
            if not isinstance(decision, IngestResult):
                raise TypeError(
                    f"ingest returned {type(decision).__name__}, expected IngestResult"
                )
            item_result.ood = bool(decision.ood_flag)
            item_result.rationale = decision.rationale
            for delta in decision.deltas:
                item_result.emitted_ops.append(delta.op)
                if delta.op in MUTATING_OPS:
                    item_result.attempted_mutation = True
                apply_result = api.apply(delta)
                if apply_result.applied:
                    item_result.applied_ops.append(delta.op)
                else:
                    item_result.rejected.append(f"{delta.op}: {apply_result.reason}")
        except Exception as exc:  # An adversarial corpus should report crashes, not stop.
            item_result.error = f"{type(exc).__name__}: {exc}"

        for claim_id, before in before_confidence.items():
            claim = graph.claims.get(claim_id)
            if claim is not None and _direction(before, claim.confidence) != "unchanged":
                item_result.confidence_changes[claim_id] = (before, claim.confidence)
        item_result.pending_ids = sorted(graph.pending)

        if item_result.error is None:
            _check_expectation(
                raw_item.get("expect", {}), item_result, graph, before_confidence
            )
        else:
            item_result.checks = [item_result.error]
        item_results.append(item_result)

    return CaseResult(
        case_id=case["id"],
        title=case["title"],
        category=case["category"],
        passed=all(item.passed for item in item_results),
        expected_outcome=case["expected_outcome"],
        items=item_results,
    )


def _format_changes(changes: dict[str, tuple[float, float]]) -> str:
    if not changes:
        return "none"
    return ", ".join(
        f"{claim_id} {before:.3f}->{after:.3f}"
        for claim_id, (before, after) in changes.items()
    )


def print_report(dataset: dict[str, Any], results: list[CaseResult], candidate: Path) -> None:
    passed = sum(result.passed for result in results)
    failed = len(results) - passed
    print(f"Adversarial policy audit: {candidate}")
    print(f"Cases: {len(results)} | policy held: {passed} | policy broke: {failed}")

    categories = sorted({result.category for result in results})
    for category in categories:
        selected = [result for result in results if result.category == category]
        category_passed = sum(result.passed for result in selected)
        print(f"  {category}: {category_passed}/{len(selected)} held")

    print()
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"[{mark}] {result.case_id} - {result.title}")
        print(f"  expected: {result.expected_outcome}")
        for item in result.items:
            print(
                f"  {item.evidence_id}: ood={item.ood}, "
                f"ops={item.applied_ops or ['<none>']}, "
                f"confidence={_format_changes(item.confidence_changes)}, "
                f"pending={item.pending_ids or ['<none>']}"
            )
            if item.rationale:
                print(f"    rationale: {item.rationale}")
            for check in item.checks:
                print(f"    mismatch: {check}")
            for rejection in item.rejected:
                print(f"    API rejection: {rejection}")
            if item.error:
                print(f"    crash: {item.error}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "solution",
        nargs="?",
        default=str(ROOT / "starter" / "my_solution.py"),
        help="path to the candidate Python module",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).with_name("cases.json")),
        help="path to an adversarial cases JSON file",
    )
    args = parser.parse_args()

    candidate = Path(args.solution).resolve()
    dataset_path = Path(args.dataset).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    seed_path = (ROOT / dataset["seed"]).resolve()
    ingest = _load_ingest(candidate)
    profiles = dataset.get("provenance_profiles", {})
    results = [
        run_case(case, ingest, seed_path, profiles) for case in dataset["cases"]
    ]
    print_report(dataset, results, candidate)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
