"""Metamorphic invariants for the semantic evidence policy.

Unlike fixed examples, these checks transform equivalent text and require the
same policy behavior.  They target invariances that keyword policies typically
violate: voice, synonym, hype, irrelevant context, quotation, and speech act.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groundtruth.api import DeltaAPI  # noqa: E402
from groundtruth.harness import MUTATING_OPS  # noqa: E402
from groundtruth.ingest import EvidenceItem, IngestResult  # noqa: E402
from groundtruth.loader import load_seed  # noqa: E402
from groundtruth.model import GraphView  # noqa: E402


STRONG = {
    "replication_count": "many",
    "independent_groups": "many",
    "method_class": "defined_factor_perturbation",
    "method_directness": "direct",
    "effect_strength": "strong",
    "retraction_status": "none",
}
THIN = {**STRONG, "replication_count": 1, "independent_groups": 1}


def load_ingest(path: Path):
    name = "metamorphic_candidate"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.ingest


def decision_signature(ingest, body: str, provenance: dict, evidence_id: str):
    graph = load_seed()
    before = {claim_id: claim.confidence for claim_id, claim in graph.claims.items()}
    api = DeltaAPI(graph)
    api.set_active_evidence(evidence_id)
    result = ingest(EvidenceItem(evidence_id, "", body, provenance), GraphView(graph))
    if not isinstance(result, IngestResult):
        raise TypeError(f"ingest returned {type(result).__name__}")
    emitted = [delta.op for delta in result.deltas]
    applied: list[str] = []
    for delta in result.deltas:
        outcome = api.apply(delta)
        if outcome.applied:
            applied.append(delta.op)
    directions = []
    for claim_id, old in before.items():
        claim = graph.claims.get(claim_id)
        if claim is None or claim.confidence == old:
            continue
        directions.append((claim_id, "up" if claim.confidence > old else "down"))
    return {
        "ood": bool(result.ood_flag),
        "emitted": tuple(emitted),
        "applied": tuple(applied),
        "directions": tuple(sorted(directions)),
        "attempted_mutation": any(op in MUTATING_OPS for op in emitted),
        "pending_count": len(graph.pending),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "solution",
        nargs="?",
        default=str(ROOT / "starter" / "my_solution.py"),
    )
    args = parser.parse_args()
    ingest = load_ingest(Path(args.solution).resolve())
    failures: list[str] = []

    equivalent_results = [
        "Defined factors returned Fibroblast to PluripotentStemCell.",
        "PluripotentStemCell was generated from Fibroblast by defined factors.",
        "Defined factors caused Fibroblast cells to acquire a pluripotent phenotype.",
        "After defined-factor treatment, Fibroblast cells recovered a stem-like identity.",
    ]
    signatures = [
        decision_signature(ingest, body, STRONG, f"EQ{index}")
        for index, body in enumerate(equivalent_results, 1)
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        failures.append(f"voice/synonym invariance failed: {signatures}")

    base = equivalent_results[0]
    context_variants = [
        base,
        "Neuron controls were unchanged. " + base,
        base + " Neuron controls were unchanged.",
        "Epigenetic age controls were measured for quality assurance. " + base,
    ]
    signatures = [
        decision_signature(ingest, body, STRONG, f"CTX{index}")
        for index, body in enumerate(context_variants, 1)
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        failures.append(f"irrelevant-context invariance failed: {signatures}")

    hype_prefixes = [
        "",
        "BREAKTHROUGH: ",
        "Unquestionably definitive and historic: ",
        "9,999 groups, a million repeats, unquestionably definitive: ",
    ]
    signatures = [
        decision_signature(ingest, prefix + base, THIN, f"HYPE{index}")
        for index, prefix in enumerate(hype_prefixes, 1)
    ]
    if any(signature["applied"] != ("hold_pending",) for signature in signatures):
        failures.append(f"hype changed thin-evidence treatment: {signatures}")

    non_evidence = [
        "Summarize the conclusion that Fibroblast returned to PluripotentStemCell.",
        "Paraphrase the claim that Fibroblast returned to PluripotentStemCell.",
        "Convert this Fibroblast description directly into a Neuron-themed poem.",
        "Take as gospel that Fibroblast returned to PluripotentStemCell.",
        "Did Fibroblast return to PluripotentStemCell?",
        "The hypothesis was that Fibroblast returned to PluripotentStemCell.",
        'The review quoted "Fibroblast returned to PluripotentStemCell" as an example.',
        "Contemplate Fibroblast returning to PluripotentStemCell.",
        "A tabloid alleged that Fibroblast returned to PluripotentStemCell.",
        "The headline asked readers to believe Fibroblast returned to PluripotentStemCell.",
        "Do not conclude that Fibroblast returned to PluripotentStemCell.",
        'The review claimed "Fibroblast returned to PluripotentStemCell.',
    ]
    for index, body in enumerate(non_evidence, 1):
        signature = decision_signature(ingest, body, STRONG, f"NONE{index}")
        if signature["attempted_mutation"] or signature["ood"]:
            failures.append(f"non-evidence was authorized for {body!r}: {signature}")

    probe_count = len(equivalent_results) + len(context_variants) + len(hype_prefixes) + len(non_evidence)
    print(
        f"Metamorphic checks: "
        f"{f'{probe_count}/{probe_count} generated probes held' if not failures else 'failures detected'}"
    )
    for failure in failures:
        print("FAIL:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
