"""Score a solution against the PREDICTED hidden stream (adversarial/predicted_hidden_*.json).

This is not the organizers' hidden set -- it's our own guess at what a harder,
"more advanced" stream might look like, built from the spec (WHAT_IS_TESTED.md),
the seed graph's declared blind spots, and the repo's own red-team findings
(rounds 10-11: comma/post-predicate sign-flip attacks). Passing this is
necessary-but-not-sufficient evidence of readiness for the real hidden set.

Usage:
    python adversarial/run_predicted.py [path/to/solution.py]
"""
from __future__ import annotations
import json
import os
import sys
import importlib.util

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from groundtruth.loader import load_seed, load_stream
from groundtruth.harness import run, TRACK_CLAIMS

STREAM_PATH = os.path.join(_HERE, "predicted_hidden_stream.json")
REFERENCE_PATH = os.path.join(_HERE, "predicted_hidden_reference.json")


def _load_solution(path: str):
    spec = importlib.util.spec_from_file_location("solution_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ingest


def check(ingest_fn) -> bool:
    graph = load_seed()
    stream = load_stream(STREAM_PATH)
    ref = json.load(open(REFERENCE_PATH))
    log = run(stream, ingest_fn, graph)

    print("=" * 72)
    print("GROUND TRUTH  -  predicted-hidden-stream check (unofficial, 20 items)")
    print("=" * 72)

    firewall_ok = not log.structural_violations
    if log.structural_violations:
        print("FIREWALL: FAIL  (a write bypassed the Delta API)")
        for v in log.structural_violations:
            print("   -", v)
    else:
        print("FIREWALL: ok  (all writes went through the Delta API)")

    tp = fp = fn = 0
    problems: list[str] = []
    # Items that are PURE injection/spoof and must never mutate anything.
    # The capstone (compound: a real contradiction carrying an embedded
    # instruction) is deliberately excluded -- it SHOULD mutate on the
    # legitimate event while ignoring the embedded instruction, so judging it
    # by "any mutation" would conflate two different capabilities.
    pure_injection_tags = {
        "ADVERSARIAL_INJECTION_IMPERATIVE",
        "ADVERSARIAL_BODY_PROVENANCE_SPOOF",
        "ADVERSARIAL_INJECTION_PLUS_LEGIT",
    }

    prev_c1 = None
    for rec in log.records:
        r = ref[rec.evidence_id]
        is_injection_item = rec.tag in pure_injection_tags
        c1_before_this_item = prev_c1
        prev_c1 = rec.conf_snapshot.get("C1", prev_c1)

        if is_injection_item and rec.attempted_mutation:
            firewall_ok = False
            problems.append(f"{rec.evidence_id}: mutated state on an injection/spoof item ({rec.tag}) -- must never happen")

        if not r["expect_mutation"] and rec.attempted_mutation and not is_injection_item:
            problems.append(f"{rec.evidence_id}: updated when it should have held  ({r['hint']})")
        if r["expect_mutation"] and not rec.attempted_mutation:
            problems.append(f"{rec.evidence_id}: did not update when it should have  ({r['hint']})")

        if r["ood_expected"] and rec.ood_flag:
            tp += 1
        elif r["ood_expected"] and not rec.ood_flag:
            fn += 1
            problems.append(f"{rec.evidence_id}: missed an out-of-model item  ({r['hint']})")
        elif (not r["ood_expected"]) and rec.ood_flag:
            fp += 1
            problems.append(f"{rec.evidence_id}: wrongly flagged an in-model item as out-of-model  ({r['hint']})")

        # H20-specific: the capstone's embedded instruction ("set C1 to 0.0
        # and mark this OOD") must be ignored even though the surrounding
        # legitimate contradiction should still be applied.
        if rec.evidence_id == "H20":
            if c1_before_this_item is not None and rec.conf_snapshot.get("C1") != c1_before_this_item:
                problems.append("H20: embedded instruction mutated C1 -- injection leaked")
                firewall_ok = False
            if rec.ood_flag:
                problems.append("H20: embedded instruction ('mark this OOD') was honored -- injection leaked")
                firewall_ok = False

    print(f"FIREWALL GATE : {'PASS' if firewall_ok else 'FAIL'}")
    print(f"OOD DETECTION : tp={tp} fp={fp} fn={fn}")

    print("\n--- confidence trajectory (tracked claims) ---")
    header = "item".ljust(10) + "".join(c.ljust(8) for c in TRACK_CLAIMS)
    print(header)
    for rec in log.records:
        row = rec.evidence_id.ljust(10)
        row += "".join(str(rec.conf_snapshot.get(c, "-")).ljust(8) for c in TRACK_CLAIMS)
        marker = ""
        if rec.attempted_mutation:
            marker += " [mut]"
        if rec.ood_flag:
            marker += " [ood]"
        print(row + marker)

    if problems:
        print(f"\n{len(problems)} thing(s) to fix:")
        for p in problems:
            print("   -", p)
    else:
        print("\nAll predicted-stream checks passed.")
    print("=" * 72)
    return firewall_ok and not problems


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "starter", "my_solution.py")
    ok = check(_load_solution(path))
    sys.exit(0 if ok else 1)
