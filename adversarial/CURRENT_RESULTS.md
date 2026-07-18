# Current semantic-policy validation

Target: `starter/my_solution.py` exporting `starter/semantic_policy.py`

## Results

- Official practice sandbox: **pass**
- Fixed adversarial corpus: **62/62 cases pass**
- Metamorphic suite: **24/24 generated probes pass**
- Structural firewall: all writes remain typed and attributed
- Audit taxonomy: every decision carries a closed `EvidenceClass` label

For comparison, `origin/combined-solution` at `5cc69c6` passes the official
practice sandbox but holds only **13/62** cases in this corpus and fails the
metamorphic suite. Its useful idea—explicit internal evidence classification—was
retained as a typed audit label. Its permissive parser, provenance coercions,
coarse pending IDs, and whole-document OOD scans were not retained.

The original keyword-heavy policy passed 12/39 adversarial cases. The recent
nested keyword revision passed 14/39. The semantic policy passes the original 39
cases plus 23 new cases covering instruction-shaped requests, mixed-clause
laundering, reported allegations, unseen imperative verbs, synonym equivalence,
contradictory clauses, malformed provenance, hype, quotations, hypotheses,
missing event roles, ambiguous retractions, and invalidation-before-validation.

Passing this corpus is evidence of improvement, not a proof of perfect natural-
language security. Adaptive red-teaming and authenticated provenance remain
necessary.

## Reproduce

```powershell
python selfcheck.py
python adversarial/run_adversarial.py starter/my_solution.py
python adversarial/metamorphic_checks.py starter/my_solution.py
```
