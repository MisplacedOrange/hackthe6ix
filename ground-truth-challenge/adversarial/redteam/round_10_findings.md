# Red-Team Round 10 — Sign flip via failed-replication contamination across comma-joined clauses

**Target:** authoritative repo-root `starter/my_solution.py` + `epistemic_core.py`.
**Status of this directory:** the `adversarial/redteam/` tree (rounds 01–09, `run_round.py`, `README.md` ledger) was deleted mid-session during a repo cleanup; only this round survives. The repro below is therefore **self-contained** and does not depend on the removed runner.

**Result:** 2 attacks land (sign inversion); 2 controls correct.

## Reproduction (self-contained)

```bash
cd <repo-root>
python - <<'PY'
import importlib.util, sys
sys.path.insert(0, '.')
s = importlib.util.spec_from_file_location('sol', 'starter/my_solution.py')
m = importlib.util.module_from_spec(s); sys.modules['sol']=m; s.loader.exec_module(m)
from groundtruth.loader import load_seed
from groundtruth.model import GraphView
from groundtruth.ingest import EvidenceItem
prov = {'replication_count':'many','independent_groups':4,'method_class':'defined_factor_perturbation',
        'method_directness':'direct','effect_strength':'strong','retraction_status':'none'}
def run(label, body):
    g = load_seed(); b = g.claims['C3c'].confidence
    r = m.ingest(EvidenceItem('X','',body,prov,''), GraphView(g))
    nc = [round(d.payload['new_confidence'],3) for d in r.deltas
          if d.op=='revise_confidence' and d.payload.get('claim_id')=='C3c']
    print(f'{label}: C3c {b} -> {nc}')
run('baseline reversal (want DOWN ~0.43)', 'A defined factor returned fibroblasts to a pluripotent state.')
run('comma failrep (FLIP -> UP 0.947)', 'A defined factor returned fibroblasts to a pluripotent state, though an unrelated control assay failed to replicate.')
run('semicolon failrep (safe -> DOWN 0.43)', 'A defined factor returned fibroblasts to a pluripotent state; a parallel staining protocol failed to replicate.')
PY
```

Observed:

| Body | `C3c` 0.92 → | |
|------|--------------|---|
| "… returned to a pluripotent state." | 0.43 (down) | correct |
| "… returned to a pluripotent state**, though an unrelated control assay failed to replicate.**" | **0.947 (up)** | **FLIP** |
| "… returned to a pluripotent state**, whereas a parallel staining protocol could not reproduce.**" | **0.947 (up)** | **FLIP** |
| "… returned to a pluripotent state**;** a parallel staining protocol failed to replicate." | 0.43 (down) | correct (semicolon) |

---

## The finding

A strong, replicated reversal (a contradiction that should lower `C3c` to 0.43) is turned into a **confidence increase** (0.947) by comma-joining a failed-replication remark about a *different* experiment into the same sentence. The failed-replication statement is not about the reversal — it concerns "an unrelated control assay" / "a parallel staining protocol" — but it captures the whole event.

### Why it fails

`_split_clauses` segments only on sentence terminators and semicolons:

```python
parts = re.split(r"(?<=[.!?;])\s+|\s*;\s*", text)
```

Commas are **not** split points, so *"A defined factor returned fibroblasts to a pluripotent state, though an unrelated control assay failed to replicate"* is a **single clause**. `_failed_replication` then scans that entire clause:

```python
def _failed_replication(clause):
    match = re.search(r"\b(?:failed to replicate|failed to reproduce|could not replicate|"
                      r"could not reproduce|did not replicate|...)\b", clause)
    ...
```

It matches the injected phrase, so `event.failed_replication = True` for the reversal event. `ingest` then routes on `if quality.retracted or event.failed_replication:` and `_targets` maps a failed-replication potency-reversal to `(scoped_claim, +1, "failed replication supports the scoped prior")` — a **support**. The contradiction is scored as its own failure to replicate and *raises* the belief it refutes. The semicolon control proves the mechanism: the identical remark, separated into its own clause, does not contaminate the reversal, and `C3c` correctly falls to 0.43.

### Why it's exploitable (and dual-use)

The body is attacker-controlled, so appending *", though an unrelated assay failed to replicate"* to any real contradiction inverts its sign. It is also a latent accuracy bug: a single evidence item that legitimately reports a positive result **and** a negative control ("the effect reproduced, though the antibody control failed to replicate") — routine in real papers — would be scored backwards. This is the third distinct sign/action manipulation in the open set, all reached by a comma.

### Root cause

**Two clause-scoped predicates are computed over a unit that is not actually a single clause.** `_failed_replication` (and, in round 9, `_denied_near`) assume the text they scan is one atomic assertion, but `_split_clauses` under-segments — it never breaks on the comma/coordination boundaries (", though", ", whereas", ", even as", ", although") that separate independent propositions in scientific prose. So a predicate about experiment B is attributed to experiment A. The failure is the same shape as R9 (polarity from unscoped token presence), now via a *different* predicate (`failed_replication` instead of `denied`) exploiting the *same* segmentation gap.

### What to learn / fix

- **Segment on coordination, not just terminators.** Split (or at least sub-scope) at comma + concessive/contrastive connectives ("though," "although," "whereas," "even as," "while," "but") so a failed-replication or denial predicate binds only to its own proposition.
- **Bind `failed_replication` to the event's predicate, not the whole clause.** The failed-replication signal should apply only when it governs the *same* transition/subject as the primary event, not any co-occurring sentence fragment.
- **A "failed replication" that concerns a different subject than the primary result must not flip the primary result.** Cross-check the subject of the failed-replication phrase (its nearest state/experiment) against the event's source/destination; if they differ, treat it as a separate (or background) event.
- **Meta (rounds 9–10):** every predicate that decides *polarity or eligibility* must operate on a correctly-scoped sub-clause. Under-segmentation is now the shared enabler of two sign-flips and (via speech-act) one suppression — fixing clause segmentation is the highest-leverage single change.

---

## Open-findings status (as re-derived this session; prior ledger was deleted)

- **R7** — abstention-as-suppression (append a 2nd eligible event → `no_op`). Open.
- **R8** — speech-act framing suppression ("We hypothesized…/As background…" → ineligible). Open.
- **R9** — litotes sign flip ("it cannot be denied that …" → RAISES the refuted claim). Open.
- **R10 (this round)** — failed-replication contamination across a comma → sign flip. Open.
- R1–R6 were fixed by the successive rewrites (verified in rounds 7–8 before the ledger was removed).

R9 and R10 are the same root cause (clause under-segmentation → an unscoped predicate flips the sign); R7 and R8 are the sibling "body controls whether the system acts." All four are closed by the same discipline: **scoped, per-proposition parsing, with act/abstain and polarity taken from the strongest trusted signal actually governing the event — never from a token the body can add anywhere in the sentence.**
