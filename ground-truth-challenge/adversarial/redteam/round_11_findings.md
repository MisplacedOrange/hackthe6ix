# Red-Team Round 11 — Post-predicate denial contamination (the segmentation flip is bidirectional)

**Target:** authoritative repo-root `starter/my_solution.py` + `epistemic_core.py`.
**Directory note:** `adversarial/redteam/` is being repeatedly wiped by a repo cleanup (rounds 01–10 have been deleted more than once mid-session). This finding is therefore captured in full in the run transcript and is verifiable by the **self-contained repro** below, independent of any runner.
**Result:** 2 attacks land (sign flip); 2 controls correct.

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
run('baseline (DOWN 0.43)', 'A defined factor returned fibroblasts to a pluripotent state.')
run('post comma flip (UP 0.947)', 'A defined factor returned fibroblasts to a pluripotent state, but in the sham arm the effect was absent.')
run('semicolon (DOWN 0.43)', 'A defined factor returned fibroblasts to a pluripotent state; in the sham arm the effect was absent.')
PY
```

---

## The finding

Rounds 9 and 10 flipped a contradiction's sign with distractor text placed **before** the predicate (a litotes frame; a comma-joined failed-replication). Round 11 shows the same flip with distractor text placed **after** the predicate:

- *"A defined factor returned fibroblasts to a pluripotent state**, but in the sham arm the effect was absent.**"* → `C3c` 0.92 → **0.947 (up)**.
- *"… returned to a pluripotent state**, although no such transition occurred in vehicle controls.**"* → **0.947 (up)**.

Both should lower `C3c` to 0.43; the semicolon-isolated form does exactly that.

### Why it fails

`_denied_near` evaluates a **bidirectional** window around the predicate:

```python
def _denied_near(clause, predicate_start):
    prefix   = clause[max(0, predicate_start - 90) : predicate_start]
    combined = clause[max(0, predicate_start - 120) : predicate_start + 100]   # <- extends AFTER the predicate
    ...
    if re.search(r"\b(?:no such transition|no transition occurred|effect was absent)\b", combined):
        return True
```

Because `_split_clauses` does not break on commas, the trailing concessive ("…, but … the effect was absent") stays in the same clause, and its text falls inside `combined`'s +100-character post-predicate reach. `_denied_near` returns True, polarity becomes `DENIED`, and the reversal is scored as a denial-of-reversal that **supports** the prior — a sign inversion. The failed-replication phrase about the sham arm concerns a *different* experimental condition than the primary reversal, but the window does not care which subject it belongs to.

### Why this specifically matters for the fix

R9/R10 might tempt a narrow patch: "only look at negation cues *before* the predicate." Round 11 pre-empts that — the `combined` window already reaches after the predicate, and trailing concessives are at least as natural in scientific prose as leading ones ("X occurred, but the control showed no effect"). So the contamination is **bidirectional**, and any window-based fix (front, back, or both) is defeated by placing the distractor on the unguarded side or by widening the sentence. The only robust fix is structural segmentation, not window tuning.

### Root cause

Same as R9/R10, made sharper: **polarity is decided by token presence inside a fixed character window, and the window is not a clause.** `_split_clauses` under-segments (no comma/coordination boundaries), so `_denied_near` scans across independent propositions in both directions. The predicate that decides the *sign of a graph mutation* is computed over text that belongs to a different experiment.

### What to learn / fix

- **Segment on coordination before any polarity/eligibility predicate runs.** Split at `, but` / `, although` / `, though` / `, whereas` / `, while` / `, even as` (and their leading-clause forms), so denial and failed-replication cues bind only to their own proposition. This single change closes R9, R10, and R11 together.
- **Do not use character windows as a proxy for scope.** Replace the ±90/±100 windows with sub-clause membership: a cue counts only if it governs the predicate's own sub-clause.
- **Subject-check negation.** "the effect was absent" / "no such transition" should flip a result only when its nearest subject matches the primary event's subject (here it is the *sham/vehicle* arm, not the treated fibroblasts).
- **When sign is ambiguous, hold.** A clause containing both an affirmed reversal and a negated control is genuinely two facts; abstaining or splitting is correct, silently flipping to support is not.

---

## Open-findings status (re-derived; prior ledger repeatedly deleted)

| # | Finding | Class | Status |
|---|---------|-------|--------|
| R7 | append a 2nd eligible event → total abstention | suppression | open |
| R8 | "We hypothesized…/As background…" framing → ineligible | suppression | open |
| R9 | litotes "cannot be denied that …" → RAISES refuted claim | sign flip (pre-predicate) | open |
| R10 | comma-joined "…failed to replicate" → RAISES | sign flip (pre-predicate) | open |
| R11 | trailing "…, but the effect was absent" → RAISES | sign flip (post-predicate) | open |
| R1–R6 | extraction meaning + cumulative accumulation | — | fixed by rewrites |

**R9, R10, R11 are one bug** (clause under-segmentation feeding an unscoped, bidirectional polarity/failed-replication predicate) with three triggers and both directions — collectively the strongest evidence that **coordination-aware clause segmentation is the single highest-leverage fix**. R7/R8 are the sibling "body controls whether the system acts," closed by per-event authorization and assertion-dominant speech-act classification.
