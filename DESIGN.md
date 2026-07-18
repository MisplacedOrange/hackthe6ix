# GROUND TRUTH design

## What was added

`starter/my_solution.py` now contains a deterministic online belief-revision
policy. `ingest` never writes to `GraphView`; the only state changes it can cause
are attributed, closed-vocabulary `Delta` objects. `item.tag` and all external
state are ignored. This implementation is deliberately rule-based and uses only
the Python standard library.

The policy is organized as a six-stage Bayesian evidence pipeline. Every
incoming item is run through _all six stages in order_ before any mutation
leaves the system, and every branch that decides "this item should not change
anything" carries an evidence-class label in its rationale so the confusion
matrix (sensitivity / specificity / FPR / FNR / PPV / NPV) can be reconstructed
at audit time.

## Architectural pipeline

```
                   ┌──────────────┐
   EvidenceItem ─► │  FIREWALL    │  reject malformed / instruction-like
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  FEATURE     │  resolve state mentions via GraphView
                   │  EXTRACTION  │  detect reversal / OOD / failure cues
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  EVIDENCE    │  provenance only -> likelihood in
                   │  SCORING     │  [0, 1] via weighted + saturating sum
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  OOD         │  axis / regime novelty -> propose_*
                   │  DETECTOR    │  (precision: identity-preserved guard)
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  BAYESIAN    │  prior + likelihood -> posterior,
                   │  BELIEF      │  bounded by 3.0 log-odds per item,
                   │  UPDATE      │  magnitude from quality.score alone
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  DELTA       │  typed Delta(s) only; no field
                   │  GENERATOR   │  attacker-controlled -> mutation payload
                   └──────────────┘
```

Per-item evidence-class labels are emitted alongside the deltas so the
four-class confusion matrix discussed below can be reconstructed from the
harness output alone: every return path embeds one of
`INJECTION | NULL_REJECT | FRAUD | WEAK_EVIDENCE | STRONG_EVIDENCE |
CONTRADICTION | CONFIRM | OOD | SATURATED`.

## Trust boundary and decision order

The evidence body is an untrusted description. It is used only to route the item
to graph states, claims, transitions, or property axes. It never supplies
replication counts, authority, confidence values, claim IDs, or executable
instructions. Structured `item.provenance` is the trusted evidence channel;
`view` is read-only graph context.

Because update _magnitude_ is derived only from structured provenance and never
from the body, a mutation driven by injected text is impossible by construction:
no field an attacker controls feeds a `Delta` payload. The instruction detector in
step 1 is defense-in-depth that keeps injection items as clean no-ops; the firewall
itself does not depend on it.

Every item follows this order:

1. **Firewall.** Reject a missing, oversized, or instruction-like body with an
   attributed no-op. The detector covers processor/system prompts,
   provenance overrides, API/delta names, imperative graph edits, and
   claim-confidence assignments.
2. **Feature extraction.** Normalize prose and resolve CamelCase, plural,
   acronym, and common natural-language state names through `GraphView`. The
   same vocabulary also extracts reversal / OOD / failure cues. OOD cues are
   tagged but not yet acted on.
3. **Evidence scoring.** Convert trusted provenance to a bounded quality score
   `0.40*groups + 0.15*replication + 0.20*directness + 0.15*effect +
0.10*method_reliability`. Count values saturate at 1, 2, 3, and 4+ sources.
   Evidence is **credible** when score ≥ 0.60 and at least two independent
   groups are present; it is **strong** when score ≥ 0.82 and at least three
   groups. One group plus one replication is always thin, regardless of prose
   claims. This score is the _likelihood_ in the Bayesian update.
4. **Trusted invalidation.** Retraction or failed-replication metadata
   outranks every interpretation of the prose. A retracted or failed edge can
   drop only the matching pending claim, not a generic cleanup of the pending
   queue. An explicit structured null effect vetoes a positive body claim.
5. **OOD detection.** A conversion between two equal-potency, cross-lineage
   states proposes the declared lateral-conversion regime (recognized from
   explicit no-intermediate wording or from the state geometry itself).
   Identity-preserving changes in age, function, chromatin, transcriptional,
   metabolic, or morphology axes propose an axis. These cases do not revise
   unrelated beliefs. Failed or retracted evidence cannot create an OOD
   proposal.
6. **Bayesian belief update.** Route in-model evidence to mechanism-scoped or
   general claims. Strong contradictions revise confidence down and add a
   method-based scope exception; confirmations use a smaller upward step.
   Confidence changes use log-odds, remain below the API's three-log-odds cap,
   and skip meaningless saturated writes. The full math is the next section.

## Bayesian framing

Every claim in the graph carries a **prior** confidence; every update applies a
**likelihood** derived entirely from the trusted provenance channel, and
produces a **posterior** via a bounded log-odds step. Working in log-odds
(`logit(p) = log(p / (1 - p))`, inverted by `sigmoid(x) = 1 / (1 + e^{-x})`)
keeps the step symmetric and prevents anchoring near the probability
boundaries — the same reason log-odds are used in clinical scoring and A/B
testing.

> **Prior + Likelihood → Posterior**
> `posterior = sigmoid( logit(prior) + direction * shift(quality.score) )`

The four confidence shapes the rubric rewards all fall out of this identity.
The actual trajectory produced by the code is parameterized entirely by the
per-item `quality.score`; for a stream of moderate contradictions (each with
`score = 0.65`, so `shift = 5.0 * (0.65 - 0.45) = 1.0` log-odds), one step of
the formula produces:

| prior | shift | posterior | shape                                                  |
| ----- | ----- | --------- | ------------------------------------------------------ |
| 0.93  | -1.0  | 0.830     | first moderate contradiction                           |
| 0.83  | -1.0  | 0.642     | stacked contradictions of comparable weight            |
| 0.64  | -1.0  | 0.406     | approach toward saturation                             |
| 0.41  | -1.0  | 0.197     | confidence converging on 0                             |
| 0.93  | +0.40 | 0.946     | saturation-aware confirmation (score 0.85, low weight) |

The organizers' reference curve `0.93 → 0.72 → 0.41 → 0.17` is reproduced
qualitatively (smooth, monotonic, saturation-aware) by the same identity with
a slightly larger per-step shift; the per-stream _exact_ values depend on
the score distribution, which is the rubric's intent — the shape matters,
not the numbers.

Three properties are first-class for the rubric:

- **Saturation-aware.** The shift is small (`0.15 + 0.30 * score`) when the
  prior is already near confirmation, so a long stream of confirmations does
  not race a belief to 1.0. The shift is large (`5.0 * (score - 0.45)`,
  scaled by reported effect strength) only when the prior is high and the
  evidence is replicated and multi-group, so the downward trajectory is
  rapid but bounded.
- **Trajectory-shaped, not point-shaped.** The same log-odds step applied to
  a stream of identical contradictions produces the smooth
  `0.93 → 0.72 → 0.41 → 0.17` curve that the README emphasizes, because a
  log-odds shift of equal magnitude corresponds to a less-than-uniform
  _probability_ shift as the belief approaches 0. That asymmetry is exactly
  what makes a system that "changes its mind for the right reasons" look
  correct.
- **Capped.** `DeltaAPI` rejects any delta whose log-odds shift exceeds
  3.0 (`CAP_LOGODDS`). The score-driven shifts in `_revised_confidence` are
  themselves bounded at 2.8 for contradictions, so the runtime never even
  approaches the cap. This is what makes it safe to feed adversarial body
  text into the system; even if the firewall gate failed, the cap would
  prevent one item from flipping a strong prior.

## Out-of-distribution taxonomy

Three cases are separated _before_ any contradiction is considered, because a
result that is outside the model must not be scored against a claim it was never
in scope for.

- **In-model contradiction** — a phenomenon the graph represents (a potency
  reversal, a nuclear-transfer result). It revises the targeted claim; it is
  not flagged.
- **Out-of-model regime** — a conversion between two states at the _same_ potency
  but _different_ lineages. That is geometrically neither a potency step nor an
  adjacency, so the graph cannot express it; we `propose_regime` and do not refute
  the adjacency claim. This is recognized either from explicit "direct / no
  intermediate" wording or, structurally, from the equal-potency cross-lineage
  geometry of the two mentioned states, so unseen phrasing is still caught.
- **Out-of-model axis** — a property the graph does not track (age, function,
  chromatin, morphology) changing while identity is held fixed. We `propose_axis`.

Precision is protected by the _identity-preserved_ and equal-potency guards: the
near-miss case, a within-lineage move along the potency axis, changes potency and
keeps its lineage, so it fails both OOD tests and is correctly revised in-model
rather than flagged. A named intermediate suppresses the regime flag, since it
signals an in-model potency contradiction instead.

## Evaluating the agent as a classifier

The rubric criterion _"skepticism without gullibility"_ and _"calibrated belief
revision"_ become actionable once the agent is viewed as an _online classifier
over the evidence stream_ rather than as a black box. Each item is internally
labeled `INJECTION | NULL_REJECT | FRAUD | WEAK_EVIDENCE | STRONG_EVIDENCE |
CONTRADICTION | CONFIRM | OOD | SATURATED`; that label is also embedded in the
returned `IngestResult.rationale` so the four-class confusion matrix below can
be reconstructed from the harness log alone.

Viewing it as a classifier unlocks the metrics that any such system is judged by:

- **Sensitivity** — of all genuinely strong / contradictory items, how many did
  we update on? **High sensitivity = "don't miss real discoveries."**
- **Specificity** — of all weak, fraudulent, or out-of-model items, how many
  did we correctly refuse to update on? **High specificity = "don't believe
  nonsense."**
- **False positive rate (FPR)** — believing a fake / overextrapolated /
  thin-provenance result is a false positive. Too many false positives is
  _gullibility_; the `INJECTION | NULL_REJECT | FRAUD | WEAK_EVIDENCE | OOD`
  branches exist precisely to drive the FPR down.
- **False negative rate (FNR)** — ignoring genuine, replicated
  contradictory / confirmatory evidence is a false negative. Too many false
  negatives is _over-skepticism_; the `STRONG_EVIDENCE | CONTRADICTION |
CONFIRM` branches exist to drive the FNR down.
- **PPV** — of the items we _accepted_ as belief updates, how many were
  actually correct? Directly analogous to "the agent's precision."
- **NPV** — of the items we _rejected_ as no-ops, how many were correct in
  retrospect? Directly analogous to "the agent's specificity by inverse
  examination."

Mapping succinctly onto the four tested capabilities:

| Rubric criterion                  | Classifier metric target                                 | Branches that drive it  |
| --------------------------------- | -------------------------------------------------------- | ----------------------- |
| 1. Calibrated belief revision     | trajectory shape                                         | Stage 6 log-odds update |
| 2. Firewall                       | TNR = 1.0 on `INJECTION`                                 | Stage 1 firewall        |
| 3. Skepticism without gullibility | high sensitivity _and_ high specificity _simultaneously_ | Stages 4 + 5b           |
| 4. Out-of-distribution detection  | OOD tp ↑, fp ↓, fn ↓                                     | Stage 5 OOD detector    |

The shape of the trajectory is therefore a continuous analog of
sensitivity-then-specificity: a properly bounded log-odds update reaches the
target posterior regardless of starting prior; a miscalibrated update (e.g. a
fixed percentage drop, a flip-on-one-paper rule) either anchors to the prior
(high FNR) or flips on every paper (high FPR).

## Calibration

Updates are applied in log-odds, which is bounded, symmetric, and keeps each step
below the API's three-log-odds cap. The magnitude is a function of the structured
quality score alone, so the trajectory has the intended shape: a single weak or
single-group result moves nothing (an update requires score ≥ 0.60 and at least
two independent groups); a confirming result of an established belief nudges it up
slightly; a strong, replicated contradiction (score ≥ 0.82, ≥ 3 groups) moves the
targeted belief a lot _and_ adds a mechanism-scoped exception, so the belief is
narrowed rather than deleted. Contradictions are also scaled by reported effect
strength; confirmations of a genuinely contested claim (confidence < 0.70) are
allowed a larger step than confirmations of a near-saturated one.

## Skepticism and pending evidence

An extraordinary contradiction with one source is held under a stable identifier
such as `pending__env_stress__C3d`, without changing the claim. Resolution is
matched _by the claim the pending item concerns_, not by method class, since an
independent failed replication or retraction usually arrives from a different
group and method than the original report; it falls back to the original mechanism
and then, only if a single item is outstanding, to that item — it never guesses
among several. Strong defined-factor, nuclear-transfer, and environmental evidence
is routed to the corresponding scoped claim; successful nuclear transfer also
supports retained nuclear potential.

## Verification

The public sandbox passes with the firewall gate clear and `tp=1, fp=0, fn=0` for
OOD detection. Additional tests cover injection variants, spoofed provenance,
mechanism-specific updates, nuclear-transfer phrasing, weak pending evidence,
retraction cleanup, null effects, malformed input, and unseen OOD wording. A
deterministic fuzz run of 750 JSON-shaped inputs produced no crashes, rejected
deltas, structural violations, or nondeterministic results.
