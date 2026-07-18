# GROUND TRUTH design

## What was added

`starter/my_solution.py` now contains a deterministic online belief-revision
policy. `ingest` never writes to `GraphView`; the only state changes it can cause
are attributed, closed-vocabulary `Delta` objects. `item.tag` and all external
state are ignored. This implementation is deliberately rule-based and uses only
the Python standard library.

## Trust boundary and decision order

The evidence body is an untrusted description. It is used only to route the item
to graph states, claims, transitions, or property axes. It never supplies
replication counts, authority, confidence values, claim IDs, or executable
instructions. Structured `item.provenance` is the trusted evidence channel;
`view` is read-only graph context.

Because update *magnitude* is derived only from structured provenance and never
from the body, a mutation driven by injected text is impossible by construction:
no field an attacker controls feeds a `Delta` payload. The instruction detector in
step 1 is defense-in-depth that keeps injection items as clean no-ops; the firewall
itself does not depend on it.

Every item follows this order:

1. Reject a missing, oversized, or instruction-like body with an attributed
   no-op. The detector covers processor/system prompts, provenance overrides,
   API/delta names, imperative graph edits, and claim-confidence assignments.
2. Normalize prose and resolve CamelCase, plural, acronym, and common natural
   language state names through `GraphView`.
3. Convert provenance to a bounded quality score:
   `0.40*groups + 0.15*replication + 0.20*directness + 0.15*effect +
   0.10*method_reliability`. Count values saturate at 1, 2, 3, and 4+ sources.
   Evidence is credible only when score ≥ 0.60 and at least two independent
   groups are present; it is strong at score ≥ 0.82 and at least three groups.
   One group plus one replication is always thin, regardless of prose claims.
4. Resolve trusted retractions or failed replications before interpreting the
   original phenomenon. Only a matching pending item is dropped. An explicit
   structured null effect vetoes a positive body claim.
5. Detect OOD evidence. A conversion between two equal-potency, cross-lineage
   states proposes the declared lateral-conversion regime (recognized from
   explicit no-intermediate wording or from the state geometry itself).
   Identity-preserving changes in age, function, chromatin, transcriptional,
   metabolic, or morphology axes propose an axis. These cases do not revise
   unrelated beliefs. Failed or retracted evidence cannot create an OOD proposal.
6. Route in-model evidence to mechanism-scoped or general claims. Strong
   contradictions revise confidence down and add a method-based scope exception;
   confirmations use a smaller upward step. Confidence changes use log-odds,
   remain below the API’s three-log-odds cap, and skip meaningless saturated
   writes.

## Out-of-distribution taxonomy

Three cases are separated *before* any contradiction is considered, because a
result that is outside the model must not be scored against a claim it was never
in scope for.

- **In-model contradiction** — a phenomenon the graph represents (a potency
  reversal, a nuclear-transfer result). It revises the targeted claim; it is not
  flagged.
- **Out-of-model regime** — a conversion between two states at the *same* potency
  but *different* lineages. That is geometrically neither a potency step nor an
  adjacency, so the graph cannot express it; we `propose_regime` and do not refute
  the adjacency claim. This is recognized either from explicit "direct / no
  intermediate" wording or, structurally, from the equal-potency cross-lineage
  geometry of the two mentioned states, so unseen phrasing is still caught.
- **Out-of-model axis** — a property the graph does not track (age, function,
  chromatin, morphology) changing while identity is held fixed. We `propose_axis`.

Precision is protected by the *identity-preserved* and equal-potency guards: the
near-miss case, a within-lineage move along the potency axis, changes potency and
keeps its lineage, so it fails both OOD tests and is correctly revised in-model
rather than flagged. A named intermediate suppresses the regime flag, since it
signals an in-model potency contradiction instead.

## Calibration

Updates are applied in log-odds, which is bounded, symmetric, and keeps each step
below the API's three-log-odds cap. The magnitude is a function of the structured
quality score alone, so the trajectory has the intended shape: a single weak or
single-group result moves nothing (an update requires score ≥ 0.60 and at least
two independent groups); a confirming result of an established belief nudges it up
slightly; a strong, replicated contradiction (score ≥ 0.82, ≥ 3 groups) moves the
targeted belief a lot *and* adds a mechanism-scoped exception, so the belief is
narrowed rather than deleted. Contradictions are also scaled by reported effect
strength; confirmations of a genuinely contested claim (confidence < 0.70) are
allowed a larger step than confirmations of a near-saturated one.

## Skepticism and pending evidence

An extraordinary contradiction with one source is held under a stable identifier
such as `pending__env_stress__C3d`, without changing the claim. Resolution is
matched *by the claim the pending item concerns*, not by method class, since an
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
