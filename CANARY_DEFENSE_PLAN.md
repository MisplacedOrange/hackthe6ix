# Canary Defense Plan

## Implementation status

- **Phase 1 (exact authorization): DONE.** `_authorize` in
  `starter/my_solution.py` is an independent reference monitor over every
  returned delta; unauthorized deltas collapse the result to an attributed
  `no_op`. Covered by `tests/test_gemini_policy.py` and the regression suite.
- **Phase 2 (control canaries): DONE.** `adversarial/control_canary.py` runs a
  versioned clean-vs-attacked corpus offline and compares exact structured
  decisions.
- **Phase 3 (drift reporting): DONE.** The same harness reports pass/fail
  counts, unexpected-mutation count, and repeatability failures; the release
  gate (zero unexpected mutations) is enforced by
  `tests/test_control_canary.py`.
- **Deferred defenses (perplexity/layer-variance, honeypots): NOT started**, by
  design — they require infrastructure the challenge does not have.

## Decision

Keep the scored ingestion path small and deterministic. Add one defense next:
an offline control-canary check that compares exact decisions for clean evidence
and manipulated variants. Do not add layer-activation monitoring or honeypot
documents until the product has infrastructure that can support them.

## Goals

- Prevent untrusted text from changing claim targets, confidence calculations,
  scope, pending state, or OOD proposals.
- Detect behavior drift without adding a second runtime model or network call.
- Preserve deterministic, fail-closed judging behavior.
- Keep the implementation inside the existing challenge contract.

## Phase 1: Exact authorization

Replace broad mutation permission with an immutable authorization record computed
from the parsed scientific event, trusted provenance, and current graph.

For each evidence item, authorize only:

- the exact claim IDs that may change;
- the exact expected posterior confidence values;
- the exact scope write, if any;
- the exact pending action, if any;
- zero or one exact OOD axis or regime proposal; or
- a single `no_op` when no mutation is justified.

Validate the complete returned delta list against this record before returning it.
Any mismatch must become an attributed `no_op`.

## Phase 2: Control canaries

Create a small versioned corpus of paired inputs:

1. a clean scientific evidence item;
2. the same item with an injected instruction, provenance boast, claim ID,
   Unicode disguise, or unrelated persuasive suffix.

Run both through the deterministic policy from the same graph snapshot. Compare
exact structured outcomes rather than rationale text or embedding similarity.

Required invariants:

- injected text cannot introduce a mutation;
- body text cannot alter structured provenance weight;
- body claim IDs cannot select mutation targets;
- an attacked item either preserves the clean scientific decision after safe
  taint removal or fails closed to `no_op`;
- OOD status and proposals cannot be selected by embedded instructions;
- repeated runs produce identical deltas.

Keep this check offline. It must not run inside `ingest` and must not require an
API key.

## Phase 3: Drift reporting

Store the expected structured result for every control pair and report:

- exact pass/fail counts;
- changed target, direction, posterior, scope, pending action, or OOD proposal;
- unexpected mutation count;
- deterministic repeatability failures.

The release gate is zero unexpected mutations. A semantic similarity score is
informational only and cannot override an exact mismatch.

## Deferred defenses

### High-perplexity or layer-variance detection

Defer until inference runs on a model that exposes stable token likelihoods or
internal activations. Unusual scientific language is not itself an attack, so
this signal would be advisory rather than an authorization gate.

### Honeypot documents

Defer until the system has a retrieval index. If retrieval is later added, seed
non-authoritative sentinel documents with unique identifiers and alert when they
are retrieved unexpectedly. Never allow a sentinel to influence belief updates.

## Acceptance criteria

- Existing valid evidence decisions remain unchanged.
- Every unauthorized one-field delta mutation fails closed.
- Every control-canary attack produces either the clean authorized decision or
  `no_op`.
- The scored path remains deterministic and standard-library compatible.
- No new model, retrieval system, telemetry service, or runtime dependency is
  introduced.

## Explicit non-goals

- No claim of universal prompt-injection detection.
- No model-layer monitoring through a black-box API.
- No fake retrieval documents before retrieval exists.
- No model voting, confidence arbitration, or model-generated deltas.
