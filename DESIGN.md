# GROUND TRUTH reasoning design

## Trust boundary

`starter/my_solution.py` is the scored policy. It reads an untrusted evidence
body and a trusted structured provenance channel, then returns only attributed,
closed-vocabulary `Delta` objects. It never writes through `GraphView`, imports
the sidecar, reads `item.tag`, or uses network/model state. `groundtruth/` is
unchanged and remains the only mutation API.

Body text may suggest a clause-local semantic event, but it cannot name a claim,
operation, confidence, scope key, or evidence weight. Unicode normalization,
control/instruction detection, quotation handling, speech-act filtering, and
closed semantic extraction happen before provenance admission. Unknown enums,
invalid types, ambiguous events, and malformed identifiers fail closed as an
attributed `no_op`.

## Calibrated revision

`starter/epistemic_core.py` defines immutable lab/claim contracts, provenance
weights, strict pending codecs, aggregation, bounded revisions, receipts, and
semantic preflight. Thin contradictions are held as fully anchored
`pending__v3__...` records containing quantized evidence quality and an origin
fingerprint. Only distinct origins contribute; duplicate IDs, malformed tokens,
ambiguous retractions, and cross-family replays cannot mutate state. Legacy v2
records migrate as one conservative source. Four sequential reports and an
equivalent packet use the same bounded aggregate and diminishing returns.

Every result carries one closed evidence class and a stable observable receipt
with the event, target(s), quality, provenance status, prior/posterior, bounded
log-odds movement, pending count, and selected action. Preflight rejects
unattributed or duplicate revisions, invalid pending drops, unapproved scope,
OOD/in-model mixtures, non-finite values, and any operation outside the closed
contract.

## Reasoning sidecar

`reasoning_graph/` is non-scoring standard-library code. It stores typed,
versioned nodes and edges through an append-only hash-chained journal with
atomic SQLite transactions, snapshots, replay, correction history, and
fail-closed serialization. It answers why/history, prediction provenance,
contradictions, open questions, context rules/failures, consensus divergence,
and invalidation impact. Local hashes detect tampering but are not signatures;
provenance is unverified unless an external boundary supplies authentication.

Working, episodic, semantic, and personalized memory are separate. Personal
beliefs are owner-scoped and never silently become lab consensus. Recorder
events preserve proposed/rejected operations as audit facts; only applied
operations can create authoritative belief edges.

`scientific_harness/` adds deterministic handoff sensors, labelled context
assembly, a read-only reviewer, human-gated rule candidates, audit/reporting,
and a public-interface-only challenge adapter. `starter/harness_guard.py`
preserves valid scored results and converts exceptions or structurally invalid
results into attributed no-ops. `demo/` replays pending evidence, corroboration,
why/provenance, injection, OOD, dissent, rule review, and invalidation offline.

## Verification

The integration matrix runs the practice self-check, public scorer, 62-case
adversarial corpus, 24 metamorphic probes, all unit/integration tests,
compile-all, anti-bypass scans, and hash-seed demo determinism on Python 3.10,
3.11, and 3.13. Passing these checks demonstrates containment for the tested
threat model; it does not establish scientific truth or authenticated lineage.

## Hardening changelog

A 20-item predicted-hidden-stream fixture (`adversarial/predicted_hidden_*.json`,
built from the spec's four capabilities, the seed graph's declared blind spots,
and prior red-team findings) surfaced real gaps, since fixed:

- **State-alias coverage.** `pluripotent-like state` and bare `pluripotency`
  were not recognized as naming the pluripotent state, so otherwise clean,
  strongly-provenanced contradictions silently produced no revision at all.
- **Comma-joined sign-flip (closes redteam R10/R11).** A distractor clause
  about an unrelated experiment, comma-joined into the same sentence as a
  real contradiction (`"...returned to a pluripotent state, though an
  unrelated assay failed to replicate"`), previously flipped the revision's
  sign. Coordinating/concessive commas (though, although, whereas, while,
  but) now split into their own clause before polarity is read.
- **Role-ambiguity false trigger.** A trailing exclusion clause naming what
  did *not* happen (`"...without passing through any intermediate or
  pluripotent state"`) was counted as a third participant, wrongly
  suppressing genuine lateral-conversion (OOD regime) detection.
- **Negation-idiom false trigger.** `"never-before-seen"` (a common hedge
  meaning "novel") tripped the bare `never` denial check, scoring a
  confirmation as a contradiction.
- **Counterfactual-as-evidence.** `"If a defined factor were applied, cells
  would return to a pluripotent state"` was read as an observed result. Such
  conditional/counterfactual framing is now classified as hypothesis, never
  as evidence.
- **Unicode evasion.** Normalization now strips unassigned/surrogate/other-
  symbol codepoints and a named set of invisible joiners/fillers, applied
  consistently to both the control-pattern detection paths (previously one
  path was hardened and the other was not).
- **Control-pattern coverage.** Widened to match mid-sentence (not just
  bracket-led) role/instruction phrasing, bare "provenance" as an override
  target, direct claim-ID references (`set C3c`), and passive/modal mutation
  phrasing (`"C3c confidence is updated"`, `"must be set"`).

Rejected: a set of unused "mathematical foundations" helpers (cosine
similarity, Bayes posterior, isolation-forest scoring, AST-based checks)
proposed on another branch. `ingest()` never calls them, so they cannot
affect any scored capability — pure surface area with no performance
benefit, and excluded on that basis.
