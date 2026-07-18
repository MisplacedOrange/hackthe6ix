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
