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
5. Detect OOD evidence. Direct conversion between distinct terminal identities
   proposes the declared lateral-conversion regime. Identity-preserving changes
   in age, function, chromatin, transcriptional, metabolic, or morphology axes
   propose an axis. These cases do not revise unrelated beliefs. Failed or
   retracted evidence cannot create an OOD proposal.
6. Route in-model evidence to mechanism-scoped or general claims. Strong
   contradictions revise confidence down and add a method-based scope exception;
   confirmations use a smaller upward step. Confidence changes use log-odds,
   remain below the API’s three-log-odds cap, and skip meaningless saturated
   writes.

## Function map

The code is documented with a docstring on every function and property. The
helpers are grouped by responsibility: `_normalized`, `_slug`, `_contains`, and
`_looks_like_instruction` implement normalization and firewall checks;
`_number`, `_saturation`, `_mechanism`, `_quality`, and the `EvidenceQuality`
properties score provenance; `_singular`, `_state_mentions`, `_domain_value`,
`_identity_preserved`, `_ood_axis`, `_is_lateral_conversion`, and `_event`
extract graph-aware semantics; `_claims`, `_claim_kind`, `_first_kind`,
`_scoped_claim`, and `_targets` route evidence to claims; `_pending_id` and
`_matching_pending` provide deterministic pending resolution; `_revised_confidence`
performs the calibrated update; `_no_change` constructs safe no-ops; and
`ingest` coordinates the pipeline.

## Skepticism and pending evidence

An extraordinary contradiction with one source is held under a stable identifier
such as `pending__env_stress__C3d`, without changing the claim. A later retraction
or independent failed replication drops only that pending identifier. Strong
defined-factor, nuclear-transfer, and environmental evidence is routed to the
corresponding scoped claim; successful nuclear transfer also supports retained
nuclear potential. A direct terminal-to-terminal conversion is flagged as a new
regime rather than treated as a contradiction of the existing adjacency claim.

## Verification

The public sandbox passes with the firewall gate clear and `tp=1, fp=0, fn=0` for
OOD detection. Additional tests cover injection variants, spoofed provenance,
mechanism-specific updates, nuclear-transfer phrasing, weak pending evidence,
retraction cleanup, null effects, malformed input, and unseen OOD wording. A
deterministic fuzz run of 750 JSON-shaped inputs produced no crashes, rejected
deltas, structural violations, or nondeterministic results.
