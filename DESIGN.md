# GROUND TRUTH semantic-policy design

## Security boundary

`starter/my_solution.py` is a small entrypoint. The policy lives in
`starter/semantic_policy.py` and returns only attributed, closed-vocabulary
`Delta` objects; it never writes through `GraphView`.

The evidence body is untrusted. It may propose a clause-local scientific event,
but it cannot directly name an operation, claim ID, confidence, scope field, or
evidence weight. Structured provenance is syntactically privileged for scoring,
but it is **not authenticated by this challenge interface**. A real deployment
must verify the issuer and a signature binding the provenance to the complete
typed event before calling this policy.

The implementation is deterministic and standard-library-only. It is not
keyword-free: semantic frames and the small direct-control gate still use
lexical patterns. The important improvement is that a matched word is not itself
an authorization. A complete typed event and valid structured provenance must
pass separate admission checks before deterministic code can choose a delta.

## Decision pipeline

1. Normalize Unicode, remove formatting controls, reject oversized input, and
   fail closed on unbalanced quotation or brackets.
2. Detect only composite control-plane patterns, such as a role plus directive,
   an override plus protected channel, or an output objective plus claim target.
   Individual words such as `system`, `model`, `direct`, and `confidence` are not
   direct firewall triggers.
3. Split the body into clauses and classify their speech acts. Questions,
   hypotheses, quotations, reported allegations, and writing/processing
   instructions are not observations. If any clause is instruction-shaped, the
   whole evidence item abstains so a second factual-looking clause cannot be used
   to launder the command.
4. Extract a closed `SemanticEvent`: proposition, polarity, source,
   destination, property axis, and OOD regime. Transition updates normally
   require explicit graph roles; narrow directional phrases cover the official
   within-lineage case. Conflicting or unrelated multiple events abstain.
5. Parse provenance with anchored grammars and exact enums. Counts must be
   nonnegative integers. Unknown values or malformed types return `no_op`; they
   cannot create, revise, or delete graph state.
6. Map the admitted event to a finite graph claim using deterministic code.
   Body prose never supplies a claim ID, target confidence, or delta operation.
7. Use bounded log-odds changes derived from structured quality. Thin
   contradictions become origin-distinct pending records; sufficiently grounded
   contradictions revise confidence and may add a mechanism-scoped exception.

## Auditable evidence classes

Each return path receives a typed `EvidenceClass` after the semantic and
provenance gates have made their decision. The closed taxonomy distinguishes
control-plane injection, invalid input, non-evidence, null results, trusted
invalidation, weak evidence, contradiction, confirmation, OOD evidence, and
saturation. The class is appended to the rationale so decisions can be counted
and audited without introducing another graph-mutation channel.

This adopts the combined branch's useful classifier idea, but not its classifier
implementation: a class describes the decision that the stricter policy already
made; it cannot authorize a claim target or compensate for malformed provenance.

## Pending evidence and invalidation

Pending IDs have the form:

```text
pending__v2__<semantic fingerprint>__<origin fingerprint>
```

The semantic fingerprint binds mechanism, target claim, proposition, source,
destination, and property axis. The origin fingerprint prevents two reports of
the same phenomenon from overwriting one another.

Validation occurs before invalidation. A malformed “failed to replicate” item
cannot delete pending evidence. A valid retraction may clear one exact-semantic
pending report, but it abstains if several origins match because the current
input schema has no `retracts_evidence_id`. Strong confirmation may clear all
same-event pending reports because it confirms the phenomenon rather than
claiming to identify one retracted origin.

## Out-of-distribution behavior

- A direct equal-potency, cross-lineage endpoint conversion proposes the
  `lateral_somatic_conversion` regime.
- An identity-preserving change to an unmodeled property proposes the relevant
  axis or regime.
- A within-lineage potency reversal stays in-model and revises a represented
  claim rather than being labelled OOD.
- Questions, commands, examples, and writing requests are not scientific OOD;
  they are simply inadmissible evidence.

## Hard limits

This policy cannot prove scientific truth from prose and metadata supplied by
the same untrusted party. It also cannot guarantee recall for every synonym or
novel grammatical construction. A production design should add:

- signatures over issuer, study ID, typed event fields, raw-artifact hash,
  provenance, timestamp, and explicit retraction linkage;
- independent study identities instead of self-reported counts;
- an LLM or scientific NLI model only as an untrusted extractor, using a closed
  schema and abstention on parser/verifier disagreement;
- source-span retention, external evidence retrieval, and dependency-aware
  rollback for already-applied revisions.

Passing the repository's tests demonstrates containment on the tested threat
model. It is not a proof of prompt-injection security or factual truth.

## Verification

```powershell
python public_scorer.py starter/my_solution.py
python adversarial/run_adversarial.py starter/my_solution.py
python adversarial/metamorphic_checks.py starter/my_solution.py
```
