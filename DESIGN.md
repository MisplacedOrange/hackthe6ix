# Paladin — Design

`ingest` reads a read-only graph, treats `item.body` as untrusted text, and
treats structured `item.provenance` as the only evidence-quality channel. Text
may *identify* a scientific event, but it can never choose a claim ID, an
operation, a confidence value, a scope key, provenance weight, or OOD status.
Every decision is returned as an attributed member of the closed `Delta`
vocabulary; the graph is never mutated directly.

## Firewall

The body is Unicode-normalized (NFKC; invisible, format, and confusable
characters stripped) before inspection. `_direct_control_attempt` matches six
control-plane regex families ("ignore previous instructions", "set confidence
to…", spoofed system/grader messages, imperatives targeting claims/deltas) over
both folded and confusable-mapped text; a clause-level speech-act classifier
rejects imperatives and task framing. **A control-plane hit is an absolute gate:
it fails closed to an attributed `no_op` and is never escalated to any model.**
Structured provenance with any unrecognized field is invalid and also fails
closed. Because untrusted text cannot express a delta and cannot supply targets,
no persuasive document can mutate the graph.

## Evidence weighting

Provenance fields use bounded count grammars and closed alias maps for
directness, effect, method class, and retraction. Counts saturate for
diminishing returns. Accepted evidence gets a deterministic quality score:

```
0.40·saturate(independent_groups) + 0.15·saturate(replications)
+ 0.20·directness + 0.15·effect + 0.10·method_reliability
```

Thresholds gate action: `credible` (groups ≥ 2, score ≥ 0.60) is required to
move a belief; `strong` (groups ≥ 3, score ≥ 0.82) unlocks scoping.

## Revision

Confidence updates in bounded log-odds (`logit`/`sigmoid`, movement capped at
3.0). Contradictions move more than confirmations; a well-powered study
confirming a weak prior moves meaningfully; confirming an already-high prior
only nudges. A strong, mechanism-specific contradiction emits `set_scope` — a
mechanism-keyed exception — rather than deleting the general belief: *true in
general, false under this condition.*

## Skepticism

Thin extraordinary contradictions are held as versioned `hold_pending` tokens
carrying only a semantic fingerprint, quantized provenance, and a hash of the
trusted evidence ID — never body text. Four distinct compatible origins promote a
pending family (duplicate origins do not accumulate); a retraction or failed
replication drops exactly one matching family. Duplicate evidence origins are
rejected.

## Out-of-distribution

Topology is read from graph potency levels and lineage identity. A credible
direct cross-lineage transition with no intermediate proposes the declared
`lateral_somatic_conversion` regime; an identity-preserving change on an excluded
property proposes that axis. Potency reversals and adjacent/same-lineage
transitions stay in-model and revise claims. Every OOD proposal requires that the
axis/regime already appear in the graph's own declared `axes_excluded` /
`regimes_not_modeled` — an arbitrary body string can never become one.

## Two backstops

**Canary (optional, local-only).** A body the parser cannot cleanly classify
(unbalanced delimiters, or more than one candidate event) may be shown to a
sacrificial model that sees *only* the raw text — no provenance, graph, IDs, or
mutation API. It returns a closed `benign`/`injection`/`abstain` verdict with
verbatim quotes; ungrounded quotes downgrade to `abstain`. It proposes nothing,
and admission still requires the deterministic parser to have independently
resolved exactly one eligible event. Without `GEMINI_API_KEY` it fails closed, so
judging runs the pure deterministic policy.

**Authorization monitor.** Every result passes through `_authorize`, which
re-derives from the graph and structured provenance alone what each delta may be
(valid claim ID, in-range confidence, mechanism-keyed scope, pre-existing pending
target, declared OOD value, attribution to the active item). Any delta it cannot
justify collapses the whole result to one `no_op`. Defense in depth: the policy
already never sources these from text, and the monitor guarantees no future path
can either.
