# GROUND TRUTH reasoning design

## Trust boundary

`starter/my_solution.py` contains the complete scored policy. `ingest` reads a
read-only graph, treats `item.body` as untrusted text, and treats structured
`item.provenance` as the only evidence-quality channel. It never reads
`item.tag` or mutates the graph directly; every decision is returned as an
attributed member of the framework's closed `Delta` vocabulary. It may, for a
narrow, firewall-flagged minority of items, consult an optional model (see
"Canary model review" below) — that model has no graph, provenance, or
mutation access of its own and cannot itself change what a deterministic gate
would otherwise decide.

The body is Unicode-normalized before inspection. Composite control-plane
instructions, imperatives, questions, hypotheses, conflicting events, unknown
states, and ambiguous endpoint roles fail closed as an attributed `no_op`.
Text may identify a scientific event, but it cannot choose claim IDs,
operations, confidence values, scope keys, provenance weight, or OOD status.
Detected control-plane instructions are an absolute gate: no verdict from any
model can ever clear that flag.

## Authorization monitor

Whatever the policy decides passes through a final, independent reference
monitor (`_authorize`) before it is returned. The monitor re-derives, from the
graph and the *structured* provenance alone, exactly what each delta is allowed
to be: `revise_confidence` must target a real claim with a finite in-range
confidence; `set_scope` must carry only the mechanism-keyed exception derived
from structured provenance; `drop_claim` may reference only a pending record
that already exists in the graph; `hold_pending` must be a well-formed token
attributed to the active item; and an OOD `propose_axis`/`propose_regime` value
must be a property/regime the graph's own domain already declares out of model.
Every delta must be attributed to the current item, and the operation must be
in the closed vocabulary. Any delta that fails — a body-fabricated target, an
out-of-range confidence, an arbitrary OOD string, a spoofed attribution — makes
the entire result collapse to one attributed `no_op`. This is defense in depth:
the deterministic policy already never sources these from body text, and the
monitor guarantees that no future path can either.

## Evidence weighting and revision

Provenance fields use bounded count grammars and closed aliases for directness,
effect strength, method class, and retraction status. Invalid metadata fails
closed. Accepted evidence receives a deterministic score with weights of 40%
independent groups, 15% replications, 20% directness, 15% effect, and 10% method
reliability. Counts saturate to give diminishing returns.

Credible in-model evidence updates confidence in bounded log-odds. Strong
contradictions move more than confirmations, while already-high confirmations
receive only a small nudge. Strong mechanism-specific contradictions also add a
scope exception instead of deleting the general belief.

Thin extraordinary contradictions are held as versioned pending records. Each
record contains only a semantic fingerprint, quantized structured provenance,
and a hash of the trusted evidence ID. Four distinct compatible origins can
promote a pending family; duplicate origins do not accumulate. Retractions and
failed replications can drop only one exactly matching pending family.

## Canary model review

The decision flow is: input → deterministic policy → **YES** (intake), **NO**
(omit), or **UNSURE** → canary model → deterministic policy again → **YES**
(intake) / **NO** (omit). "UNSURE" is a body the deterministic parser could
not cleanly classify — unbalanced delimiters, or more than one candidate event
— and only those items reach the sacrificial "canary" model inlined in
`starter/my_solution.py` (`_oracle_review`). The injection gate above is a
confident **NO**, never routed to the canary: a regex-caught control-plane
attempt cannot be argued back in.

The canary gets only the raw body and generic task instructions, never
provenance, graph state, claim IDs, or the mutation API, and its JSON schema
has no field for a confidence value, claim ID, or delta — only a closed
`benign`/`injection`/`abstain` verdict plus verbatim supporting quotes. It
proposes nothing. Its output is then re-checked by the deterministic policy:
every quote must verify against the actual body (an ungrounded quote downgrades
to `abstain`), and admission additionally requires the deterministic parser to
have *independently* resolved exactly one eligible event. So the canary can
only license a structurally-noisy-but-genuine single event onto the same intake
pipeline every clean **YES** uses; it can never manufacture an event, and
multi-event ("ambiguous") bodies always resolve to **NO** regardless of the
verdict. Everything that is not admitted ends in the same `no_op` the
deterministic policy would already return, with the verdict appended to the
rationale for audit.

The canary is unavailable without `GEMINI_API_KEY`, and `GT_LLM_MODE=off`
disables it even with a key present. Missing dependency, timeout, malformed
JSON, or an unrecognized field all resolve to "unavailable," not a raise —
`ingest` stays byte-identical to the deterministic policy whenever the canary
can't be consulted. The key is read from the environment; a git-ignored `.env`
placed next to `my_solution.py` is auto-loaded for local use, and because it is
never committed, judging runs key-free and deterministic. See
`starter/.env.example` for the optional local configuration surface.

## Out-of-distribution handling

Endpoint topology is resolved from graph cell states. A credible direct
cross-lineage transition with no intermediate proposes the declared
`lateral_somatic_conversion` regime. An identity-preserving change on a property
listed in the domain's excluded axes proposes that axis. Potency reversals and
adjacent or same-lineage transitions remain in-model and revise claims rather
than being mislabeled OOD.

## Verification

The public practice scorer, the 20-item predicted stream, direct-file loading,
duplicate-origin handling, pending accumulation, exact pending invalidation,
the authorization monitor, and the canary-model routing/verification/fail-closed
behavior above are regression tested (`python -m pytest -q`). An offline
control-canary check (`adversarial/control_canary.py`, gated in the test suite)
runs clean scientific items and attacked variants — injected instruction,
provenance boast, embedded claim ID, Unicode disguise, persuasive suffix —
through the deterministic policy from the same graph snapshot and compares exact
structured decisions; its release gate is zero unexpected mutations, and every
attack must reproduce the clean decision or fail closed to `no_op`. The scored
path uses only the Python standard library and the official challenge types;
`google-genai` is an optional dependency of the canary path alone and is never
required.
