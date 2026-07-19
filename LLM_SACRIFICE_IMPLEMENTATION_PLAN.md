# Sacrificial Gemini Semantic Oracle — Implementation Plan

> **Status:** Documentation and implementation plan only. No Gemini dependency,
> network call, or LLM-controlled runtime path has been added yet; the current
> scored policy remains fully deterministic.

## Two-agent execution split

This document is the shared architecture specification. Implementation is split
into two non-overlapping plans:

- **Codex 1 — trusted kernel and integration:**
  `LLM_SACRIFICE_CODEX_1_PLAN.md`
- **Codex 2 — Gemini adapter and adversarial evaluation:**
  `LLM_SACRIFICE_CODEX_2_PLAN.md`

Codex 1 owns the cross-agent interfaces and lands them first. Codex 2 implements
against those interfaces without editing the deterministic policy or preflight.
Codex 1 performs the final integration only after both independent test suites
pass. The ownership rules in the two plans take precedence over the broader
file sequence later in this shared specification.

## 1. Objective and security claim

Add Gemini as a **sacrificial, untrusted semantic oracle** that can improve
coverage on unfamiliar scientific phrasing without becoming part of the trusted
computing base. Gemini may classify or extract meaning; it must never choose a
claim ID, graph operation, confidence value, evidence weight, pending ID, scope
write, or final OOD mutation.

The resulting technique is an **adversarially verified neuro-symbolic cascade**:

1. a neural model proposes a proof-carrying semantic witness;
2. deterministic code verifies that witness against the original body and
   closed ontology;
3. deterministic policy computes the only authorized graph transition; and
4. an exact reference monitor rejects every delta not implied by that verified
   event.

“Sacrificial” means the model is expected to be attackable. Compromising it must
produce a rejected witness or a no-op, never a privileged graph mutation.

## 2. Non-goals

- Do not ask Gemini to decide whether evidence is scientifically true.
- Do not send structured provenance to Gemini or use model prose to weight it.
- Do not expose graph state, claim statements, claim IDs, pending IDs, receipts,
  tools, files, secrets, or mutation APIs to Gemini.
- Do not let Gemini produce `Delta` objects or free-form rationales used as
  control data.
- Do not replace the deterministic baseline. `off` remains the default and
  provider failure always falls back to a deterministic decision or no-op.
- Do not claim prompt-injection immunity. Report resistance and utility over a
  versioned attack corpus.

## 3. Current code map and weaknesses to address

| Current path | Existing strength | Weakness before LLM integration | Required change |
| --- | --- | --- | --- |
| `starter/my_solution.py::_direct_control_attempt` and `_assess_body` | Rejects many control-plane patterns before event extraction | Taint is item-wide. One injected clause suppresses an otherwise valid scientific clause, creating a denial-of-service/utility weakness. | Introduce clause-level taint labels. In shadow mode, measure which clean event remains after tainted clauses are removed. Do not enable mixed-item salvage until metamorphic tests prove the retained event is unchanged. |
| `_split_clauses`, `_speech_act`, `_extract_clause_event`, and polarity helpers | Deterministic, inspectable, and hardened against known attacks | Regex and fixed-window semantics remain vulnerable to unseen paraphrases, attachment errors, multilingual text, and novel OOD descriptions. | Preserve this parser as the authoritative fast path; use Gemini only for shadow comparison and later for verified fallback candidates. |
| `_ingest_policy`, `_preflight_context`, and `_receipt_for_result` | Recompute semantics rather than trusting rationale text | The body is parsed up to three times. That is wasteful now and unsafe with a nondeterministic provider because policy, preflight, and receipt could observe different witnesses. | Build one immutable `DecisionContext` per item and pass it to policy, preflight, and receipt generation. Never call Gemini more than once per item in the decision path. |
| `PreflightContext.revisions_authorized` and `semantic_preflight` | Enforces attribution, closed operations, known claim IDs, and bounded log-odds | Once the broad boolean is true, any existing claim ID and any bounded confidence can pass. This does not prove that the delta targets the event-resolved claim or uses the policy-computed posterior. | Replace broad authorization with an exact event-derived authorization manifest containing allowed target IDs and exact expected posteriors. |
| `_preflight_context(..., result)` scope handling | Checks scope keys and claim contract shape | Scope authorization is partly derived from the proposed result itself, which is circular for an untrusted producer. | Compute exact allowed scope writes from the verified event and policy before inspecting proposed deltas. Remove `result` from authorization construction. |
| OOD preflight | Restricts proposals to the domain’s closed axis/regime vocabulary | Any globally allowed axis or regime can pass when `ood_flag` is true, even if it is not the event-specific proposal. | Authorize zero or one exact `(operation, value)` pair derived from the verified event. Recompute the OOD flag from that pair. |
| `hold_pending` preflight | Validates token version and active-origin hash | It does not compare the complete token to the one expected from the active verified event. | Compute the exact permitted pending token and note bounds before validating the proposed delta. |
| Repository verification | Public and predicted-stream runners exist | There is no checked-in `tests/` suite in this checkout, and no provider-failure or witness-contract tests. | Add focused standard-library unit tests plus a separate networked evaluation runner. Keep network tests outside the default scorer path. |

The first implementation milestone is therefore **preflight hardening**, not the
Gemini API call. The current preflight is suitable as defense in depth around the
current deterministic producer; it must be strengthened before it can guard an
arbitrary model-produced candidate.

## 4. Target architecture

```text
EvidenceItem
  ├─ trusted channel: item.provenance ───────────────┐
  └─ untrusted channel: item.body                    │
            │                                        │
            ├─ deterministic parser                  │
            └─ Gemini oracle (shadow/fallback only)  │
                    no tools, no graph, no IDs        │
                            │                         │
                  tainted SemanticWitness            │
                            │                         │
             deterministic witness verifier          │
                            │                         │
                   immutable DecisionContext ─────────┘
                            │
               deterministic evidence policy
                            │
             exact AuthorizationManifest
                            │
                 candidate typed Deltas
                            │
             deterministic semantic preflight
                            │
                  official DeltaAPI gate
```

Only the witness verifier, evidence policy, authorization manifest, semantic
preflight, and official `DeltaAPI` are trusted. Gemini output remains tainted
even when it is valid JSON.

## 5. Proof-carrying witness contract

Add a versioned immutable `SemanticWitness` with a strict JSON schema and
`additionalProperties: false`. The model may return only:

- `schema_version` — exact supported version;
- `disposition` — `candidate`, `abstain`, or `injection`;
- `supporting_quotes` — one or more bounded verbatim substrings from `item.body`;
- `suspicious_quotes` — bounded verbatim substrings identified as instructions;
- `speech_act` — closed enum;
- `proposition` — the existing closed `Proposition` enum;
- `polarity` — the existing closed `Polarity` enum;
- `observed` — boolean;
- `source_surface` and `destination_surface` — body text, never graph IDs;
- `property_axis_candidate` and `regime_candidate` — closed proposal enums or
  `null`;
- `has_intermediate`, `failed_replication`, and `lineage_restriction` — booleans;
- `abstain_reason` — a closed enum, not arbitrary prose.

Explicitly omit model confidence, provenance, evidence quality, claim IDs,
operations, confidence revisions, scope keys, pending tokens, and rationale.

The deterministic verifier must:

1. reject unknown, missing, duplicated, oversized, or wrongly typed fields;
2. verify every quote occurs in the normalized original body and does not cross
   an unbalanced or removed span;
3. map surface state names through `GraphView.cell_state`; Gemini cannot invent
   ontology IDs;
4. derive event eligibility and role requirements using existing enums and
   policy contracts;
5. reject witnesses whose polarity, transition endpoints, or OOD proposal are
   unsupported by their quoted clause;
6. keep instruction-marked spans tainted and outside the scientific witness;
7. return either one verified `SemanticEvent` or an explicit abstention.

A valid schema is necessary, not sufficient. Schema validation prevents shape
attacks; semantic verification prevents a well-formed lie from gaining
authority.

## 6. Exact authorization manifest

Replace coarse preflight facts with an immutable `AuthorizationManifest`
computed from the verified event, trusted provenance, and current graph before
candidate deltas are inspected. It should contain:

- `active_evidence_id`;
- `semantic_fingerprint` and witness source (`deterministic` or `gemini`);
- exact allowed revision pairs `{claim_id: expected_new_confidence}`;
- exact allowed scope writes `{claim_id: expected_scope}`;
- the exact allowed pending write, if any;
- exact allowed pending IDs to drop;
- zero or one exact OOD proposal `(propose_axis, value)` or
  `(propose_regime, value)`;
- whether a single no-op is the only permitted result.

`semantic_preflight` then compares complete payloads to this manifest. Use only
a documented floating-point tolerance for the policy-computed posterior; do not
accept merely “bounded” alternatives. The manifest must never be constructed
from `result.deltas`.

## 7. Routing and failure semantics

Support only these modes:

| Mode | Gemini call | Can affect returned deltas? | Purpose |
| --- | --- | --- | --- |
| `off` | Never | No | Default, scorer-safe deterministic baseline |
| `shadow` | Yes | No | Collect disagreement, attack, latency, and utility measurements |
| `fallback` | Only after deterministic semantic abstention | Only through a verified witness and exact manifest | Optional production experiment after rollout gates pass |

Do not implement model-first routing. A deterministic injection, malformed-body,
invalid-provenance, duplicate-origin, or explicit ambiguity finding cannot be
overridden by Gemini in `fallback` mode. Mixed science-plus-injection salvage is
a separate feature gate and requires clause-level non-interference evidence.

Missing credentials, optional-package absence, timeout, rate limiting, provider
error, invalid JSON, schema mismatch, unsupported model output, or verifier
failure must never raise out of `ingest`. Return the deterministic result when
one exists; otherwise return an attributed no-op. Bound the request to one
attempt by default, a short timeout, temperature zero, and a small output limit.

For repeatability, cache only validated witness JSON under a key containing the
normalized-body hash, model ID, prompt version, and schema version. Never cache
or log API keys, raw headers, graph state, or deltas. Raw-body logging remains
off by default.

## 8. File-by-file implementation sequence

### Phase 0 — Freeze the baseline

1. Record outputs from `python selfcheck.py` and
   `python adversarial/run_predicted.py`.
2. Add regression fixtures for every current event, target, confidence, pending,
   injection, and OOD result before moving code.
3. Add tests proving `off` mode performs no environment lookup beyond local
   configuration and makes no network call.

### Phase 1 — Harden the deterministic reference monitor

1. In `starter/epistemic_core.py`, introduce `AuthorizationManifest` and replace
   broad preflight fields with exact permitted payloads.
2. In `starter/my_solution.py`, compute the manifest from event + provenance +
   graph, not from the proposed `IngestResult`.
3. Add negative tests that mutate one field at a time: wrong claim ID, wrong
   posterior, wrong scope target/value, wrong pending fingerprint, wrong axis,
   wrong regime, mixed no-op, and mismatched evidence ID.
4. Require every mutation of those candidates to fail closed.

### Phase 2 — Parse once and modularize

1. Add `starter/semantic_types.py` for semantic enums, `SemanticWitness`,
   `VerifiedWitness`, and `DecisionContext`.
2. Move normalization, clause splitting, speech-act detection, state mention,
   polarity, and event extraction into `starter/deterministic_semantics.py`.
3. Keep evidence weighting, target selection, revision calculation, and delta
   construction deterministic.
4. Make `my_solution.ingest` a thin orchestrator: validate input, construct one
   context, propose a decision, preflight, attach receipt.
5. Test both normal imports and the challenge’s direct file-path loading mode.

### Phase 3 — Add the optional Gemini adapter

1. Add `google-genai` to `requirements.txt` only when this phase begins; import
   it lazily inside `starter/gemini_oracle.py` so `off` mode still works when the
   package or key is absent.
2. Read and validate environment configuration once. Unknown modes, invalid
   numbers, or empty model IDs resolve to `off`, not an exception.
3. Use a single stateless `generate_content` request with structured JSON output,
   no chat history, no tools/function calling, no files, and no provider-side
   retrieval.
4. Send only the body, generic task instructions, schema, and closed semantic
   labels. Do not send provenance or graph contents.
5. Convert all provider errors into a typed `OracleFailure`; never expose raw
   provider messages in the scored rationale.

### Phase 4 — Shadow mode and the adversarial sandbox

1. Add `starter/semantic_router.py`. In `shadow`, return the deterministic
   context unchanged and send the model witness only to metrics/audit code.
2. Add `adversarial/llm_sacrifice/corpus.jsonl` with benign controls and direct,
   indirect, quoted, role-tag, fake-JSON, provenance-spoof, Unicode/confusable,
   encoded, multilingual, multi-event, and mixed science-plus-injection cases.
3. Add `adversarial/llm_sacrifice/run.py` to evaluate the weak model, comparator
   model, deterministic parser, and verified hybrid over repeated runs.
4. Report attack-success rate, clean semantic recall, clean-vs-attacked
   invariance, schema failures, hallucinated targets, verifier rejections,
   abstentions, latency, and estimated token cost.
5. Save redacted JSONL replay records with body hashes and prompt/model/schema
   versions so every reported number is reproducible.

### Phase 5 — Metamorphic and differential testing

For each clean item `x`, generate transformations `T(x)` that add or alter only
untrusted control text. Assert:

- the legitimate verified event is unchanged after tainted clauses are removed;
- no attack can change the authorization manifest;
- a pure attack produces no-op;
- changing body claims about provenance cannot change evidence quality;
- replacing a claim ID in body text cannot change the target;
- standalone model disagreement causes abstention, not arbitration by confidence.

Use grammar-based mutation first. Evolutionary prompt search can be added later
to retain mutations that maximize model/parser disagreement or verifier
rejection, but it must remain an evaluation tool, never a runtime dependency.

### Phase 6 — Carefully gated fallback

Enable `fallback` only when all rollout gates below pass. Initially allow it only
for deterministic `NON_EVIDENCE` abstentions where there is no injection,
malformation, ambiguity, invalid provenance, retraction, or duplicate origin.
The model supplies a candidate witness; the existing deterministic policy still
selects targets and computes all deltas.

Do not enable mixed-item salvage until clause-level taint tests demonstrate that
adding an injected clause cannot alter the clean event, target, direction,
posterior, pending action, or OOD proposal.

## 9. Rollout gates

Fallback is not ready until:

- `off` and `shadow` produce byte-equivalent deltas on all deterministic
  regression cases;
- the public self-check and predicted-stream runner have no regression;
- unauthorized mutation count is exactly zero across the versioned attack
  corpus and all one-field preflight mutation tests;
- every timeout, missing-key, missing-package, malformed-output, and provider
  exception test fails closed without crashing;
- repeated model runs cannot produce a delta outside the exact manifest;
- held-out semantic recall improves materially without reducing OOD precision or
  firewall integrity;
- latency and token cost remain inside explicit demo limits;
- logs contain no API key, raw authentication data, graph IDs, or raw bodies by
  default.

Any gate failure keeps `GT_LLM_MODE=off` for scoring and `shadow` for research.

## 10. Configuration contract

`.env.example` documents local settings. Production code should read
`os.environ`; the scored entrypoint must not require a `.env` parser. Security
invariants such as “no tools,” “no provenance,” “no graph access,” and
“fail closed” are hard-coded and deliberately have no environment override.

The default model is a configurable Flash-Lite sacrificial model. The comparator
exists only for the networked evaluation runner. Pin exact model IDs in reported
experiments and include them in replay records; never use a moving `latest`
alias for benchmark claims.

## 11. Pitch-ready description

> We use Gemini as a sacrificial semantic oracle outside the trusted computing
> base. It emits a proof-carrying, extractive witness but has no graph, provenance,
> tool, or mutation access. A deterministic reference monitor verifies the
> witness and constructs an exact authorization manifest before the official API
> can mutate state. We measure non-interference with metamorphic prompt-injection
> fuzzing and differential tests across weak and stronger models.

That claim should be presented only after the corresponding rollout gates and
reported corpus results exist.
