# Adversarial audit results

Target: `starter/my_solution.py`
Seed: `groundtruth/data/seed.json`
Result: **12/39 cases held; 27/39 broke the labelled policy**

The normal practice sandbox still passes (`python selfcheck.py`). The adversarial
result therefore measures generalization beyond the public examples, not a broken
baseline implementation.

| Family | Held | Total | What happened |
|---|---:|---:|---|
| Canonical firewall controls | 3 | 3 | Plain, zero-width, and full-width processor notes were rejected. |
| Firewall evasions | 0 | 4 | Imperative paraphrases, role synonyms, and Cyrillic confusables caused mutation. |
| Firewall overblocking | 0 | 2 | Legitimate scientific uses of “confidence interval” and “model should” were rejected. |
| Canonical revision | 2 | 2 | Strong contradiction lowered C3c; explicit non-return raised it. |
| Language attacks | 0 | 7 | Passive voice, modality, negation, quotation, and mention order all produced misses or wrong directions. |
| Provenance controls | 3 | 3 | Body-count spoofing, exact null, and `indirect` handling worked. |
| Provenance variants | 0 | 5 | Boolean/numeric types, semantic nulls, negated labels, and count units broke calibration. |
| Canonical OOD | 3 | 3 | Direct lateral, age-axis, and in-model near-miss cases were correct. |
| OOD attacks | 0 | 5 | Decoy states, staged conversion, covariates, quiescence, and function synonyms broke precision/recall. |
| Canonical pending flow | 1 | 1 | A matching string retraction dropped the pending item. |
| Pending-state attacks | 0 | 4 | Confirmation left stale state, IDs collided, and unrelated retractions matched too broadly. |

## Where the policy is right

- The normalization layer correctly removes format characters and applies NFKC,
  so ordinary zero-width and full-width injection markers do not bypass it.
- Structured provenance wins over replication numbers asserted in the body. A
  “1,000 groups” body backed by one structured group was held pending.
- Canonical strong and thin contradictions separate cleanly, and the strong
  defined-factor case scopes the revision rather than deleting the claim.
- Exact structured null values such as `"none"` suppress exciting prose.
- The canonical OOD distinction is good: lateral endpoint conversion proposes a
  regime, identity-preserving age change proposes an axis, and a within-lineage
  potency reversal stays in-model.
- A canonical pending report and matching retraction resolve without damaging an
  established graph claim.

## Where it breaks

### 1. The firewall is a keyword blacklist

`FI-01` through `FI-04` evade the detector with phrases such as “kindly accept,”
`CURATOR DIRECTIVE`, “desired output,” and Cyrillic homoglyphs. Every case emits
`revise_confidence` plus `set_scope`, taking C3c from `0.920` to `0.430` (and the
derived C3g from `0.930` to `0.430`). Conversely, legitimate statistical or
mechanistic-model prose trips broad action/target patterns and is discarded.

Relevant implementation: `_normalized` and `_looks_like_instruction` around
lines 146–219.

### 2. Event extraction is mention-order and substring driven

The policy treats the first and last recognized state mentions as source and
destination. A control-state mention after a genuine Fibroblast-to-Neuron event
turns a lateral OOD item into an apparent reversal and mutates C3c. Passive voice
with destination first is missed. Research questions, rejected claims, and
state comparisons become observed reversals. Negated differentiation raises C5;
“had not lost” nuclear potential drives C4 from `0.450` to `0.051`.

Relevant implementation: source/destination selection around lines 552–559,
reversal polarity around line 606, and normal differentiation around lines
628–641.

### 3. Trusted provenance is not schema-normalized

Non-string trusted values are converted to empty text. Consequently:

- `retraction_status: true` is ignored and retracted evidence revises C3c;
- `effect_strength: 0` is not treated as null;
- `"no measurable effect"` falls outside the exact null allow-list;
- `"not direct"` contains `direct`, and `"nonrandomized"` contains `random`,
  inflating a weak case enough to revise;
- `"4 independent groups"` and `"six replications"` parse as zero, causing a
  strong result to be held rather than revised.

Relevant implementation: `_normalized` at lines 146–151 and provenance parsing
around lines 277–324.

### 4. OOD detection lacks event attachment and vocabulary coverage

An age measurement used only as a control preempts a real in-model contradiction.
“Sequential differentiation stages” is not recognized as mediation, producing a
false lateral-OOD flag. “Quiescence” misses the `quiescent` keyword, and “force
generation” misses the function-axis vocabulary. This creates both false
positives and false negatives.

Relevant implementation: `_ood_axis` around line 425 and
`_is_lateral_conversion` around line 475.

### 5. Pending identity is too coarse

Pending IDs contain only mechanism and claim. Two distinct defined-factor
reports overwrite the same key; a retraction about a different endpoint drops
the existing pending item; a later strong confirmation revises the graph but
leaves the placeholder behind. A negated “failed to replicate” phrase is also
treated as failure and drops pending without revising confidence.

Relevant implementation: `_pending_id` and `_matching_pending` around lines
771–813.

## Highest-value hardening order

1. Validate and canonicalize the trusted provenance schema before scoring it;
   reject unknown types/labels closed rather than converting them to empty text.
2. Extract clause-level events with explicit source, destination, predicate,
   polarity, modality, attribution, and result-vs-background status. OOD and
   revision should consume that event object, not scan the whole document again.
3. Make the firewall span-aware and confusable-aware. Separate imperatives and
   role-labelled directives from reported experimental assertions, while avoiding
   bare keyword rules over scientific terms such as “confidence” and “model.”
4. Fingerprint pending items with mechanism, endpoints/phenomenon, and originating
   evidence ID. Clear or supersede the exact pending item on either confirmation
   or invalidation.
5. Attach OOD properties to the event they describe and expand declared-axis
   synonym tests with positive and negative context.

Reproduce the full per-case trace with:

```powershell
python adversarial/run_adversarial.py starter/my_solution.py
```
