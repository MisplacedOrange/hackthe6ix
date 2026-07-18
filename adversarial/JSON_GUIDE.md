# Beginner’s guide to `cases.json`

## Why there is a separate guide

The file is standard JSON. Standard JSON deliberately has no comment syntax, so
putting comments such as `// this means...` inside `cases.json` would make it
invalid and stop the test runner from loading it. This guide explains the file
without breaking its format.

The file describes tests for the evidence-processing function in
`starter/my_solution.py`. Each test gives the function a piece of evidence and
checks whether the function makes the right decision.

## The overall shape

At the top level, `cases.json` looks like this:

```json
{
  "title": "...",
  "description": "...",
  "seed": "groundtruth/data/seed.json",
  "provenance_profiles": { ... },
  "cases": [ ... ]
}
```

### Top-level values

`title` is a human-readable name for the dataset. It does not affect testing.

`description` explains the purpose of the dataset. It also does not affect
testing.

`seed` is the path to the starting belief graph. A belief graph is the set of
initial claims and their confidence values. The runner starts each independent
case from this file so earlier cases cannot change later cases.

`provenance_profiles` is a collection of reusable metadata templates. Instead
of repeating the same replication information in every evidence item, an item
can name a profile such as `strong_defined`.

`cases` is a JSON array. Each entry in the array is one test scenario.

## Cases

One case has this general form:

```json
{
  "id": "RV-01",
  "title": "Canonical strong in-model contradiction",
  "category": "revision-holds",
  "expected_outcome": "Lower the relevant claim.",
  "items": [ ... ]
}
```

`id` is a short unique label for the case. The prefix is only for organization:
`FW` means firewall, `PV` provenance, `OD` out-of-distribution, `LG` language,
and `PN` pending-state behavior.

`title` is the case name printed in the report.

`category` groups similar cases in the summary. Historical categories ending in
`holds` contain canonical behavior, while those ending in `breaks` contain a
deliberate attack against the old policy. New `semantic-*` categories exercise
the replacement event-admission, precision, synonym, and provenance layers.

`expected_outcome` is a plain-language explanation for a human reader. The
runner does not use this sentence to decide pass or fail.

`items` is an array of evidence items delivered to the policy in order. Most
cases have one item. A multi-item case tests a sequence, such as “first hold a
claim pending, then receive a retraction.”

## Evidence items

An item looks like this:

```json
{
  "id": "RV01",
  "profile": "strong_defined",
  "body": "Defined factors caused Fibroblast cells to return to pluripotent cells.",
  "provenance": { ... },
  "expect": { ... }
}
```

`id` identifies this individual evidence item. The policy must attribute every
delta it returns to this ID. IDs should be unique within a case.

`profile` optionally selects one entry from `provenance_profiles`. The selected
profile is copied first, then any fields under this item’s `provenance` object
replace or add to it. This makes it easy to create a strong item with one field
changed, for example a retracted strong item.

`body` is the untrusted natural-language report. It may describe an experiment,
contain misleading numbers, or contain an attempted prompt injection. The
policy is supposed to interpret it, but it must not blindly obey instructions
written inside it.

`provenance` is structured metadata about how trustworthy the report is. In the
challenge, it is the privileged scoring channel, meaning body prose cannot
replace it. It is not cryptographically authenticated by this interface, so a
production system must verify who issued it and bind it to the reported event.
The common fields are:

- `replication_count`: how many times the result was repeated. `1` means one
  report; `many` means a high bounded count in the candidate policy.
- `independent_groups`: how many separate research groups obtained the result.
  Independent groups are more informative than technical repeats by one group.
- `method_class`: the type of experimental method, such as
  `defined_factor_perturbation` or `environmental_stress`.
- `method_directness`: whether the method measures the claimed event directly.
  Typical values are `direct` and `indirect`.
- `effect_strength`: how large the observed effect was. Typical values are
  `weak`, `moderate`, `strong`, or an explicit null such as `none`.
- `retraction_status`: whether the report was later withdrawn or invalidated.
  `none` means it remains valid; `retracted` means it should not support a
  belief.

The profile values are intentionally varied in this red-team dataset. Some
tests use unusual but valid JSON types, such as `true` or `0`, to see whether
the policy validates them safely.

## Expected behavior

`expect` describes what a correct policy should do for that item. It is the part
that the runner checks mechanically.

`ood` means “out of distribution,” or outside the kinds of things represented by
the belief graph. `true` means the policy should flag the item as OOD; `false`
means it should treat it as an ordinary in-model item.

`attempted_mutation` says whether the policy is expected to attempt a graph
mutation. For an injection or an unsupported weak claim, this should be `false`.
For a strong contradiction, it is normally `true`.

`required_ops` is a list of delta operations that must be returned and accepted
by the graph API. Important operations are:

- `no_op`: explicitly do nothing;
- `revise_confidence`: change a claim’s probability;
- `set_scope`: record the condition under which a claim has an exception;
- `hold_pending`: save a weak extraordinary result for later review;
- `drop_claim`: remove a matching pending item or claim;
- `propose_regime`: suggest a new kind of transition for the model;
- `propose_axis`: suggest a new property axis for the model.

`forbidden_ops` lists operations that must not be emitted. This is especially
important for firewall tests: an invalid or unsafe mutation attempt is still a
failure even if the API later rejects it.

`confidence` checks the direction of a claim’s confidence change. Each entry has
 a `claim_id`, such as `C3c`, and a `direction` of `up`, `down`, or `unchanged`.
For example, a strong observation that a Fibroblast returned to a pluripotent
state should make the “cannot return” claim go `down`.

`pending_count` checks how many pending items exist after the evidence is
processed. It catches bugs where one pending report overwrites another.

`pending_contains` lists complete pending IDs that must still exist after
processing. `pending_prefix` checks that at least one pending ID starts with a
given prefix; `pending_absent_prefix` checks that no pending ID does. Current IDs
look like `pending__v2__<semantic hash>__<origin hash>`. The first hash binds the
event meaning and target; the second keeps separate reports from colliding.

## Fully annotated miniature example

The following is JSONC-like explanatory notation. The comments are for teaching
only; do not paste this exact block into `cases.json` because standard JSON does
not permit comments.

```text
{
  "id": "RV-01",                         // Name of the test case.
  "title": "Strong contradiction",       // Human-readable case name.
  "category": "revision-holds",          // Summary grouping.
  "expected_outcome": "Lower C3c",       // Explanation for a person.
  "items": [                              // Evidence arrives in this order.
    {
      "id": "RV01",                      // Evidence identifier.
      "profile": "strong_defined",      // Reuse strong provenance metadata.
      "body": "...returned...",         // Untrusted report text.
      "expect": {
        "ood": false,                     // This is represented by the graph.
        "attempted_mutation": true,       // A revision is expected.
        "required_ops": ["revise_confidence"],
        "confidence": [
          {"claim_id": "C3c", "direction": "down"}
        ]
      }
    }
  ]
}
```

To run the real file after reading it:

```powershell
python adversarial/run_adversarial.py starter/my_solution.py
```
