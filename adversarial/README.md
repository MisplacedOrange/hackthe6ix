# Adversarial policy dataset

This directory contains a human-labelled red-team corpus for the repository's
candidate policy, `starter/my_solution.py`. It does not modify the candidate.

The corpus has 62 isolated cases (70 evidence items). Most cases start and end
with a fresh copy of `groundtruth/data/seed.json`; multi-item cases intentionally
preserve state to test pending claims, confirmations, and retractions.

Run it from the repository root:

```powershell
python adversarial/run_adversarial.py starter/my_solution.py
```

The runner exits nonzero when it finds policy mismatches. That is expected for a
red-team corpus. It reports emitted/applied deltas, OOD decisions, confidence
changes, pending state, rationale, and the exact failed expectation.

To test another candidate:

```powershell
python adversarial/run_adversarial.py path/to/solution.py
```

## Dataset structure

`cases.json` contains:

- reusable structured-provenance profiles;
- independent cases with a title, category, and semantic expected outcome;
- one or more ordered evidence items per case;
- machine-checkable expectations for OOD, mutation attempts, required/forbidden
  delta operations, confidence direction, and pending state.

The expectations are an explicit threat model, not an organizer-provided hidden
answer key. Several probes deliberately use plausible JSON variants not present
in the six-item practice data, such as boolean retraction status, numeric zero
effect, and count strings with units. If the production schema rejects those at
an earlier boundary, keep the cases as boundary-validation tests and enforce that
schema before `ingest`.

## Attack families

- canonical controls showing behavior the policy gets right;
- prompt-injection paraphrases, Unicode confusables, and firewall false positives;
- polarity, modality, passive voice, quotation, and event-argument confusion;
- structured-provenance type, negation, substring, and count parsing;
- OOD endpoint decoys, covariates, mediated transitions, and missing synonyms;
- pending-item collisions, stale pending state, and overly broad retraction
  matching;
- mixed-clause instruction laundering, reported allegations, truncated quotes,
  missing event roles, and invalid-provenance deletion attempts.

See `RESULTS.md` for the audited candidate's current strengths and failures.
`RESULTS.md` preserves the original keyword-policy baseline;
`CURRENT_RESULTS.md` records the replacement semantic policy.
