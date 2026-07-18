# Research basis for the semantic policy

The implementation follows a common conclusion across prompt-injection and
factuality research: language-model or parser output may propose meaning, but a
separate deterministic component must authorize effects.

## Prompt-injection research

- [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208)
  trains models to respect explicit trust levels. The challenge policy applies
  the same hierarchy structurally: body prose is untrusted, structured
  provenance is privileged, and only the delta API can mutate state.
- [StruQ: Defending Against Prompt Injection with Structured Queries](https://arxiv.org/abs/2402.06363)
  separates trusted instructions from untrusted data. Its full defense requires
  specially trained models, so this repository implements the separation but
  does not claim StruQ-equivalent model robustness.
- [Defending Against Indirect Prompt Injection Attacks With Spotlighting](https://arxiv.org/abs/2403.14720)
  supports marking untrusted content as an additional layer. Marking is not used
  as an authorization boundary here because delimiters alone can be ignored.
- [SecAlign: Defending Against Prompt Injection with Preference Optimization](https://arxiv.org/abs/2410.05451)
  and the Instruction Hierarchy show that training can improve model behavior.
  This offline standard-library policy cannot retrain a frontier model.
- [Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813)
  (CaMeL) is the closest architectural match. CaMeL separates susceptible
  language processing from deterministic control/data-flow enforcement. Here,
  semantic extraction produces an untrusted `SemanticEvent`; deterministic code
  alone maps it to claims and deltas.
- [The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents](https://arxiv.org/abs/2412.16682)
  motivates checking whether a candidate action contributes to the trusted task.
  The local equivalent is the event-admission gate: writing requests, questions,
  quotations, and hypotheses are not scientific result events.

## Factuality and falsified-data research

- [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974)
  (SciFact) uses atomic scientific claims and support/refute/insufficient-evidence
  judgments. The policy similarly atomicizes clauses and abstains when polarity
  or roles are unresolved.
- [MultiVerS: Improving scientific claim verification with weak supervision and full-document context](https://arxiv.org/abs/2112.01640)
  reinforces claim-level verification against evidence documents. Full external
  verification is outside the challenge contract but is the recommended next
  upstream layer.
- [ProVe: A Pipeline for Automated Provenance Verification of Knowledge Graphs against Textual Sources](https://arxiv.org/abs/2210.14846)
  validates candidate graph statements against source text. The implementation
  retains clause-local event attachment and exact pending fingerprints; a
  production version should additionally retain source spans and hashes.
- [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251)
  motivates decomposing persuasive paragraphs into atomic propositions before
  verification.
- [Long-form factuality in large language models](https://arxiv.org/abs/2403.18802)
  describes an atomicize/search/judge evaluation pipeline across frontier-model
  outputs. Search support is useful evidence, but should remain outside the graph
  writer and be restricted to trusted sources.
- [Factored Verification: Detecting and Reducing Hallucination in Summaries of Academic Papers](https://arxiv.org/abs/2310.10627)
  reports subtle factual errors in summaries from GPT-4 and Claude 2 and supports
  verifying claims separately rather than trusting document-level fluency.
- [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664)
  motivates measuring disagreement over meaning rather than surface wording. The
  offline policy approximates this with paraphrase/metamorphic invariants; a
  model-backed extractor should abstain when independent semantic parses disagree.
- [Trusty URIs: Verifiable, Immutable, and Permanent Digital Artifacts for Linked Data](https://arxiv.org/abs/1401.5775)
  and [SciChain: Trustworthy Scientific Data Provenance](https://arxiv.org/abs/2002.00141)
  motivate content hashes, authenticated identities, and tamper-evident provenance.

## What was implemented

- closed enums for speech act, proposition, polarity, directness, and effect;
- clause-local source/destination resolution instead of first/last document mention;
- active/passive voice and predicate-local negation;
- questions, instructions, quotations, hypotheses, and contradictory documents
  fail closed;
- mixed instruction/result items and unbalanced quotations fail closed;
- only composite, high-precision control-plane phrases remain in the direct gate;
- confusable-character folding for those control patterns;
- anchored structured-provenance grammars with exact enum parsing;
- provenance validation occurs before every invalidation or authoritative write;
- body prose cannot add evidence weight or infer a missing mechanism;
- OOD decisions consume the eligible primary event, not the whole document;
- pending IDs contain semantic and origin fingerprints;
- exact confirmation/retraction matching with no claim-only fallback;
- origin-ambiguous retractions abstain instead of deleting several reports;
- fixed adversarial and metamorphic tests.

## What cannot be solved inside this contract

A perfectly fabricated assertion paired with equally fabricated “trusted”
provenance is indistinguishable from a truthful item using only the supplied
body and metadata. A production anti-falsification boundary must authenticate and
bind at least:

```text
issuer + study ID + source state + relation + destination state + polarity
+ raw artifact hash + provenance fields + timestamp + retraction linkage
```

A signature proves origin and integrity, not scientific honesty. Truth confidence
still requires raw artifacts, independent authenticated studies, replication, or
curated external evidence.
