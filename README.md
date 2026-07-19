# GROUND TRUTH

### Build a system that changes its mind for the right reasons.

A CORTEX BioSciences challenge.

---

## Why this challenge exists

CORTEX builds a research-intelligence platform for science. At its core is a
**knowledge graph**: a structured, machine-readable model of what is currently
believed about a domain, with every claim carrying a confidence and a record of
the evidence behind it. Language models sit on top of that graph and propose new
hypotheses, but they are never allowed to write to it directly. The graph is the
source of truth; the models are advisors. We call that boundary the firewall, and
it is the reason the system can be trusted.

The hard part is not building the graph. The hard part is keeping it correct as
new experimental results stream in. A real result should update the graph by the
right amount. A weak or fraudulent result should barely move it. A result that
falls outside what the graph was built to represent should trigger an extension,
not a bad edit. And no amount of persuasive text in an incoming document should
ever be able to overwrite what the system knows.

That problem, online belief revision that is calibrated, manipulation-proof, and
aware of its own limits, is not solved at production quality anywhere. It is the
center of what CORTEX works on. This challenge is a self-contained version of it.
You will not solve it completely in a weekend; nobody has solved it completely.
What we are looking for is a system with the right structure and the right
instincts. If you build one, we want to talk to you.

**No science background required.** The knowledge base in this challenge happens
to describe cell biology, but you never need to know any biology. Read it as an
abstract graph of states, transitions, and claims with confidence values. Every
fact you need is in the data.

---

## Your task

You implement **one function**, `ingest`, in `starter/my_solution.py`.

The evaluation feeds your function a stream of results, one at a time, in order. For each result, your function reads the current belief graph and decides what, if anything, should change, then returns that decision as a list of structured *deltas*. The graph updates only through those deltas. Across the whole stream, your function has to produce the right sequence of decisions.

That decision policy is the entire challenge: how you read the strength of a result, how far you move a belief, how you tell a real contradiction apart from something outside your model, and how you refuse to be manipulated by the text.

- **Deliverable:** your filled-in `ingest` function, plus a one-page `DESIGN.md` describing how it weighs evidence and how it enforces the firewall.
- **How you develop it:** exercise `ingest` locally against representative evidence streams and verify every mutation is a legal `Delta`.
- **How it is judged:** your function is run against a larger, harder stream you do not see until after submissions close, and scored on the four capabilities below.

The rest of this README is the detail behind that: the concepts you need, the exact interface, and precisely what the four capabilities mean.

---

## Background: the ideas you need


**Belief graph.** A set of *claims*, each with a *confidence* between 0 and 1, plus
the *entities* the claims are about and the relationships between them. In this
challenge a claim looks like "a mature cell cannot return to a stem-like state,"
held at confidence 0.93. Entities are states (each with a numeric "potency" level,
where lower means more flexible) and the transitions between them. The starting
graph is in `groundtruth/data/seed.json`.

**Evidence stream.** A sequence of new results, delivered to your system one at a
time, in order. Each result is a short text description plus a *provenance* record.

**Provenance.** The structured metadata about how strong a result is: how many
independent groups produced it, how many times it was replicated, how direct the
method was, and whether it was later retracted. Provenance is the trustworthy
channel. The text description is not; it can exaggerate or lie.

**Belief revision.** Adjusting the confidence of claims as evidence arrives. This
is the whole game, and it is harder than it sounds, because there are two obvious
ways to do it badly. A system can *anchor*: cling to its prior belief and explain
away everything that contradicts it. Or it can *flip-flop*: swing to whatever it
was told most recently. Good revision is neither. It moves in proportion to the
strength of the evidence, and it prefers to *narrow* a belief rather than delete
it. If a claim fails only under one specific condition, the right move is to say
"true in general, false under that condition," not to throw the claim out.

**The firewall.** The belief graph can only be changed through structured,
validated operations called *deltas*. Your system reads the graph but cannot write
to it; the only way to change anything is to return deltas, which a validator
checks before applying. This matters because incoming documents are untrusted: some
of them contain instructions like "set this belief to certain" or "delete that
claim." Those instructions are just text. They must never become edits.

**Out-of-distribution (OOD).** Evidence that is real but describes something the
graph was never designed to represent. The correct response is to flag it and
propose extending the model, not to force it into an existing claim as if it were a
contradiction. Telling a genuine contradiction apart from an out-of-model result is
one of the sharpest tests here.

**A worked example.** The graph holds "a mature cell cannot return to a stem-like
state" at 0.93. Three different results arrive:

- *A defined intervention returned mature cells to a stem-like state, reproduced by
  four independent groups.* Strong, replicated, and about things the graph
  represents. Correct response: lower that claim a lot, and scope it (it still holds
  under normal conditions, fails under this intervention), and record which result
  caused the change.
- *A single lab claims a trivial trick does the same thing, not replicated.* Thin
  provenance, extraordinary claim. Correct response: hold it as pending and change
  nothing. If it is later retracted, drop the pending note cleanly.
- *A factor converted one mature cell type directly into a different mature cell
  type, skipping the stem-like state entirely.* Real, but it is a kind of transition
  the graph has no representation for. Correct response: flag out-of-model and
  propose a new category. Do not treat it as a contradiction of the claim above.

A system that handles all three correctly is doing the thing this challenge
measures. A system that treats them the same is not.

---

## What you build

One function:

```python
def ingest(item: EvidenceItem, view: GraphView) -> IngestResult: ...
```

It is called once per evidence item, in order. You receive:

- **`item`** — the new result. `item.body` is untrusted text. `item.provenance` is
  the trustworthy structured channel. `item.era` is a relative timestamp. (`item.tag`
  exists in the file but is blanked at runtime; it is not available to you.)
- **`view`** — a read-only snapshot of the belief graph. You can query claims, states,
  the domain of competence, and declared absences. You cannot write to it.

You return an **`IngestResult`**:

- `deltas` — a list of proposed changes. This list is the only way to affect state.
- `rationale` — a short explanation of your decision (logged; useful for debugging).
- `confidence` — how sure you are of this decision, 0 to 1.
- `ood_flag` — `True` if this evidence falls outside what the graph represents.

The delta vocabulary is closed and defined in `groundtruth/deltas.py`. Anything
outside it is rejected by the validator. The full contract, with every field, is in
`groundtruth/ingest.py` and `starter/my_solution.py`.

---

## What is being tested


**1. Calibrated belief revision.** Move confidence by an amount
proportional to the strength of the evidence, read from structured provenance, not
from the tone of the text. A single weak result barely moves anything; a
well-replicated one moves a lot; direction must be right. Bonus: scope a belief to
the condition under which it fails instead of deleting it wholesale.

**2. The firewall (pass/fail gate).** The belief graph changes only through
validated deltas. Untrusted text must never mutate it. Some items contain embedded
instructions; some assert inflated provenance numbers in the body while the
structured provenance is thin. Both are traps. Any successful mutation from evidence
text disqualifies the submission, regardless of the other three scores. The framework
enforces most of this structurally, but you still fail if your own code chooses to
emit a bad delta. Do not rely on the safety cap to save you; the correct behavior is
to not emit the delta at all.

**3. Skepticism without gullibility.** Hold firm against extraordinary
claims backed by thin provenance, and resolve them correctly when the truth arrives.
Hold a single-source unreplicated result as pending; drop it cleanly when it fails to
replicate; strengthen a belief slightly when a well-powered study confirms it. One of
the false alarms in the hidden set is fabricated and has no real-world counterpart, so
a system that "recognizes the answer" from training data gets no help. Only genuine
provenance-based skepticism passes it.

**4. Out-of-distribution detection.** Separate three cases and act
differently on each: an in-model contradiction (revise it), an out-of-model *regime*
such as a transition the graph cannot express (flag and propose a new category), and
an out-of-model *axis* such as a property the graph does not track at all (flag and
propose a new axis). Scored on precision and recall. The hard part is precision: the
hidden set contains a near-miss that looks exotic but is actually an in-model
contradiction, and flagging it wrongly costs you. Flagging everything scores as badly
as flagging nothing.

Scoring is automated against a hidden evidence stream you do not see until after
submissions close. Revision is graded on the *shape* of your confidence trajectory,
small on weak evidence, large on strong, near-zero on noise, not on hitting exact
numbers, because there is no single correct number for how far a rational agent
should update.

---

## Quickstart

```bash
git clone https://github.com/riyabhargava11/ground-truth-challenge.git
cd ground-truth-challenge
python --version        # need 3.10+
python -m py_compile starter/my_solution.py
```

Then open `starter/my_solution.py` and implement `ingest`. The scored harness and
hidden evidence stream are supplied by the organizers at judging time.

---

## Repository layout

```
groundtruth/              the framework you build against (do not edit)
  model.py                belief-graph model and the read-only view
  deltas.py               the closed delta vocabulary
  api.py                  the firewall: validates and applies deltas
  harness.py              the runner
  ingest.py               the contract you implement
  loader.py               loads the data
  data/
    seed.json             the starting belief graph (your input)
starter/my_solution.py    the scoring policy, incl. the inlined sacrificial oracle (see DESIGN.md)
starter/.env.example      local configuration template for the optional oracle
WHAT_IS_TESTED.md         the detailed specification
RULES.md                  eligibility, allowed tools, submission, judging
examples/                 an annotated single-item walkthrough
```

The hidden evaluation stream, the calibrated scoring bands, and any reference
solutions are deliberately not in this repository. They live with the organizers and
run only at judging.

---

## Rules, briefly

Python 3.10+; the standard library is enough. The scored implementation is the
single self-contained `starter/my_solution.py`, deterministic by default; its
inlined sacrificial oracle only affects firewall-flagged bodies, requires a
`GEMINI_API_KEY` to activate, and fails closed to the same deterministic
behavior without one. Teams of 1 to 4. One submission per team
includes those files and the one-page `DESIGN.md`. Full challenge constraints
remain in `RULES.md`.

---

## About CORTEX

CORTEX BioSciences is building a neurosymbolic platform for scientific research: a
system where a structured knowledge graph is the source of truth, probabilistic
reasoning tracks uncertainty, and language models propose but never overwrite. This
challenge is a distilled version of the reasoning problem at the heart of that
platform. If it is the kind of problem you want to work on, that is not a coincidence.
