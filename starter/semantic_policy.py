"""Semantic, fail-closed evidence policy for the GROUND TRUTH challenge.

The body is an untrusted data channel.  It may propose an atomic scientific
event, but it cannot choose graph operations, claim IDs, confidence values, or
evidence weight.  This module separates five stages:

1. normalize and firewall the untrusted body;
2. extract and admit clause-local, typed scientific events;
3. validate structured provenance using anchored grammars and exact enums;
4. map admitted events to bounded typed deltas deterministically;
5. attach a closed evidence class so every decision is auditable.

The small composite control-pattern gate is defense in depth.  It deliberately
does not block individual words such as "model", "confidence", or "direct".
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import re
import unicodedata
from typing import Any, Iterable

from groundtruth.deltas import Delta, no_op
from groundtruth.ingest import EvidenceItem, IngestResult
from groundtruth.model import Claim, GraphView, logit, sigmoid


class SpeechAct(str, Enum):
    RESULT = "result"
    INSTRUCTION = "instruction"
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    BACKGROUND = "background"
    UNKNOWN = "unknown"


class Proposition(str, Enum):
    POTENCY_REVERSAL = "potency_reversal"
    DIFFERENTIATION = "differentiation"
    NUCLEAR_RETENTION = "nuclear_retention"
    CELL_TRANSITION = "cell_transition"
    PROPERTY_CHANGE = "property_change"
    UNKNOWN = "unknown"


class Polarity(str, Enum):
    AFFIRMED = "affirmed"
    DENIED = "denied"
    UNKNOWN = "unknown"


class Directness(str, Enum):
    DIRECT = "direct"
    SEMI_DIRECT = "semi_direct"
    INDIRECT = "indirect"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class EffectStrength(str, Enum):
    NULL = "null"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    UNKNOWN = "unknown"


class EvidenceClass(str, Enum):
    """Closed audit taxonomy for the policy's final disposition.

    The class explains why a delta was or was not emitted.  It never selects a
    graph target or authorizes an operation; those choices remain downstream of
    semantic admission and structured-provenance validation.
    """

    INJECTION = "INJECTION"
    INVALID = "INVALID"
    NON_EVIDENCE = "NON_EVIDENCE"
    NULL_REJECT = "NULL_REJECT"
    INVALIDATION = "INVALIDATION"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    CONTRADICTION = "CONTRADICTION"
    CONFIRMATION = "CONFIRMATION"
    OOD = "OOD"
    SATURATED = "SATURATED"


@dataclass(frozen=True)
class StateMention:
    start: int
    end: int
    state: Any


@dataclass(frozen=True)
class SemanticEvent:
    clause_index: int
    clause: str
    speech_act: SpeechAct
    proposition: Proposition
    polarity: Polarity
    observed: bool
    source: Any | None
    destination: Any | None
    property_axis: str | None = None
    ood_regime: str | None = None
    has_intermediate: bool = False
    failed_replication: bool = False
    lineage_restriction: bool = False
    ambiguous: bool = False

    @property
    def eligible(self) -> bool:
        basic = (
            self.speech_act is SpeechAct.RESULT
            and self.observed
            and self.proposition is not Proposition.UNKNOWN
            and self.polarity is not Polarity.UNKNOWN
            and not self.ambiguous
        )
        if not basic:
            return False
        # Identity transitions must name both roles.  Vague prose such as
        # "reprogramming increased potency" is not allowed to choose a graph
        # target merely because it contains a recognized biological phrase.
        if self.proposition is Proposition.POTENCY_REVERSAL:
            return self.source is not None and (
                self.destination is not None
                or bool(
                    re.search(
                        r"\b(?:less[ -]?committed|less differentiated|increased? potency|"
                        r"higher potency|dedifferentiat\w*)\b",
                        self.clause,
                    )
                )
            )
        if self.proposition is Proposition.DIFFERENTIATION:
            return self.source is not None and (
                self.destination is not None
                or bool(re.search(r"\b(?:somatic|speciali[sz]ed|terminal)\s+(?:cell|cells|lineage|lineages)\b", self.clause))
            )
        if self.proposition is Proposition.CELL_TRANSITION:
            return self.source is not None and self.destination is not None
        return True

    @property
    def to_source(self) -> bool:
        if self.destination is not None:
            return self.destination.potency_level <= 1
        return self.proposition is Proposition.POTENCY_REVERSAL and bool(
            re.search(
                r"\b(?:pluripot\w*|stemness|stem[ -]?like|source state|sourceState|"
                r"embryonic[ -]?like|ipscs?|ips cells?)\b",
                self.clause,
                re.IGNORECASE,
            )
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    score: float
    groups: float
    replications: float
    directness: Directness
    effect: EffectStrength
    effect_value: float
    effect_reported: bool
    explicit_null_effect: bool
    method_class: str
    mechanism: str
    retracted: bool
    valid: bool
    issues: tuple[str, ...]

    @property
    def thin(self) -> bool:
        return self.groups <= 1 and self.replications <= 1

    @property
    def credible(self) -> bool:
        return self.valid and self.score >= 0.60 and self.groups >= 2

    @property
    def strong(self) -> bool:
        return self.valid and self.score >= 0.82 and self.groups >= 3


@dataclass(frozen=True)
class BodyAssessment:
    events: tuple[SemanticEvent, ...]
    direct_control_attempt: bool
    instruction_event: bool
    ambiguous: bool
    malformed: bool = False


_COUNT_WORDS = {
    "none": 0.0,
    "zero": 0.0,
    "single": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "few": 2.0,
    "couple": 2.0,
    "handful": 3.0,
    "multiple": 3.0,
    "several": 4.0,
    "many": 8.0,
    "numerous": 8.0,
    "dozen": 8.0,
    "dozens": 8.0,
    "tens": 8.0,
    "hundreds": 8.0,
}


_CONFUSABLES = str.maketrans(
    {
        # Common Cyrillic/Greek characters used to disguise control-plane words.
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "у": "y",
        "х": "x",
        "і": "i",
        "ј": "j",
        "ѕ": "s",
        "Α": "a",
        "Β": "b",
        "Ε": "e",
        "Ζ": "z",
        "Η": "h",
        "Ι": "i",
        "Κ": "k",
        "Μ": "m",
        "Ν": "n",
        "Ο": "o",
        "Ρ": "p",
        "Τ": "t",
        "Υ": "y",
        "Χ": "x",
    }
)


def _normalized(value: Any, *, fold_confusables: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Cf")
    value = value.casefold()
    if fold_confusables:
        value = value.translate(_CONFUSABLES)
    return re.sub(r"\s+", " ", value).strip()


def _label(value: Any) -> str:
    return re.sub(r"[\s_-]+", " ", _normalized(value)).strip()


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", _normalized(str(value))).strip("_")
    return text[:64] or "unknown"


def _contains(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


_CONTROL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        # A privileged/control role combined with an instruction container.
        r"(?:^|[\[\(<{;:]\s*)\b(?:system|developer|assistant|processor|curator|"
        r"moderator|evaluator|judge)\s+(?:message|note|memo|instruction|directive|command)\b",
        # An override action combined with a protected control channel.
        r"\b(?:ignore|disregard|override|bypass|supersede)\b.{0,80}\b(?:system|developer|"
        r"assistant|processor|previous|prior|hidden|higher[ -]?priority)?\s*(?:instruction|"
        r"instructions|rules|policy|policies|directive|command|guardrail|prompt|provenance channel)\b",
        # A mutation action combined with a graph/control target. Confidence
        # interval is deliberately excluded because it is ordinary science.
        r"\b(?:set|change|update|revise|overwrite|delete|drop|add|mutate|emit|force|mark)\b"
        r".{0,80}\b(?:claim|belief|knowledge graph|graph state|delta|pending item|ood flag|"
        r"out of distribution|confidence(?!\s+interval))\b",
        # Output shaping combined with a claim/probability target.
        r"\b(?:desired|required|target)\s+(?:answer|output|result)\b.{0,100}\b(?:probability|"
        r"claim|confidence(?!\s+interval)|belief|(?:q|c)\d+[a-z]?)\b",
        # Imperative epistemic coercion at a clause boundary.
        r"(?:^|[\[\(<{;:]\s*)\b(?:please|kindly)?\s*(?:accept|deem|regard|register|record|"
        r"take)\b.{0,60}\b(?:as\s+(?:fact|true|established|certain|ground truth|gospel)|for granted)\b",
    )
)


def _direct_control_attempt(body: str) -> bool:
    folded = _normalized(body, fold_confusables=True).replace("_", " ")
    return any(pattern.search(folded) is not None for pattern in _CONTROL_PATTERNS)


_IMPERATIVE_START = re.compile(
    r"^(?:please|kindly)?\s*(?:summarize|write|explain|rewrite|compose|produce|generate|"
    r"convert|classify|accept|assume|imagine|pretend|deem|regard|register|record|output|"
    r"respond|show|tell|return|set|update|change|revise|ignore|disregard|override|"
    r"contemplate|consider|believe|conclude|treat)\b",
    re.IGNORECASE,
)


def _speech_act(clause: str) -> SpeechAct:
    text = clause.strip()
    if not text:
        return SpeechAct.UNKNOWN
    if _IMPERATIVE_START.search(text):
        return SpeechAct.INSTRUCTION
    # A negative command is still a command.  Without this structural check,
    # "Do not conclude that X" looked like a negated scientific result.
    if re.search(r"^do\s+not\s+[a-z]+\b", text):
        return SpeechAct.INSTRUCTION
    # Generic imperative frame: the leading verb itself is not enumerated.  The
    # protected object phrase supplies the context, so unseen verbs such as
    # "chronicle" or "paraphrase" cannot smuggle a scientific assertion.
    if re.search(
        r"^[a-z]+\s+(?:this|that|the following|the (?:claim|conclusion|answer|output|"
        r"report|description|text|paragraph|statement))\b",
        text,
    ):
        return SpeechAct.INSTRUCTION
    if re.search(
        r"\b(?:your task|the task is|you are to|you must|the processor should|"
        r"for downstream processing)\b",
        text,
    ):
        return SpeechAct.INSTRUCTION
    if "?" in text or re.search(
        r"\b(?:asked|asks|questioned|investigated|tested|examined|wondered)\s+whether\b|"
        r"\b(?:asked|urged|told)\b.{0,50}\bto\s+(?:believe|accept|assume)\b",
        text,
    ):
        return SpeechAct.QUESTION
    if re.search(
        r"\b(?:hypothesis|hypothesized|speculated|proposal|proposed mechanism|"
        r"designed to test|aimed to test|future work|might|may have|could potentially|"
        r"suppose|imagined scenario)\b",
        text,
    ):
        return SpeechAct.HYPOTHESIS
    if re.search(
        r"\b(?:background|for comparison|as an example|example sentence|grant proposal|"
        r"paper title|figure legend|control description|according to)\b|"
        r"\b(?:alleged|claimed|asserted|purported|rumou?red|insisted|maintained)\s+(?:that\s+)?",
        text,
    ):
        return SpeechAct.BACKGROUND
    return SpeechAct.RESULT


def _has_unbalanced_delimiters(text: str) -> bool:
    """Reject truncated quotations/brackets instead of parsing their payload."""
    normalized = _normalized(text)
    if normalized.count('"') % 2 or normalized.count("“") != normalized.count("”"):
        return True
    if normalized.count("‘") != normalized.count("’"):
        return True
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in normalized:
        if character in "([{":
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                return True
    return bool(stack)


def _strip_quoted_material(text: str) -> tuple[str, bool]:
    quoted = False

    def replace(match: re.Match[str]) -> str:
        nonlocal quoted
        quoted = True
        return " " * len(match.group(0))

    # Scientific prose uses both straight and curly quotation marks. Apostrophes
    # inside words do not match because a nonempty closing quote is required.
    patterns = (
        r'"[^"\n]*"',
        r"'[^'\n]+'",
        r"“[^”\n]*”",
        r"‘[^’\n]*’",
    )
    result = text
    for pattern in patterns:
        result = re.sub(pattern, replace, result)
    return result, quoted


def _split_clauses(body: str) -> list[str]:
    text = _normalized(body).replace("_", " ")
    # Semicolons and sentence boundaries separate event attachment. Colons stay
    # inside a clause so role/directive prefixes remain visible to the gate.
    parts = re.split(r"(?<=[.!?;])\s+|\s*;\s*", text)
    return [part.strip() for part in parts if part.strip()]


def _parse_count(value: Any, field: str) -> tuple[float, str | None]:
    if isinstance(value, bool):
        return 0.0, f"{field} must not be boolean"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            return 0.0, f"{field} is not a finite nonnegative integer count"
        return min(8.0, numeric), None
    if not isinstance(value, str):
        return 0.0, f"{field} has unsupported type"

    text = _label(value)
    qualifier = ""
    match = re.fullmatch(
        r"(?:(?P<qualifier>approximately|approx|about|around|roughly|over|more than|"
        r"at least|nearly|almost|under|fewer than)\s+)?"
        r"(?P<count>\d+(?:\.\d+)?|none|zero|single|one|two|three|four|five|six|seven|"
        r"eight|few|couple|handful|multiple|several|many|numerous|dozen|dozens|tens|hundreds)"
        r"(?:\s+(?:independent\s+)?(?:groups?|labs?|laboratories|replications?|repeats?|times?))?",
        text,
    )
    if match is None:
        return 0.0, f"{field} is not a recognized count"
    qualifier = match.group("qualifier") or ""
    raw_count = match.group("count")
    numeric = _COUNT_WORDS.get(raw_count, float(raw_count) if re.fullmatch(r"\d+(?:\.\d+)?", raw_count) else 0.0)
    if not numeric.is_integer():
        return 0.0, f"{field} is not an integer count"
    # Upper-bound phrases must not be treated as proof of the upper bound.
    if qualifier in {"under", "fewer than", "nearly", "almost"}:
        numeric = max(0.0, numeric - 1.0)
    return min(8.0, numeric), None


def _parse_directness(value: Any) -> tuple[Directness, str | None]:
    text = _label(value)
    aliases = {
        "direct": Directness.DIRECT,
        "direct measurement": Directness.DIRECT,
        "semi direct": Directness.SEMI_DIRECT,
        "semidirect": Directness.SEMI_DIRECT,
        "indirect": Directness.INDIRECT,
        "indirect measurement": Directness.INDIRECT,
        "inferred": Directness.INFERRED,
        "inferential": Directness.INFERRED,
        "not direct": Directness.INDIRECT,
    }
    if text in aliases:
        return aliases[text], None
    return Directness.UNKNOWN, "method_directness is not a recognized enum"


def _parse_effect(value: Any) -> tuple[EffectStrength, float, bool, str | None]:
    if isinstance(value, bool):
        return EffectStrength.UNKNOWN, 0.0, False, "effect_strength must not be boolean"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric == 0:
            return EffectStrength.NULL, 0.0, True, None
        return EffectStrength.UNKNOWN, 0.0, True, "numeric effect is only defined for zero"
    text = _label(value)
    nulls = {
        "none",
        "null",
        "no effect",
        "no measurable effect",
        "no observed effect",
        "absent",
        "zero",
    }
    if text in nulls:
        return EffectStrength.NULL, 0.0, True, None
    aliases = {
        "weak": (EffectStrength.WEAK, 0.3),
        "small": (EffectStrength.WEAK, 0.3),
        "weak effect": (EffectStrength.WEAK, 0.3),
        "small effect": (EffectStrength.WEAK, 0.3),
        "moderate": (EffectStrength.MODERATE, 0.65),
        "medium": (EffectStrength.MODERATE, 0.65),
        "moderate effect": (EffectStrength.MODERATE, 0.65),
        "strong": (EffectStrength.STRONG, 1.0),
        "large": (EffectStrength.STRONG, 1.0),
        "strong effect": (EffectStrength.STRONG, 1.0),
        "large effect": (EffectStrength.STRONG, 1.0),
    }
    if text in aliases:
        effect, numeric = aliases[text]
        return effect, numeric, True, None
    return EffectStrength.UNKNOWN, 0.0, bool(text), "effect_strength is not a recognized enum"


def _parse_method(value: Any) -> tuple[str, str, float, str | None]:
    text = _label(value)
    if not text:
        return "", "unspecified", 0.0, "method_class is missing"

    # Exact semantic classes, ordered so negated substrings cannot inflate trust.
    if re.fullmatch(r"(?:nonrandomized|non randomized)(?: observational)?(?: study)?", text):
        return text, "unspecified", 0.55, None
    if re.fullmatch(r"(?:defined factor|transcription factor)(?: expression| perturbation)?", text):
        return text, "defined_factor", 0.95, None
    if re.fullmatch(r"environmental stress(?: perturbation)?|stress perturbation", text):
        return text, "env_stress", 0.95, None
    if re.fullmatch(r"(?:somatic cell )?(?:nuclear|oocyte) transfer|scnt|cloning", text):
        return text, "oocyte_nt", 0.95, None
    if re.fullmatch(r"lineage tracing|lineage perturbation", text):
        return text, "lineage_tracing", 0.95, None
    if re.fullmatch(r"randomized(?: controlled)?(?: perturbation| study| trial)?", text):
        return text, "randomized", 0.95, None
    if re.fullmatch(r"observational(?: study)?", text):
        return text, "observational", 0.55, None
    if re.fullmatch(r"spontaneous(?: observation)?", text):
        return text, "spontaneous", 0.70, None
    return text, "unspecified", 0.0, "method_class is not a recognized enum"


def _parse_retraction(value: Any) -> tuple[bool, str | None]:
    if isinstance(value, bool):
        return value, None
    text = _label(value)
    if text in {"", "none", "no", "false", "active", "not retracted"}:
        return False, None
    if text in {
        "yes",
        "true",
        "retracted",
        "withdrawn",
        "rescinded",
        "invalidated",
        "later retracted",
        "paper retracted",
        "fraud confirmed",
    }:
        return True, None
    return False, "retraction_status is not a recognized enum"


def _saturation(count: float) -> float:
    if count <= 1:
        return 0.0
    if count == 2:
        return 0.4
    if count == 3:
        return 0.75
    return 1.0


def _provenance(item: EvidenceItem) -> ProvenanceRecord:
    raw = dict(item.provenance) if isinstance(item.provenance, dict) else {}
    issues: list[str] = []

    groups, issue = _parse_count(raw.get("independent_groups"), "independent_groups")
    if issue:
        issues.append(issue)
    replications, issue = _parse_count(raw.get("replication_count"), "replication_count")
    if issue:
        issues.append(issue)
    directness, issue = _parse_directness(raw.get("method_directness"))
    if issue:
        issues.append(issue)
    effect, effect_value, effect_reported, issue = _parse_effect(raw.get("effect_strength"))
    if issue:
        issues.append(issue)
    method_class, mechanism, reliability, issue = _parse_method(raw.get("method_class"))
    if issue:
        issues.append(issue)
    retracted, issue = _parse_retraction(raw.get("retraction_status"))
    if issue:
        issues.append(issue)

    directness_value = {
        Directness.DIRECT: 1.0,
        Directness.SEMI_DIRECT: 0.75,
        Directness.INDIRECT: 0.45,
        Directness.INFERRED: 0.25,
        Directness.UNKNOWN: 0.0,
    }[directness]
    score = (
        0.40 * _saturation(groups)
        + 0.15 * _saturation(replications)
        + 0.20 * directness_value
        + 0.15 * effect_value
        + 0.10 * reliability
    )
    return ProvenanceRecord(
        score=min(1.0, max(0.0, score)),
        groups=groups,
        replications=replications,
        directness=directness,
        effect=effect,
        effect_value=effect_value,
        effect_reported=effect_reported,
        explicit_null_effect=effect is EffectStrength.NULL,
        method_class=method_class,
        mechanism=mechanism,
        retracted=retracted,
        valid=not issues,
        issues=tuple(issues),
    )


def _singular(word: str) -> str:
    irregular = {"cells": "cell", "states": "state", "identities": "identity"}
    if word in irregular:
        return irregular[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _state_mentions(clause: str, view: GraphView) -> tuple[StateMention, ...]:
    found: dict[str, StateMention] = {}

    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*\b", clause):
        state = view.cell_state(match.group(0))
        if state is not None and state.id not in found:
            found[state.id] = StateMention(match.start(), match.end(), state)

    words = list(re.finditer(r"[a-z0-9]+", clause))
    for start in range(len(words)):
        for width in range(1, min(5, len(words) - start) + 1):
            parts = [word.group(0) for word in words[start : start + width]]
            variants = (parts, [*parts[:-1], _singular(parts[-1])])
            for variant in variants:
                candidate = "".join(part.capitalize() for part in variant)
                state = view.cell_state(candidate)
                if state is not None and state.id not in found:
                    found[state.id] = StateMention(
                        words[start].start(), words[start + width - 1].end(), state
                    )

    aliases = {
        r"\b(?:induced )?pluripotent(?: stem)? cells?\b": "PluripotentStemCell",
        r"\bips cells?\b|\bipscs?\b": "PluripotentStemCell",
        r"\bstem cells?\b": "PluripotentStemCell",
        r"\b(?:pluripotent|stem[ -]?like) (?:state|identity|phenotype)\b": "PluripotentStemCell",
        r"\bstemness\b": "PluripotentStemCell",
        r"\bmuscle cells?\b|\bskeletal muscle\b": "SkeletalMuscleCell",
        r"\bintestinal(?: epithelial)? cells?\b": "IntestinalEpithelialCell",
    }
    for pattern, state_name in aliases.items():
        match = re.search(pattern, clause)
        if match is None:
            continue
        state = view.cell_state(state_name)
        if state is not None and state.id not in found:
            found[state.id] = StateMention(match.start(), match.end(), state)

    return tuple(sorted(found.values(), key=lambda mention: mention.start))


_REVERSAL_FRAME = re.compile(
    r"\b(?:return(?:ed|s|ing)?(?:\s+back)?(?:\s+to|\b.{1,70}\bto)|"
    r"revert(?:ed|s|ing)?(?:\s+back)?(?:\s+to|\b.{1,70}\bto)|"
    r"dedifferentiat\w*(?:\s+to|\s+into)?|"
    r"reprogram\w*(?:(?:\s+to|\s+into)|\b.{1,70}\b(?:to|into))|"
    r"regain\w*\s+(?:a\s+)?(?:pluripot\w*|stemness|stem[ -]?like)|"
    r"acquir\w*\s+(?:a\s+)?(?:pluripot\w*|stemness|stem[ -]?like)|"
    r"recover\w*\s+(?:a\s+)?(?:pluripot\w*|stemness|stem[ -]?like)|"
    r"restor\w*.{0,25}(?:pluripot\w*|stemness|stem[ -]?like)|"
    r"(?:became|become|becomes)\s+(?:a\s+)?(?:pluripot\w*|stem[ -]?like)|"
    r"reset\w*\s+to|de[ -]?speciali[sz]\w*|rolled?\s+back.{0,30}(?:lineage|commitment)|"
    r"less[ -]?committed|increase\w*\s+(?:in\s+)?potency)\b"
)

_TRANSITION_FRAME = re.compile(
    r"\b(?:convert(?:ed|s|ing)?|conversion|transdifferentiat\w*|reprogram\w*|"
    r"turn(?:ed|s|ing)?\s+into|transform(?:ed|s|ing)?\s+into|switch(?:ed|es|ing)?\s+(?:to|into))\b"
)

_DIFFERENTIATION_FRAME = re.compile(
    r"\b(?:differentiat\w*|produced?\s+downstream|downstream states?|somatic lineages?|"
    r"lineage restriction|progressive restriction|speciali[sz]\w*)\b"
)

_PASSIVE_FROM_FRAME = re.compile(
    r"\b(?:was|were|is|are|had been|have been)\s+"
    r"(?:produced|generated|derived|obtained|created|induced)\s+from\b"
)


def _role_states(
    clause: str,
    mentions: tuple[StateMention, ...],
    predicate_start: int,
) -> tuple[Any | None, Any | None, bool]:
    if not mentions:
        return None, None, False

    passive = _PASSIVE_FROM_FRAME.search(clause)
    if passive is not None:
        before = [m for m in mentions if m.end <= passive.start()]
        after = [m for m in mentions if m.start >= passive.end()]
        destination = before[-1].state if before else None
        source = after[0].state if after else None
        return source, destination, source is None or destination is None

    before = [m for m in mentions if m.end <= predicate_start]
    after = [m for m in mentions if m.start >= predicate_start]
    source = before[-1].state if before else mentions[0].state
    destination = next(
        (m.state for m in after if m.state.id != source.id),
        None,
    )
    if destination is None and len(mentions) >= 2:
        destination = next((m.state for m in mentions if m.state.id != source.id), None)
    return source, destination, len({m.state.id for m in mentions}) > 2


def _denied_near(clause: str, predicate_start: int) -> bool:
    prefix = clause[max(0, predicate_start - 90) : predicate_start]
    combined = clause[max(0, predicate_start - 120) : predicate_start + 100]
    if re.search(
        r"\b(?:unable to|failed to|could not|cannot|can not|did not|does not|do not|"
        r"never|no evidence (?:that|of)|without evidence (?:that|of)|was not|were not)\b",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:rejected|refuted|disproved|ruled out|falsified|did not support)\b.{0,70}"
        r"\b(?:claim|report|hypothesis|idea)?\b",
        prefix,
    ):
        return True
    if re.search(r"\b(?:no such transition|no transition occurred|effect was absent)\b", combined):
        return True
    return False


def _failed_replication(clause: str) -> bool:
    match = re.search(
        r"\b(?:failed to replicate|failed to reproduce|could not replicate|could not reproduce|"
        r"did not replicate|did not reproduce|no effect was found|no such effect)\b",
        clause,
    )
    if match is None:
        return False
    prefix = clause[max(0, match.start() - 20) : match.start()]
    return re.search(r"\b(?:not|never)\b", prefix) is None


def _identity_preserved(clause: str) -> bool:
    return bool(
        re.search(
            r"\b(?:identity|cell type|lineage)\b.{0,35}\b(?:unchanged|preserved|retained|"
            r"intact|constant|fixed|conserved|maintained|the same)\b",
            clause,
        )
        or re.search(
            r"\b(?:without changing|without altering|while retaining|while preserving|kept the same)"
            r".{0,25}\b(?:identity|cell type|lineage)\b",
            clause,
        )
    )


def _property_event(
    index: int,
    clause: str,
    speech_act: SpeechAct,
    mentions: tuple[StateMention, ...],
) -> SemanticEvent | None:
    if not _identity_preserved(clause):
        return None
    source = mentions[0].state if len(mentions) == 1 else None

    age = re.search(
        r"\b(?:biological age|cellular age|epigenetic age|epigenetic clock|rejuvenat\w*|"
        r"younger|ageing|aging|senescen\w*|telomere\w*)\b",
        clause,
    )
    function = re.search(
        r"\b(?:cell function|functional performance|function\w*|contractil\w*|performance|"
        r"force generation|contraction strength|metabolic activity|atp production)\b",
        clause,
    )
    quiescence = re.search(
        r"\b(?:quiescen\w*|dorman\w*|activated state|activation state|phenotypic state|"
        r"identity[ -]?preserving state change)\b",
        clause,
    )
    other = re.search(
        r"\b(?:chromatin|epigenetic state|gene expression|transcriptional state|"
        r"metabolic state|morphology|cell size)\b",
        clause,
    )
    if age:
        axis, regime = "biological_age", None
    elif function:
        axis, regime = "cell_function_independent_of_identity", None
    elif quiescence:
        axis, regime = None, "identity_preserving_state_change"
    elif other:
        axis, regime = other.group(0).replace(" ", "_"), None
    else:
        return None
    return SemanticEvent(
        clause_index=index,
        clause=clause,
        speech_act=speech_act,
        proposition=Proposition.PROPERTY_CHANGE,
        polarity=Polarity.AFFIRMED,
        observed=speech_act is SpeechAct.RESULT,
        source=source,
        destination=None,
        property_axis=axis,
        ood_regime=regime,
        ambiguous=len(mentions) > 1,
    )


def _nuclear_event(
    index: int,
    clause: str,
    speech_act: SpeechAct,
    mentions: tuple[StateMention, ...],
) -> SemanticEvent | None:
    match = re.search(
        r"\b(?:full nuclear (?:developmental )?potential|nuclear developmental potential|"
        r"supported development|developmental competence)\b",
        clause,
    )
    if match is None:
        return None
    affirmed_retention = bool(
        re.search(
            r"\b(?:had not lost|has not lost|did not lose|never lost|retained|preserved|"
            r"maintained|supported development)\b",
            clause,
        )
    )
    denied_retention = bool(
        re.search(
            r"\b(?:lost|did not retain|failed to retain|could not retain|lacked)\b",
            clause,
        )
    ) and not affirmed_retention
    polarity = Polarity.DENIED if denied_retention else Polarity.AFFIRMED
    return SemanticEvent(
        clause_index=index,
        clause=clause,
        speech_act=speech_act,
        proposition=Proposition.NUCLEAR_RETENTION,
        polarity=polarity,
        observed=speech_act is SpeechAct.RESULT,
        source=mentions[0].state if mentions else None,
        destination=None,
        ambiguous=False,
    )


def _transition_event(
    index: int,
    clause: str,
    speech_act: SpeechAct,
    mentions: tuple[StateMention, ...],
) -> SemanticEvent | None:
    reversal = _REVERSAL_FRAME.search(clause)
    transition = _TRANSITION_FRAME.search(clause)
    differentiation = _DIFFERENTIATION_FRAME.search(clause)
    passive = _PASSIVE_FROM_FRAME.search(clause)

    predicate = reversal or transition or differentiation or passive
    if predicate is None:
        return None
    source, destination, role_ambiguity = _role_states(clause, mentions, predicate.start())

    # A passive "pluripotent cells were produced from Fibroblast" construction
    # is a potency reversal even though the destination precedes the predicate.
    if passive is not None and destination is not None and source is not None:
        proposition = (
            Proposition.POTENCY_REVERSAL
            if destination.potency_level < source.potency_level
            else Proposition.DIFFERENTIATION
        )
    elif source is not None and destination is not None:
        if destination.potency_level < source.potency_level:
            proposition = Proposition.POTENCY_REVERSAL
        elif (
            destination.potency_level == source.potency_level
            and destination.lineage_identity != source.lineage_identity
        ):
            proposition = Proposition.CELL_TRANSITION
        else:
            proposition = Proposition.DIFFERENTIATION
    elif reversal is not None:
        proposition = Proposition.POTENCY_REVERSAL
    elif differentiation is not None:
        proposition = Proposition.DIFFERENTIATION
    else:
        proposition = Proposition.UNKNOWN

    denied = _denied_near(clause, predicate.start())
    if re.search(r"\b(?:not|never)\s+(?:a\s+)?failed to replicate\b", clause) and re.search(
        r"\b(?:confirmed|reproduced|replicated|validated)\b", clause
    ):
        denied = False
    polarity = Polarity.DENIED if denied else Polarity.AFFIRMED
    explicit_no_intermediate = bool(
        re.search(
            r"\b(?:without (?:passing through )?(?:any |an )?intermediate|"
            r"without entering|without traversing|skipp\w*|bypass\w*)\b",
            clause,
        )
    )
    has_intermediate = not explicit_no_intermediate and bool(
        re.search(
            r"\b(?:through|via|intermediate|passing through|progenitor|stem cell|"
            r"source state|pluripotent|sequential (?:differentiation )?stages?|"
            r"multi[ -]?step|two[ -]?stage|stepwise)\b",
            clause,
        )
    )
    direct = bool(
        re.search(
            r"\b(?:direct(?:ly)?|without (?:passing|(?:any |an )?intermediate|entering|traversing)|"
            r"skipp\w*|bypass\w*|transdifferentiat\w*)\b",
            clause,
        )
    )
    if proposition is Proposition.CELL_TRANSITION and has_intermediate:
        proposition = Proposition.DIFFERENTIATION

    # Mentioning a question, example, or command makes the candidate ineligible;
    # it is not scientific OOD merely because its words describe a transition.
    observed = speech_act is SpeechAct.RESULT and not re.search(
        r"\b(?:no experiment occurred|no results? (?:were|was|have been) reported|"
        r"results? (?:are|were) pending)\b",
        clause,
    )
    if proposition is Proposition.CELL_TRANSITION and not direct and transition is None:
        role_ambiguity = True
    return SemanticEvent(
        clause_index=index,
        clause=clause,
        speech_act=speech_act,
        proposition=proposition,
        polarity=polarity,
        observed=bool(observed),
        source=source,
        destination=destination,
        has_intermediate=has_intermediate,
        failed_replication=_failed_replication(clause),
        lineage_restriction=bool(re.search(r"\b(?:lineage restriction|progressive restriction)\b", clause)),
        ambiguous=role_ambiguity and proposition not in {
            Proposition.POTENCY_REVERSAL,
            Proposition.DIFFERENTIATION,
        },
    )


def _extract_clause_event(index: int, raw_clause: str, view: GraphView) -> SemanticEvent | None:
    stripped, had_quote = _strip_quoted_material(raw_clause)
    clause = re.sub(r"\s+", " ", stripped).strip()
    speech_act = _speech_act(clause)
    if had_quote and not clause:
        return None
    mentions = _state_mentions(clause, view)

    # Property and nuclear assertions are semantically distinct from identity
    # transitions and should not be inferred from unrelated neighboring clauses.
    event = _property_event(index, clause, speech_act, mentions)
    if event is not None:
        return event
    event = _nuclear_event(index, clause, speech_act, mentions)
    if event is not None:
        return event
    return _transition_event(index, clause, speech_act, mentions)


def _event_signature(event: SemanticEvent) -> tuple[str, str, str, str]:
    source = event.source.id if event.source is not None else "unknown_source"
    destination = event.destination.id if event.destination is not None else "unknown_destination"
    return event.proposition.value, source, destination, event.property_axis or event.ood_regime or "none"


def _assess_body(body: str, view: GraphView) -> BodyAssessment:
    direct_control = _direct_control_attempt(body)
    if _has_unbalanced_delimiters(body):
        return BodyAssessment((), direct_control, False, False, True)
    events: list[SemanticEvent] = []
    instruction_event = False
    for index, clause in enumerate(_split_clauses(body)):
        stripped, _ = _strip_quoted_material(clause)
        if _speech_act(stripped) is SpeechAct.INSTRUCTION:
            instruction_event = True
            # A command is never also admitted as an experimental result.
            continue
        event = _extract_clause_event(index, clause, view)
        if event is None:
            continue
        if event.speech_act is SpeechAct.INSTRUCTION:
            instruction_event = True
            continue
        if event.eligible:
            events.append(event)

    # Dedupe equivalent clauses, but abstain on conflicting polarity or several
    # unrelated primary scientific results in one evidence item.
    unique: dict[tuple[str, str, str, str], SemanticEvent] = {}
    ambiguous = False
    for event in events:
        signature = _event_signature(event)
        previous = unique.get(signature)
        if previous is not None and previous.polarity is not event.polarity:
            ambiguous = True
        else:
            unique[signature] = event
    if len(unique) > 1:
        # A nuclear-retention consequence accompanying nuclear transfer is the
        # one supported compound result; other multi-event documents abstain.
        propositions = {event.proposition for event in unique.values()}
        if propositions != {Proposition.POTENCY_REVERSAL, Proposition.NUCLEAR_RETENTION}:
            ambiguous = True
    return BodyAssessment(tuple(unique.values()), direct_control, instruction_event, ambiguous)


def _claim_kind(claim: Claim) -> str:
    statement = _normalized(claim.statement)
    if claim.scope.get("mechanism_class"):
        return "scoped_no_return"
    if _contains(statement, ("cannot return", "cannot revert", "return to pluripotency")):
        return "no_return"
    if "do not increase potency" in statement or "monotonically" in statement:
        return "potency_monotonic"
    if _contains(statement, ("no direct transition", "distinct terminal", "distinct leaf")):
        return "no_lateral"
    if "nuclear developmental potential" in statement:
        return "nuclear_potential"
    if "differentiate into" in statement:
        return "differentiation"
    if "progressive lineage restriction" in statement:
        return "lineage_restriction"
    return "other"


def _claims(view: GraphView) -> list[Claim]:
    return [claim for cid in view.list_claim_ids() if (claim := view.get_claim(cid)) is not None]


def _first_kind(claims: list[Claim], *kinds: str) -> Claim | None:
    return next((claim for claim in claims if _claim_kind(claim) in kinds), None)


def _scoped_claim(claims: list[Claim], mechanism: str) -> Claim | None:
    aliases = {
        "defined_factor": {"defined_factor", "defined_factor_expression"},
        "env_stress": {"env_stress", "environmental_stress"},
        "oocyte_nt": {"oocyte_nt", "nuclear_transfer", "somatic_cell_nuclear_transfer"},
        "spontaneous": {"spontaneous"},
    }
    accepted = aliases.get(mechanism, {mechanism})
    for claim in claims:
        if _label(claim.scope.get("mechanism_class")) in {value.replace("_", " ") for value in accepted}:
            return claim
    return None


def _targets(
    event: SemanticEvent,
    provenance: ProvenanceRecord,
    claims: list[Claim],
) -> list[tuple[Claim, int, str]]:
    targets: list[tuple[Claim, int, str]] = []
    direction = +1 if event.polarity is Polarity.DENIED else -1

    if event.failed_replication and event.proposition is Proposition.POTENCY_REVERSAL:
        target = _scoped_claim(claims, provenance.mechanism)
        if target is not None:
            targets.append((target, +1, "failed replication supports the scoped prior"))
        return targets

    if event.proposition is Proposition.POTENCY_REVERSAL:
        if event.to_source:
            target = _scoped_claim(claims, provenance.mechanism)
            target = target or _first_kind(claims, "no_return", "potency_monotonic")
        else:
            target = _first_kind(claims, "potency_monotonic", "no_return")
        if target is not None:
            reason = (
                "observed potency reversal contradicts the claim"
                if direction < 0
                else "well-grounded denial of reversal supports the prior"
            )
            targets.append((target, direction, reason))
        if provenance.mechanism == "oocyte_nt" and event.to_source and direction < 0:
            nuclear = _first_kind(claims, "nuclear_potential")
            if nuclear is not None and (target is None or nuclear.id != target.id):
                targets.append((nuclear, +1, "nuclear transfer supports retained potential"))
        return targets

    if event.proposition is Proposition.DIFFERENTIATION:
        target = _first_kind(claims, "differentiation")
        diff_direction = +1 if event.polarity is Polarity.AFFIRMED else -1
        if target is not None:
            targets.append((target, diff_direction, "direct evidence about differentiation"))
        if event.lineage_restriction:
            restriction = _first_kind(claims, "lineage_restriction")
            if restriction is not None:
                targets.append((restriction, diff_direction, "direct evidence about lineage restriction"))
        return targets

    if event.proposition is Proposition.NUCLEAR_RETENTION:
        target = _first_kind(claims, "nuclear_potential")
        if target is not None:
            nuclear_direction = +1 if event.polarity is Polarity.AFFIRMED else -1
            targets.append((target, nuclear_direction, "direct evidence about nuclear potential"))
    return targets


def _semantic_key(event: SemanticEvent, mechanism: str, claim_id: str) -> str:
    source = event.source.id if event.source is not None else "unknown_source"
    destination = event.destination.id if event.destination is not None else "unknown_destination"
    raw = "|".join(
        (mechanism, claim_id, event.proposition.value, source, destination, event.property_axis or "")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _pending_prefix(event: SemanticEvent, mechanism: str, claim_id: str) -> str:
    return f"pending__v2__{_semantic_key(event, mechanism, claim_id)}__"


def _pending_id(event: SemanticEvent, mechanism: str, claim_id: str, evidence_id: str) -> str:
    origin = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:12]
    return f"{_pending_prefix(event, mechanism, claim_id)}{origin}"


def _matching_pending(
    view: GraphView,
    event: SemanticEvent,
    mechanism: str,
    targets: list[tuple[Claim, int, str]],
) -> list[str]:
    matches: list[str] = []
    for claim, _, _ in targets:
        prefix = _pending_prefix(event, mechanism, claim.id)
        matches.extend(pid for pid in view.pending_ids() if pid.startswith(prefix))
    return sorted(set(matches))


def _revised_confidence(claim: Claim, direction: int, quality: ProvenanceRecord) -> float:
    if direction < 0:
        shift = min(2.8, max(0.0, 5.0 * (quality.score - 0.45)))
        factor = 0.55 + 0.45 * quality.effect_value if quality.effect_reported else 0.85
        shift *= factor
    elif claim.confidence < 0.70 and quality.strong:
        shift = 0.65 + 0.75 * quality.score
    else:
        shift = 0.15 + 0.30 * quality.score
    return sigmoid(logit(claim.confidence) + direction * shift)


def _decision(
    deltas: list[Delta],
    rationale: str,
    confidence: float,
    ood: bool,
    evidence_class: EvidenceClass,
) -> IngestResult:
    """Build an auditable result without exposing a new mutation channel."""
    return IngestResult(deltas, f"{rationale} [{evidence_class.value}]", confidence, ood)


def _no_change(
    item: EvidenceItem,
    rationale: str,
    confidence: float = 0.8,
    evidence_class: EvidenceClass = EvidenceClass.SATURATED,
) -> IngestResult:
    return _decision([no_op(item.id)], rationale, confidence, False, evidence_class)


def _ood_result(
    item: EvidenceItem,
    event: SemanticEvent,
    quality: ProvenanceRecord,
) -> IngestResult | None:
    if event.proposition is Proposition.PROPERTY_CHANGE:
        if event.property_axis is not None:
            deltas = (
                [Delta("propose_axis", item.id, {"axis": event.property_axis})]
                if quality.credible
                else [no_op(item.id)]
            )
            return _decision(
                deltas,
                "eligible result concerns an unmodeled property axis",
                max(0.6, quality.score),
                True,
                EvidenceClass.OOD,
            )
        if event.ood_regime is not None:
            deltas = (
                [Delta("propose_regime", item.id, {"regime": event.ood_regime})]
                if quality.credible
                else [no_op(item.id)]
            )
            return _decision(
                deltas,
                "eligible result concerns an unmodeled identity-preserving regime",
                max(0.6, quality.score),
                True,
                EvidenceClass.OOD,
            )

    if event.proposition is Proposition.CELL_TRANSITION and not event.has_intermediate:
        deltas = (
            [Delta("propose_regime", item.id, {"regime": "lateral_somatic_conversion"})]
            if quality.credible
            else [no_op(item.id)]
        )
        return _decision(
            deltas,
            "eligible result is a direct cross-lineage endpoint conversion",
            max(0.6, quality.score),
            True,
            EvidenceClass.OOD,
        )
    return None


def ingest(item: EvidenceItem, view: GraphView) -> IngestResult:
    """Return a bounded, attributed decision for one evidence item."""
    if not isinstance(item.body, str) or not item.body.strip() or len(item.body) > 20_000:
        return _no_change(
            item,
            "malformed, empty, or oversized evidence body rejected",
            0.99,
            EvidenceClass.INVALID,
        )

    assessment = _assess_body(item.body, view)
    if assessment.direct_control_attempt:
        return _no_change(
            item,
            "composite control-plane instruction rejected",
            0.99,
            EvidenceClass.INJECTION,
        )
    if assessment.malformed:
        return _no_change(
            item,
            "unbalanced quotation or delimiter rejected",
            0.99,
            EvidenceClass.INVALID,
        )
    if assessment.instruction_event:
        return _no_change(
            item,
            "instruction-shaped item is not scientific evidence",
            0.99,
            EvidenceClass.INJECTION,
        )
    if assessment.ambiguous:
        return _no_change(
            item,
            "multiple or contradictory atomic events require abstention",
            0.95,
            EvidenceClass.NON_EVIDENCE,
        )
    if not assessment.events:
        return _no_change(
            item,
            "no eligible asserted experimental result found",
            0.9,
            EvidenceClass.NON_EVIDENCE,
        )

    quality = _provenance(item)
    event = assessment.events[0]
    claims = _claims(view)
    targets = _targets(event, quality, claims)
    pending = _matching_pending(view, event, quality.mechanism, targets)

    # Invalid metadata is never allowed to create, revise, or delete graph state.
    # It is logged as a no-op rather than as attacker-controlled pending state.
    if not quality.valid:
        return _no_change(
            item,
            "structured provenance failed closed: " + "; ".join(quality.issues),
            0.95,
            EvidenceClass.INVALID,
        )

    # Retractions and failed replications can resolve only exact semantic
    # fingerprints. A retraction cannot select an origin in the current input
    # contract, so multiple matching origins must abstain. Body-described failed
    # replication additionally needs credible structured evidence.
    if quality.retracted:
        if len(pending) == 1:
            return _decision(
                [Delta("drop_claim", item.id, {"claim_id": pending[0]})],
                "valid retraction resolved one exact matching pending event",
                0.95,
                False,
                EvidenceClass.INVALIDATION,
            )
        if len(pending) > 1:
            return _no_change(
                item,
                "retraction lacks an origin ID and matches multiple pending reports",
                0.99,
                EvidenceClass.INVALIDATION,
            )
        return _no_change(
            item,
            "retracted evidence has no exact pending dependency",
            0.98,
            EvidenceClass.INVALIDATION,
        )

    if event.failed_replication:
        if not quality.credible:
            return _no_change(
                item,
                "body-described replication failure lacks credible structured support",
                0.98,
                EvidenceClass.INVALIDATION,
            )
        if len(pending) == 1:
            return _decision(
                [Delta("drop_claim", item.id, {"claim_id": pending[0]})],
                "credible invalidating evidence resolved one exact matching pending event",
                max(0.7, quality.score),
                False,
                EvidenceClass.INVALIDATION,
            )
        if len(pending) > 1:
            return _no_change(
                item,
                "replication failure lacks an origin ID and matches multiple pending reports",
                0.99,
                EvidenceClass.INVALIDATION,
            )

    if quality.explicit_null_effect and event.polarity is Polarity.AFFIRMED and not event.failed_replication:
        return _no_change(
            item,
            "structured provenance reports an explicit null effect",
            0.95,
            EvidenceClass.NULL_REJECT,
        )

    ood = _ood_result(item, event, quality)
    if ood is not None:
        return ood

    contradictions = [target for target in targets if target[1] < 0]
    if contradictions and (quality.thin or not quality.credible):
        claim = contradictions[0][0]
        pid = _pending_id(event, quality.mechanism, claim.id, item.id)
        return _decision(
            [
                Delta(
                    "hold_pending",
                    item.id,
                    {
                        "claim_id": pid,
                        "note": (
                            f"Unconfirmed {event.proposition.value} via {quality.mechanism}; "
                            "awaiting exact independent confirmation."
                        ),
                    },
                )
            ],
            "extraordinary contradiction lacks sufficient independent provenance",
            max(0.55, 1.0 - quality.score),
            False,
            EvidenceClass.WEAK_EVIDENCE,
        )

    if not targets:
        return _no_change(
            item,
            "no represented claim matches the admitted event",
            max(0.5, quality.score),
            EvidenceClass.NON_EVIDENCE,
        )
    if not quality.credible:
        return _no_change(
            item,
            "no sufficiently grounded in-model revision",
            max(0.5, quality.score),
            EvidenceClass.WEAK_EVIDENCE,
        )

    deltas: list[Delta] = []
    reasons: list[str] = []
    for claim, direction, reason in targets:
        new_confidence = _revised_confidence(claim, direction, quality)
        if abs(new_confidence - claim.confidence) < 0.001:
            continue
        deltas.append(
            Delta(
                "revise_confidence",
                item.id,
                {"claim_id": claim.id, "new_confidence": round(new_confidence, 6)},
            )
        )
        reasons.append(reason)
        if direction < 0 and quality.strong and quality.mechanism not in {"unspecified", "observational"}:
            deltas.append(
                Delta(
                    "set_scope",
                    item.id,
                    {
                        "claim_id": claim.id,
                        "scope": {
                            "exception_under": quality.mechanism,
                            f"exception_under_{_slug(quality.mechanism)}": True,
                        },
                    },
                )
            )

    # Strong confirmation promotes and clears the exact pending semantic event.
    if pending and not event.failed_replication and not quality.retracted:
        deltas.extend(Delta("drop_claim", item.id, {"claim_id": pid}) for pid in pending)
        reasons.append("cleared exact pending event after independent confirmation")

    if not deltas:
        return _no_change(
            item,
            "evidence agrees with an already saturated belief",
            quality.score,
            EvidenceClass.SATURATED,
        )
    evidence_class = (
        EvidenceClass.CONTRADICTION
        if any(direction < 0 for _, direction, _ in targets)
        else EvidenceClass.CONFIRMATION
    )
    return _decision(
        deltas,
        "; ".join(reasons),
        min(0.98, max(0.65, quality.score)),
        False,
        evidence_class,
    )
