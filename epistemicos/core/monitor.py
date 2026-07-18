"""Deterministic action-level reference monitor (G05).

The monitor evaluates proposed effects, never whether prose "looks like" an
injection. Schema-valid text earns only PARSED integrity. Witness checks and
an external provenance registry are required for VERIFIED/REPLICATED, and all
candidate influence is computed against an isolated engine clone.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import Any

from core.contracts import EngineBreakdown, MonitorResult
from core.engine import ROOT_BUDGET
from core.shadow import (
    SHOCK_HIGH,
    SHOCK_MEDIUM,
    ShadowExecutionError,
    ShockReport,
    execute_shadow,
)
from core.types import (
    UNMODELED,
    EffectDirection,
    EvidenceIR,
    Integrity,
    Relation,
    Verdict,
    Witnessed,
)

_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)
_DUPLICATE_SIGNAL_REASON = 0.95
_RELATION_CUES: dict[Relation, tuple[str, ...]] = {
    Relation.SUPPORTS: ("support", "corroborat", "confirm"),
    Relation.CONTRADICTS: ("contradict", "conflict", "inconsistent", "refut"),
    Relation.REPLICATES: ("replicat", "reproduc"),
}
_DIRECTION_CUES: dict[EffectDirection, tuple[str, ...]] = {
    EffectDirection.POSITIVE: ("positive", "increase", "higher", "improv"),
    EffectDirection.NEGATIVE: ("negative", "decrease", "lower", "reduc", "declin"),
    EffectDirection.NULL: ("null", "no effect", "no significant", "unchanged", "zero"),
}


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Externally registered source/experiment/root and claim scope."""

    experiment_id: str
    source_id: str
    root_experiment_id: str
    claim_ids: frozenset[str]
    replicates_experiment_id: str | None = None
    independent: bool = False
    outcome_relation: Relation | None = None
    effect_direction: EffectDirection | None = None


@dataclass(frozen=True, slots=True)
class TrustedProvenance:
    """Request-bound provenance resolved by a trusted ingestion boundary.

    This is the direct-call counterpart to :class:`ProvenanceRegistry`. It is
    never constructed from compiler output. ``verified=False`` deliberately
    caps the observation at L1 even when every identifier happens to match.
    """

    source_id: str
    experiment_id: str
    root_experiment_id: str
    verified: bool
    independent_replication: bool = False
    replicated_experiment_id: str | None = None
    replicated_root_experiment_id: str | None = None
    outcome_relation: Relation | None = None
    effect_direction: EffectDirection | None = None
    replicated_outcome_relation: Relation | None = None
    replicated_effect_direction: EffectDirection | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "experiment_id", "root_experiment_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.verified) is not bool:
            raise TypeError("verified must be a bool")
        if type(self.independent_replication) is not bool:
            raise TypeError("independent_replication must be a bool")
        for name in ("replicated_experiment_id", "replicated_root_experiment_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be None or a non-empty string")
        if self.outcome_relation is not None and not isinstance(
            self.outcome_relation, Relation
        ):
            raise TypeError("outcome_relation must be a Relation or None")
        if self.effect_direction is not None and not isinstance(
            self.effect_direction, EffectDirection
        ):
            raise TypeError("effect_direction must be an EffectDirection or None")
        if self.replicated_outcome_relation is not None and not isinstance(
            self.replicated_outcome_relation, Relation
        ):
            raise TypeError("replicated_outcome_relation must be a Relation or None")
        if self.replicated_effect_direction is not None and not isinstance(
            self.replicated_effect_direction, EffectDirection
        ):
            raise TypeError("replicated_effect_direction must be an EffectDirection or None")


class ProvenanceRegistry:
    """Small deterministic provenance API for the API and seeded demo.

    Calling ``register_experiment`` is an external witness: it is trusted
    application configuration, never inferred from ``EvidenceIR`` text.
    Re-registering the exact same record is idempotent; conflicting mappings
    are rejected.
    """

    def __init__(self) -> None:
        self._records: dict[str, ProvenanceRecord] = {}
        self._sources: set[str] = set()
        self._lock = threading.RLock()

    def register_source(self, source_id: str) -> None:
        self._require_id("source_id", source_id)
        with self._lock:
            self._sources.add(source_id)

    def register_experiment(
        self,
        experiment_id: str,
        *,
        source_id: str,
        root_experiment_id: str,
        claim_ids: Set[str],
        replicates_experiment_id: str | None = None,
        independent: bool = False,
        outcome_relation: Relation | None = None,
        effect_direction: EffectDirection | None = None,
    ) -> ProvenanceRecord:
        self._require_id("experiment_id", experiment_id)
        self._require_id("source_id", source_id)
        self._require_id("root_experiment_id", root_experiment_id)
        if isinstance(independent, bool) is False:
            raise TypeError("independent must be a bool")
        claims = frozenset(claim_ids)
        if not claims or any(not isinstance(item, str) or not item for item in claims):
            raise ValueError("claim_ids must contain non-empty strings")
        if replicates_experiment_id is not None:
            self._require_id("replicates_experiment_id", replicates_experiment_id)
        elif independent:
            raise ValueError("independent replication requires a target experiment")
        if outcome_relation is not None and not isinstance(outcome_relation, Relation):
            raise TypeError("outcome_relation must be a Relation or None")
        if effect_direction is not None and not isinstance(effect_direction, EffectDirection):
            raise TypeError("effect_direction must be an EffectDirection or None")

        with self._lock:
            if replicates_experiment_id is not None and replicates_experiment_id not in self._records:
                raise ValueError("replication target must be registered first")
            if independent:
                target = self._records[replicates_experiment_id]
                if target.source_id == source_id or target.root_experiment_id == root_experiment_id:
                    raise ValueError("independent replication requires a distinct source and root")
            record = ProvenanceRecord(
                experiment_id=experiment_id,
                source_id=source_id,
                root_experiment_id=root_experiment_id,
                claim_ids=claims,
                replicates_experiment_id=replicates_experiment_id,
                independent=independent,
                outcome_relation=outcome_relation,
                effect_direction=effect_direction,
            )
            existing = self._records.get(experiment_id)
            if existing is not None and existing != record:
                raise ValueError(f"experiment {experiment_id!r} is already registered")
            self._records[experiment_id] = record
            self._sources.add(source_id)
            return record

    def resolve(self, experiment_id: str) -> ProvenanceRecord | None:
        with self._lock:
            return self._records.get(experiment_id)

    def source_registered(self, source_id: str) -> bool:
        with self._lock:
            return source_id in self._sources

    @staticmethod
    def _require_id(name: str, value: object) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class MonitorEvaluation:
    """Immutable container matching the G07 monitor seam."""

    monitor: MonitorResult
    engine: EngineBreakdown | None
    shock_report: ShockReport | None = None

    @property
    def shock(self) -> ShockReport | None:
        """Alias used by the G08 handoff while preserving the original name."""
        return self.shock_report


def _evaluation(
    verdict: Verdict,
    reasons: list[str],
    *,
    integrity: Integrity = Integrity.L1_PARSED,
    shock: float = 0.0,
    engine: EngineBreakdown | None = None,
    shock_report: ShockReport | None = None,
) -> MonitorEvaluation:
    return MonitorEvaluation(
        monitor=MonitorResult(
            verdict=verdict,
            reasons=list(reasons),
            integrity=integrity,
            shock=shock,
        ),
        engine=engine,
        shock_report=shock_report,
    )


def _numeric_value_present(witness: Witnessed[Any], text: str, *, integer: bool) -> bool:
    values: list[float] = []
    for token in _NUMBER_RE.findall(witness.support_span.slice(text)):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:  # defensive: regex output should always parse
            continue
    expected = float(witness.value)
    if integer:
        return any(value.is_integer() and int(value) == int(expected) for value in values)
    return any(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12) for value in values)


def _ontology_conflicts(
    witness: Witnessed[Any],
    text: str,
    cues: Mapping[Any, tuple[str, ...]],
) -> bool:
    """Detect an explicit cue for a different ontology value.

    A cue-free scientific phrase remains inspectable but is not rejected: a
    small lexical table is not a semantic oracle. An explicit ``supports``
    span paired with ``contradicts`` is, however, deterministically invalid.
    """
    supported_text = witness.support_span.slice(text).casefold()
    observed = {
        value
        for value, value_cues in cues.items()
        if any(cue in supported_text for cue in value_cues)
    }
    return bool(observed) and witness.value not in observed


def _witness_reasons(ir: EvidenceIR, text: str) -> list[str]:
    witnessed: list[tuple[str, Witnessed[Any]]] = [
        ("target_claim", ir.target_claim),
        ("relation", ir.relation),
        ("effect_direction", ir.effect_direction),
    ]
    if ir.effect_size is not None:
        witnessed.append(("effect_size", ir.effect_size))
    if ir.sample_size is not None:
        witnessed.append(("sample_size", ir.sample_size))

    reasons: list[str] = []
    used: dict[tuple[int, int], str] = {}
    for field_name, witness in witnessed:
        span = witness.support_span
        if not span.in_bounds(text):
            reasons.append(f"witness:{field_name}:out_of_bounds")
            continue
        key = (span.start, span.end)
        prior_field = used.get(key)
        if prior_field is not None:
            reasons.append(f"witness:span_reused:{prior_field}:{field_name}")
        else:
            used[key] = field_name

    if ir.effect_size is not None and ir.effect_size.support_span.in_bounds(text):
        if not _numeric_value_present(ir.effect_size, text, integer=False):
            reasons.append("witness:effect_size:numeric_value_missing")
    if ir.sample_size is not None and ir.sample_size.support_span.in_bounds(text):
        if not _numeric_value_present(ir.sample_size, text, integer=True):
            reasons.append("witness:sample_size:numeric_value_missing")
    if ir.relation.support_span.in_bounds(text) and _ontology_conflicts(
        ir.relation, text, _RELATION_CUES
    ):
        reasons.append("witness:relation:ontology_mismatch")
    if ir.effect_direction.support_span.in_bounds(text) and _ontology_conflicts(
        ir.effect_direction, text, _DIRECTION_CUES
    ):
        reasons.append("witness:effect_direction:ontology_mismatch")
    return reasons


def _signal_valid(value: object, *, lower: float, upper: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and lower <= float(value) <= upper
    )


def _derive_raw_bf(ir: EvidenceIR) -> float:
    """Derive a finite Bayes factor from witnessed fields only.

    This heuristic is deliberately small and deterministic; it is not a
    scientific meta-analysis. Its authority comes solely from the engine's
    integrity/root budgets, never from model-provided confidence or verdicts.
    """
    effect = abs(ir.effect_size.value) if ir.effect_size is not None else 0.1
    sample = ir.sample_size.value if ir.sample_size is not None else 1
    strength = max(0.05, min(6.0, effect * math.sqrt(sample) / 4.0))
    if ir.effect_direction.value.value == "null":
        strength *= 0.25
    if ir.relation.value is Relation.CONTRADICTS:
        return math.exp(-strength)
    return math.exp(strength)


def _requested_raw_bf(ir: EvidenceIR, raw_bf: float | None) -> float:
    if raw_bf is None:
        return _derive_raw_bf(ir)
    if not _signal_valid(raw_bf, lower=0.0, upper=float("inf")) or float(raw_bf) <= 0.0:
        raise ValueError("raw_bf must be a finite likelihood ratio greater than zero")

    value = float(raw_bf)
    relation = ir.relation.value
    if relation in (Relation.SUPPORTS, Relation.REPLICATES) and value < 1.0:
        raise ValueError("raw_bf direction contradicts the evidence relation")
    if relation is Relation.CONTRADICTS and value > 1.0:
        raise ValueError("raw_bf direction contradicts the evidence relation")
    return value


def _root_spent(engine: Any, root_experiment_id: str) -> float | None:
    getter = getattr(engine, "spent_for_root", None)
    if not callable(getter):
        getter = getattr(engine, "root_spent", None)
    if not callable(getter):
        return None
    spent = getter(root_experiment_id)
    return float(spent) if _signal_valid(spent, lower=0.0, upper=float("inf")) else None


class ReferenceMonitor:
    """Validate authority, shadow a bounded update, and route its shock."""

    def __init__(self, engine: Any, registry: ProvenanceRegistry | None = None) -> None:
        self.engine = engine
        self.registry = registry or ProvenanceRegistry()

    def evaluate(
        self,
        ir: EvidenceIR,
        normalized_text: str,
        state: Any,
        *,
        ood_distance: float = 0.0,
        duplicate_signal: float = 0.0,
        prior_roots: frozenset[str] = frozenset(),
        prior_experiments: frozenset[str] | None = None,
        raw_bf: float | None = None,
        provenance: TrustedProvenance | None = None,
        earned_integrity: Integrity | None = None,
    ) -> MonitorEvaluation:
        reasons: list[str] = []

        if not isinstance(normalized_text, str):
            return _evaluation(Verdict.REJECT, ["witness:normalized_text_invalid"])
        if not _signal_valid(ood_distance, lower=0.0, upper=2.0) or not _signal_valid(
            duplicate_signal, lower=-1.0, upper=1.0
        ):
            return _evaluation(Verdict.REJECT, ["shock:nonfinite_signal"])
        if (
            not isinstance(prior_roots, Set)
            or isinstance(prior_roots, (str, bytes))
            or any(not isinstance(root, str) for root in prior_roots)
        ):
            return _evaluation(Verdict.REJECT, ["provenance:prior_roots_invalid"])
        if prior_experiments is not None and (
            not isinstance(prior_experiments, Set)
            or isinstance(prior_experiments, (str, bytes))
            or any(not isinstance(experiment, str) for experiment in prior_experiments)
        ):
            return _evaluation(Verdict.REJECT, ["provenance:prior_experiments_invalid"])

        witness_reasons = _witness_reasons(ir, normalized_text)
        if witness_reasons:
            return _evaluation(Verdict.REJECT, witness_reasons)
        if not isinstance(ir.relation.value, Relation):
            return _evaluation(Verdict.REJECT, ["ontology:relation_unknown"])
        if not isinstance(ir.effect_direction.value, EffectDirection):
            return _evaluation(Verdict.REJECT, ["ontology:effect_direction_unknown"])

        if ir.target_claim.value == UNMODELED:
            return _evaluation(
                Verdict.PROVISIONAL,
                ["claim:unmodeled", "provenance_unverified", "integrity:parsed_only"],
            )

        claims = getattr(state, "claims", None)
        if not isinstance(claims, Mapping):
            return _evaluation(Verdict.REJECT, ["state:claims_invalid"])
        claim = claims.get(ir.target_claim.value)
        if claim is None:
            return _evaluation(Verdict.REJECT, ["claim:unknown"])
        claim_pattern = re.compile(
            rf"(?<!\w){re.escape(ir.target_claim.value)}(?!\w)",
            flags=re.IGNORECASE,
        )
        if claim_pattern.search(ir.target_claim.support_span.slice(normalized_text)) is None:
            return _evaluation(Verdict.REJECT, ["witness:target_claim:value_mismatch"])
        prior = getattr(claim, "confidence", None)
        if not _signal_valid(prior, lower=0.0, upper=1.0):
            return _evaluation(Verdict.REJECT, ["state:claim_confidence_invalid"])

        integrity_ceiling = Integrity.L1_PARSED
        record = self.registry.resolve(ir.experiment_id) if provenance is None else None
        if provenance is not None:
            if not isinstance(provenance, TrustedProvenance):
                return _evaluation(Verdict.REJECT, ["provenance:trusted_context_invalid"])
            mismatches: list[str] = []
            if provenance.source_id != ir.source_id:
                mismatches.append("provenance:source_mismatch")
            if provenance.experiment_id != ir.experiment_id:
                mismatches.append("provenance:experiment_mismatch")
            if provenance.root_experiment_id != ir.root_experiment_id:
                mismatches.append("provenance:root_mismatch")
            if mismatches:
                return _evaluation(Verdict.REJECT, mismatches)
            if (
                provenance.outcome_relation is not None
                and provenance.outcome_relation is not ir.relation.value
            ):
                return _evaluation(Verdict.REJECT, ["provenance:relation_mismatch"])
            if (
                provenance.effect_direction is not None
                and provenance.effect_direction is not ir.effect_direction.value
            ):
                return _evaluation(
                    Verdict.REJECT, ["provenance:effect_direction_mismatch"]
                )
            if provenance.verified:
                integrity_ceiling = Integrity.L2_VERIFIED
            else:
                reasons.extend(["provenance_unverified", "provenance:verification_missing"])

            if provenance.independent_replication:
                independent = (
                    provenance.verified
                    and ir.relation.value is Relation.REPLICATES
                    and ir.claimed_replication_of is not None
                    and provenance.replicated_experiment_id == ir.claimed_replication_of
                    and provenance.replicated_root_experiment_id is not None
                    and provenance.replicated_root_experiment_id
                    != provenance.root_experiment_id
                    and provenance.replicated_root_experiment_id in prior_roots
                    and (
                        prior_experiments is None
                        or provenance.replicated_experiment_id in prior_experiments
                    )
                    and provenance.outcome_relation is Relation.REPLICATES
                    and provenance.effect_direction is ir.effect_direction.value
                    and provenance.replicated_outcome_relation is not None
                    and provenance.replicated_effect_direction
                    is ir.effect_direction.value
                )
                if independent:
                    integrity_ceiling = Integrity.L3_REPLICATED
                    reasons.append("provenance:independent_replication")
                else:
                    if (
                        provenance.replicated_experiment_id is not None
                        and prior_experiments is not None
                        and provenance.replicated_experiment_id
                        not in prior_experiments
                    ):
                        reasons.append("provenance:replication_target_not_prior")
                    if (
                        provenance.outcome_relation is None
                        or provenance.effect_direction is None
                        or provenance.replicated_outcome_relation is None
                        or provenance.replicated_effect_direction is None
                    ):
                        reasons.append("provenance:replication_outcome_unverified")
                    elif (
                        provenance.outcome_relation is not Relation.REPLICATES
                        or provenance.effect_direction is not ir.effect_direction.value
                        or provenance.replicated_effect_direction
                        is not ir.effect_direction.value
                    ):
                        reasons.append("provenance:replication_outcome_mismatch")
                    reasons.append("provenance:replication_unverified")
        elif record is None:
            reasons.extend(["provenance_unverified", "provenance:experiment_unregistered"])
            if ir.relation.value is Relation.REPLICATES or ir.claimed_replication_of is not None:
                reasons.append("provenance:replication_unverified")
        else:
            mismatches: list[str] = []
            if record.source_id != ir.source_id:
                mismatches.append("provenance:source_mismatch")
            if record.root_experiment_id != ir.root_experiment_id:
                mismatches.append("provenance:root_mismatch")
            if mismatches:
                return _evaluation(Verdict.REJECT, mismatches)
            if ir.target_claim.value not in record.claim_ids:
                return _evaluation(Verdict.REJECT, ["provenance:claim_scope_violation"])
            if (
                record.outcome_relation is not None
                and record.outcome_relation is not ir.relation.value
            ):
                return _evaluation(Verdict.REJECT, ["provenance:relation_mismatch"])
            if (
                record.effect_direction is not None
                and record.effect_direction is not ir.effect_direction.value
            ):
                return _evaluation(
                    Verdict.REJECT, ["provenance:effect_direction_mismatch"]
                )
            integrity_ceiling = Integrity.L2_VERIFIED

            if ir.relation.value is Relation.REPLICATES:
                target_id = ir.claimed_replication_of
                target = self.registry.resolve(target_id) if target_id is not None else None
                independent = (
                    target_id is not None
                    and target is not None
                    and record.replicates_experiment_id == target_id
                    and record.independent
                    and record.source_id != target.source_id
                    and record.root_experiment_id != target.root_experiment_id
                    and target.root_experiment_id in prior_roots
                    and (
                        prior_experiments is None or target_id in prior_experiments
                    )
                    and ir.target_claim.value in target.claim_ids
                    and record.outcome_relation is Relation.REPLICATES
                    and record.effect_direction is ir.effect_direction.value
                    and target.outcome_relation is not None
                    and target.effect_direction is ir.effect_direction.value
                )
                if independent:
                    integrity_ceiling = Integrity.L3_REPLICATED
                    reasons.append("provenance:independent_replication")
                elif target_id is None:
                    reasons.append("provenance:replication_target_missing")
                elif target is None:
                    reasons.append("provenance:replication_target_unregistered")
                elif target.root_experiment_id not in prior_roots:
                    reasons.append("provenance:replication_target_not_prior")
                else:
                    if (
                        prior_experiments is not None
                        and target_id not in prior_experiments
                    ):
                        reasons.append("provenance:replication_target_not_prior")
                    if (
                        record.outcome_relation is None
                        or record.effect_direction is None
                        or target.outcome_relation is None
                        or target.effect_direction is None
                    ):
                        reasons.append("provenance:replication_outcome_unverified")
                    elif (
                        record.outcome_relation is not Relation.REPLICATES
                        or record.effect_direction is not ir.effect_direction.value
                        or target.effect_direction is not ir.effect_direction.value
                    ):
                        reasons.append("provenance:replication_outcome_mismatch")
                    reasons.append("provenance:replication_unverified")

        if earned_integrity is None:
            integrity = integrity_ceiling
        else:
            try:
                if isinstance(earned_integrity, bool):
                    raise ValueError
                integrity = Integrity(earned_integrity)
            except (TypeError, ValueError):
                return _evaluation(Verdict.REJECT, ["integrity:override_invalid"])
            if integrity > integrity_ceiling:
                return _evaluation(
                    Verdict.REJECT,
                    [*reasons, "integrity:exceeds_provenance_witness"],
                    integrity=integrity_ceiling,
                )
            if integrity is Integrity.L0_RAW:
                return _evaluation(
                    Verdict.REJECT,
                    [*reasons, "integrity:external_downgrade"],
                    integrity=integrity,
                )

        trusted_root = (
            provenance.root_experiment_id
            if provenance is not None
            else record.root_experiment_id
            if record is not None
            else ir.root_experiment_id
        )
        trusted_source = (
            provenance.source_id
            if provenance is not None
            else record.source_id
            if record is not None
            else ir.source_id
        )
        spent = _root_spent(self.engine, trusted_root)
        if integrity >= Integrity.L2_VERIFIED and spent is not None and spent >= ROOT_BUDGET - 1e-12:
            return _evaluation(
                Verdict.REJECT,
                [*reasons, f"budget:root_exhausted:{trusted_root}"],
                integrity=integrity,
            )

        try:
            requested_raw_bf = _requested_raw_bf(ir, raw_bf)
        except ValueError:
            return _evaluation(
                Verdict.REJECT,
                [*reasons, "engine:raw_bf_invalid"],
                integrity=integrity,
            )

        # An extreme reported effect is a deterministic statistical-OOD signal
        # only after provenance reaches L2. L1 semantic lies remain inert and
        # provisional; a verified-but-implausible magnitude is shadowed with
        # high shock and routed to escrow.
        effect_ood = 0.0
        if (
            integrity >= Integrity.L2_VERIFIED
            and ir.effect_size is not None
            and abs(ir.effect_size.value) >= 5.0
        ):
            effect_ood = 1.0
            reasons.append("signal:effect_size_ood")
        effective_ood = max(float(ood_distance), effect_ood)
        try:
            shadow = execute_shadow(
                self.engine,
                claim_id=ir.target_claim.value,
                root_experiment_id=trusted_root,
                source_id=trusted_source,
                prior=float(prior),
                raw_bf=requested_raw_bf,
                integrity=integrity,
                relation=ir.relation.value,
                ood_distance=effective_ood,
                duplicate_signal=float(duplicate_signal),
                state=state,
            )
        except (ShadowExecutionError, ValueError, TypeError, KeyError):
            return _evaluation(
                Verdict.REJECT,
                [*reasons, "shadow:execution_failed"],
                integrity=integrity,
            )

        report = shadow.shock
        if report.invariant_violations:
            invariant_reasons = [
                "shadow:nonfinite" if item == "nonfinite" else f"shadow:invariant:{item}"
                for item in report.invariant_violations
            ]
            return _evaluation(
                Verdict.REJECT,
                [*reasons, *invariant_reasons],
                integrity=integrity,
                shock=report.score if report.finite else 0.0,
                engine=shadow.engine,
                shock_report=report,
            )
        if not report.finite:
            return _evaluation(
                Verdict.REJECT,
                [*reasons, "shadow:nonfinite"],
                integrity=integrity,
                engine=shadow.engine,
                shock_report=report,
            )

        if duplicate_signal >= _DUPLICATE_SIGNAL_REASON:
            reasons.append("signal:duplicate")
        if report.score >= SHOCK_HIGH:
            verdict = Verdict.ESCROW
            reasons.append("shock:high")
        elif report.score >= SHOCK_MEDIUM:
            verdict = Verdict.PROVISIONAL
            reasons.append("shock:medium")
        elif integrity is Integrity.L1_PARSED:
            verdict = Verdict.PROVISIONAL
            reasons.append("integrity:parsed_only")
        else:
            verdict = Verdict.COMMIT
            reasons.append("shock:low")

        return _evaluation(
            verdict,
            reasons,
            integrity=integrity,
            shock=report.score,
            engine=shadow.engine,
            shock_report=report,
        )


__all__ = [
    "MonitorEvaluation",
    "ProvenanceRecord",
    "ProvenanceRegistry",
    "ReferenceMonitor",
    "TrustedProvenance",
]
