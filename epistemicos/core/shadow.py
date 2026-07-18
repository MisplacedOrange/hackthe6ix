"""Isolated shadow execution and deterministic finite shock scoring (G05)."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from core.contracts import EngineBreakdown
from core.engine import ROOT_BUDGET, Engine
from core.types import Integrity, Relation

SHOCK_MEDIUM = 1.0
SHOCK_HIGH = 2.0

_DISPLACEMENT_WEIGHT = 1.0
_INVARIANT_WEIGHT = 10.0
_AFFECTED_WEIGHT = 0.05
_OOD_WEIGHT = 2.0
_DUPLICATE_WEIGHT = 0.25
_PROBABILITY_FLOOR = 1e-9


class ShadowExecutionError(RuntimeError):
    """Raised when an isolated shadow update cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class ShockReport:
    """Auditable components of the deterministic shock score."""

    score: float
    max_logit_displacement: float
    invariant_violations: tuple[str, ...]
    affected_nodes: int
    ood_distance: float
    duplicate_signal: float

    @property
    def finite(self) -> bool:
        return all(
            math.isfinite(value)
            for value in (
                self.score,
                self.max_logit_displacement,
                self.ood_distance,
                self.duplicate_signal,
            )
        )


@dataclass(frozen=True, slots=True)
class ShadowExecution:
    """A proposed engine breakdown plus its deterministic shock report."""

    engine: EngineBreakdown
    shock: ShockReport
    state: Any | None = None


def clone_engine(engine: Any) -> Any:
    """Create an isolated engine snapshot without mutating the live instance.

    Current ``Engine`` exposes ``clone``. The explicit fallback keeps G05
    compatible with an older G04 implementation without using ``deepcopy``
    on thread locks.
    """
    clone_method = getattr(engine, "clone", None)
    if callable(clone_method):
        cloned = clone_method()
        if cloned is engine:
            raise ShadowExecutionError("engine clone is not isolated")
        return cloned

    if isinstance(engine, Engine):
        cloned = Engine()
        lock = getattr(engine, "_lock", None)
        if lock is None:
            cloned._spent = dict(engine._spent)
            cloned.spend_log = [record.copy() for record in engine.spend_log]
            cloned._trust = dict(engine._trust)
        else:
            with lock:
                cloned._spent = dict(engine._spent)
                cloned.spend_log = [record.copy() for record in engine.spend_log]
                cloned._trust = dict(engine._trust)
        return cloned
    raise ShadowExecutionError("engine does not provide an isolated clone")


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _logit(probability: float) -> float:
    bounded = min(1.0 - _PROBABILITY_FLOOR, max(_PROBABILITY_FLOOR, probability))
    return math.log(bounded / (1.0 - bounded))


def clone_state(state: Any) -> Any:
    """Return an isolated graph snapshot suitable for candidate execution."""
    model_copy = getattr(state, "model_copy", None)
    cloned = model_copy(deep=True) if callable(model_copy) else deepcopy(state)
    if cloned is state:
        raise ShadowExecutionError("graph clone is not isolated")

    live_claims = getattr(state, "claims", None)
    cloned_claims = getattr(cloned, "claims", None)
    if isinstance(live_claims, Mapping) and isinstance(cloned_claims, Mapping):
        for claim_id in live_claims.keys() & cloned_claims.keys():
            if live_claims[claim_id] is cloned_claims[claim_id]:
                raise ShadowExecutionError(f"claim clone is not isolated: {claim_id}")
    return cloned


def _claim_confidences(state: Any) -> dict[str, float]:
    claims = getattr(state, "claims", None)
    if not isinstance(claims, Mapping):
        raise ShadowExecutionError("state does not expose a claims mapping")

    confidences: dict[str, float] = {}
    for claim_id, claim in claims.items():
        confidence = getattr(claim, "confidence", None)
        if not isinstance(claim_id, str) or not _finite_number(confidence):
            raise ShadowExecutionError("state contains an invalid claim")
        confidences[claim_id] = float(confidence)
    return confidences


def _execute_on_shadow_graph(
    state: Any,
    *,
    claim_id: str,
    posterior: object,
) -> tuple[Any, int, tuple[str, ...]]:
    """Apply a candidate posterior to a copied graph and check graph scope."""
    live_before = _claim_confidences(state)
    shadow_state = clone_state(state)
    shadow_claims = getattr(shadow_state, "claims")
    violations: list[str] = []

    if claim_id not in shadow_claims:
        return shadow_state, 0, ("target_missing",)
    if not _finite_number(posterior):
        return shadow_state, 0, ("nonfinite",)

    shadow_claims[claim_id].confidence = float(posterior)
    live_after = _claim_confidences(state)
    shadow_after = _claim_confidences(shadow_state)

    if live_after != live_before:
        violations.append("live_graph_mutated")
    if shadow_after.keys() != live_before.keys():
        violations.append("claim_set_changed")

    affected_ids = {
        node_id
        for node_id in live_before.keys() & shadow_after.keys()
        if shadow_after[node_id] != live_before[node_id]
    }
    unexpected = affected_ids - {claim_id}
    if unexpected:
        violations.extend(f"outside_neighborhood:{node_id}" for node_id in sorted(unexpected))
    if len(affected_ids) > 1:
        violations.append("affected_node_limit")
    return shadow_state, len(affected_ids), tuple(violations)


def _validate_breakdown(
    breakdown: Mapping[str, object],
    *,
    requested_prior: float,
    requested_integrity: Integrity,
    relation: Relation,
) -> tuple[str, ...]:
    violations: list[str] = []
    required = {
        "prior",
        "raw_bf",
        "bounded_delta",
        "root_spent",
        "posterior",
        "integrity",
    }
    missing = required - set(breakdown)
    if missing:
        return tuple(f"missing:{field}" for field in sorted(missing))

    numeric_fields = ("prior", "raw_bf", "bounded_delta", "root_spent", "posterior")
    if any(not _finite_number(breakdown[field]) for field in numeric_fields):
        return ("nonfinite",)

    prior = float(breakdown["prior"])
    raw_bf = float(breakdown["raw_bf"])
    delta = float(breakdown["bounded_delta"])
    spent = float(breakdown["root_spent"])
    posterior = float(breakdown["posterior"])

    if not math.isclose(prior, requested_prior, rel_tol=0.0, abs_tol=1e-12):
        violations.append("prior_changed")
    if raw_bf <= 0.0:
        violations.append("raw_bf_nonpositive")
    if not 0.0 <= posterior <= 1.0:
        violations.append("posterior_out_of_bounds")
    if not math.isclose(prior + delta, posterior, rel_tol=0.0, abs_tol=1e-12):
        violations.append("delta_mismatch")
    if not 0.0 <= spent <= ROOT_BUDGET + 1e-12:
        violations.append("root_budget")
    if breakdown["integrity"] != requested_integrity:
        violations.append("integrity_changed")
    if requested_integrity <= Integrity.L1_PARSED and delta != 0.0:
        violations.append("low_integrity_influence")
    if relation in (Relation.SUPPORTS, Relation.REPLICATES) and delta < 0.0:
        violations.append("relation_direction")
    if relation is Relation.CONTRADICTS and delta > 0.0:
        violations.append("relation_direction")
    return tuple(violations)


def _normalize_breakdown(proposed: object) -> EngineBreakdown:
    """Normalize the G04 mapping seam and attribute-based test adapters."""
    fields = (
        "prior",
        "raw_bf",
        "bounded_delta",
        "root_spent",
        "posterior",
        "integrity",
    )
    if isinstance(proposed, Mapping):
        values = {field: proposed[field] for field in fields if field in proposed}
    else:
        values = {
            field: getattr(proposed, field)
            for field in fields
            if hasattr(proposed, field)
        }
    # Keep missing fields visible to invariant validation instead of allowing
    # a constructor KeyError to collapse them into a generic execution error.
    return values  # type: ignore[return-value]


def execute_shadow(
    engine: Any,
    *,
    claim_id: str,
    root_experiment_id: str,
    source_id: str,
    prior: float,
    raw_bf: float,
    integrity: Integrity,
    relation: Relation,
    ood_distance: float,
    duplicate_signal: float,
    state: Any | None = None,
) -> ShadowExecution:
    """Propose against copied engine/graph state and score the candidate effect."""
    shadow = clone_engine(engine)
    proposed = shadow.propose(
        claim_id,
        root_experiment_id,
        source_id,
        prior,
        raw_bf,
        integrity,
    )
    breakdown = _normalize_breakdown(proposed)
    violations = _validate_breakdown(
        breakdown,
        requested_prior=prior,
        requested_integrity=integrity,
        relation=relation,
    )

    shadow_state = None
    if state is not None:
        shadow_state, affected, graph_violations = _execute_on_shadow_graph(
            state,
            claim_id=claim_id,
            posterior=breakdown.get("posterior"),
        )
        violations = (*violations, *graph_violations)
    else:
        affected = (
            int(float(breakdown["bounded_delta"]) != 0.0)
            if "bounded_delta" in breakdown and _finite_number(breakdown["bounded_delta"])
            else 0
        )

    if "nonfinite" in violations or any(item.startswith("missing:") for item in violations):
        displacement = 0.0
    else:
        displacement = abs(_logit(float(breakdown["posterior"])) - _logit(prior))
    score = (
        _DISPLACEMENT_WEIGHT * displacement
        + _INVARIANT_WEIGHT * len(violations)
        + _AFFECTED_WEIGHT * affected
        + _OOD_WEIGHT * ood_distance
        + _DUPLICATE_WEIGHT * max(0.0, duplicate_signal)
    )
    report = ShockReport(
        score=score,
        max_logit_displacement=displacement,
        invariant_violations=violations,
        affected_nodes=affected,
        ood_distance=ood_distance,
        duplicate_signal=duplicate_signal,
    )
    return ShadowExecution(engine=breakdown, shock=report, state=shadow_state)


__all__ = [
    "SHOCK_HIGH",
    "SHOCK_MEDIUM",
    "ShadowExecution",
    "ShadowExecutionError",
    "ShockReport",
    "clone_engine",
    "clone_state",
    "execute_shadow",
]
