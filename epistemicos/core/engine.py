"""Bounded update engine (G04).

Deterministic confidence arithmetic. An LLM may *propose* evidence; only this
module may turn a proposal into a belief delta, and every delta is clipped by
two independent budgets:

1. **Integrity budget** -- per-transaction trust region:
   ``D_KL(Bernoulli(posterior) || Bernoulli(prior)) <= EPS[integrity]``.
   L0 and L1 commit nothing (L1 is provisional/display only).
2. **Root budget** -- all evidence descending from one root experiment shares
   one global, monotonic budget across every claim. The charged currency is
   absolute applied logit displacement, so derivative articles cannot
   salami-slice their way to certainty and contrary ordinary evidence cannot
   refund a lineage's prior influence.

Source trust scales the effective Bayes factor (``raw_bf ** trust``); new
sources start tight, survived escrows widen trust, confirmed fabrication
collapses it, and reputation reweighting is capped per epoch so retroactive
cascades are bounded.

Everything here is pure Python + math and fully deterministic.
"""

from __future__ import annotations

import math
import threading
from typing import Mapping, TypedDict

from core.contracts import EngineBreakdown
from core.types import Integrity

__all__ = [
    "EPS",
    "ROOT_BUDGET",
    "TRUST_INIT",
    "TRUST_MAX",
    "TRUST_FABRICATION",
    "MAX_CASCADE",
    "bernoulli_kl",
    "bound_update",
    "SpendRecord",
    "Engine",
]

# ---------------------------------------------------------------------------
# Budgets (module constants -- the coordination seam for G05's monitor checks)
# ---------------------------------------------------------------------------

#: Per-transaction KL budget by earned integrity level.
#: L1 is *provisional only*: the hypothesis may be displayed, but it commits
#: zero belief, hence a zero KL budget just like L0.
EPS: dict[Integrity, float] = {
    Integrity.L0_RAW: 0.0,
    Integrity.L1_PARSED: 0.0,
    Integrity.L2_VERIFIED: 0.02,
    Integrity.L3_REPLICATED: 0.5,
}

#: Total absolute applied-logit displacement available to one root lineage.
ROOT_BUDGET: float = 0.6

# ---------------------------------------------------------------------------
# Trust constants
# ---------------------------------------------------------------------------

TRUST_INIT: float = 0.3  #: unseen sources start tight
TRUST_MAX: float = 0.9  #: ceiling reached through survived escrows
TRUST_STEP: float = 0.1  #: widening per survived escrow
TRUST_FABRICATION: float = 0.05  #: collapse on confirmed fabrication
MAX_CASCADE: float = 0.1  #: max |trust change| per source per reputation epoch

_P_MIN = 1e-9
_P_MAX = 1.0 - 1e-9


def _finite_real(name: str, value: float) -> float:
    """Return a finite real value, rejecting bools and coercion-prone inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(name: str, value: float) -> float:
    result = _finite_real(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _clamp01(p: float) -> float:
    """Clamp a probability away from the 0/1 boundaries for numeric safety."""
    if p < _P_MIN:
        return _P_MIN
    if p > _P_MAX:
        return _P_MAX
    return p


def bernoulli_kl(p: float, q: float) -> float:
    """D_KL(Bernoulli(p) || Bernoulli(q)) in nats, safe at the 0/1 boundaries."""
    p = _clamp01(_probability("p", p))
    q = _clamp01(_probability("q", q))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def _logit(p: float) -> float:
    bounded = _clamp01(p)
    return math.log(bounded / (1.0 - bounded))


def _logistic(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clip(prior: float, target: float, ok) -> float:
    """Largest step from ``prior`` toward ``target`` with ``ok(step)`` true.

    Deterministic bisection on the interpolation fraction. Requires
    ``ok(prior)`` to hold (callers guard this). Because every budget
    predicate used here is a sublevel set of a function convex in the
    candidate (Bernoulli KL is convex in its first argument), the valid set
    is an interval containing ``prior``, so the predicate flips exactly once
    along the segment and bisection is sound.
    """
    if target == prior or ok(target):
        return target
    lo, hi = 0.0, 1.0  # fraction of the way from prior to target
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(prior + mid * (target - prior)):
            lo = mid
        else:
            hi = mid
    return prior + lo * (target - prior)


def _finalize(prior: float, target: float, ok) -> tuple[float, float]:
    """Turn a clipped target into an exactly-reversible commit.

    Returns ``(posterior, delta)`` satisfying:

    * ``posterior == prior + delta``   (float-exact)
    * ``posterior - delta == prior``   (float-exact -- retraction by negating
      the recorded delta restores the prior bit-for-bit)
    * ``ok(posterior)`` (all KL budgets hold)

    If rounding breaks reversibility, the delta is nudged one ulp toward zero
    (shrinking the update, never enlarging it -- budgets stay satisfied by
    convexity) until the round trip is exact; converges in a few steps and
    terminates at ``delta == 0`` in the worst case.
    """
    delta = target - prior
    for _ in range(128):
        posterior = prior + delta
        if posterior - delta == prior and ok(posterior):
            return posterior, delta
        if delta == 0.0:
            return prior, 0.0
        delta = math.nextafter(delta, 0.0)
    return prior, 0.0


def bound_update(prior: float, raw_bf: float, integrity: Integrity, trust: float) -> float:
    """Posterior for a raw Bayes factor, trust-scaled and KL-clipped.

    ``raw_bf`` is a likelihood ratio (> 0): > 1 supports the claim, < 1
    contradicts it. Trust in [0, 1] scales it as ``raw_bf ** trust`` (trust 0
    neutralizes any evidence; trust 1 passes it through). The odds-form
    posterior is then clipped toward the prior so that
    ``bernoulli_kl(posterior, prior) <= EPS[integrity]``. Deterministic.
    """
    prior = _probability("prior", prior)
    raw_bf = _finite_real("raw_bf", raw_bf)
    trust = _finite_real("trust", trust)
    if raw_bf <= 0.0:
        raise ValueError(f"raw_bf must be > 0 (likelihood ratio), got {raw_bf}")
    if not 0.0 <= trust <= 1.0:
        raise ValueError(f"trust must be in [0, 1], got {trust}")
    eps = EPS[Integrity(integrity)]
    if eps <= 0.0:
        return prior  # L0/L1: zero committed influence, exactly.
    p = _clamp01(prior)
    target = _logistic(_logit(p) + trust * math.log(raw_bf))
    return _clip(prior, target, lambda cand: bernoulli_kl(cand, prior) <= eps)


class SpendRecord(TypedDict):
    """One entry in the engine's ordered spend log (audit trail for G05/G08).

    ``kl`` is the event's Bernoulli KL displacement. ``root_cost`` is the
    non-negative root-budget charge: ``abs(logit_delta)``.
    """

    claim_id: str
    root_experiment_id: str
    source_id: str
    prior: float
    posterior: float
    delta: float
    kl: float
    logit_delta: float
    root_cost: float
    integrity: Integrity


class Engine:
    """Thread-safe bounded update engine with one budget per root lineage."""

    def __init__(self) -> None:
        self._spent: dict[str, float] = {}
        self.spend_log: list[SpendRecord] = []
        self._trust: dict[str, float] = {}
        self._lock = threading.RLock()

    def root_spent(
        self, claim_or_root_id: str, root_experiment_id: str | None = None
    ) -> float:
        """Return global root spend; the two-argument form is legacy-compatible."""
        root_id = root_experiment_id or claim_or_root_id
        with self._lock:
            return self._spent.get(root_id, 0.0)

    def spent_for_root(self, root_experiment_id: str) -> float:
        with self._lock:
            return self._spent.get(root_experiment_id, 0.0)

    def proposals_for_root(self, root_experiment_id: str) -> list[SpendRecord]:
        with self._lock:
            return [
                rec.copy()
                for rec in self.spend_log
                if rec["root_experiment_id"] == root_experiment_id
            ]

    def clone(self) -> "Engine":
        """Return an isolated snapshot for shadow execution."""
        with self._lock:
            cloned = Engine()
            cloned._spent = dict(self._spent)
            cloned.spend_log = [record.copy() for record in self.spend_log]
            cloned._trust = dict(self._trust)
            return cloned

    def propose(
        self,
        claim_id: str,
        root_experiment_id: str,
        source_id: str,
        prior: float,
        raw_bf: float,
        integrity: Integrity,
    ) -> EngineBreakdown:
        """Atomically compute and charge an integrity/root-bounded update."""
        prior = _probability("prior", prior)
        raw_bf = _finite_real("raw_bf", raw_bf)
        if raw_bf <= 0.0:
            raise ValueError("raw_bf must be > 0")
        if isinstance(integrity, bool):
            raise ValueError("integrity must be an Integrity value")
        integrity = Integrity(integrity)
        with self._lock:
            return self._propose_locked(
                claim_id,
                root_experiment_id,
                source_id,
                prior,
                raw_bf,
                integrity,
            )

    def _propose_locked(
        self,
        claim_id: str,
        root_experiment_id: str,
        source_id: str,
        prior: float,
        raw_bf: float,
        integrity: Integrity,
    ) -> EngineBreakdown:
        already = self._spent.get(root_experiment_id, 0.0)
        eps = EPS[integrity]
        remaining = max(0.0, ROOT_BUDGET - already)
        if eps <= 0.0 or remaining <= 1e-12:
            return self._breakdown(prior, raw_bf, 0.0, already, prior, integrity)

        trust = self._trust.get(source_id, TRUST_INIT)
        target = bound_update(prior, raw_bf, integrity, trust)
        prior_logit = _logit(prior)
        desired_logit_delta = _logit(target) - prior_logit
        applied_logit_delta = math.copysign(
            min(abs(desired_logit_delta), remaining), desired_logit_delta
        )
        root_limited_target = _logistic(prior_logit + applied_logit_delta)

        def ok(candidate: float) -> bool:
            event_kl = bernoulli_kl(candidate, prior)
            root_cost = abs(_logit(candidate) - prior_logit)
            return event_kl <= eps + 1e-15 and root_cost <= remaining + 1e-15

        posterior, delta = _finalize(prior, root_limited_target, ok)
        if delta == 0.0:
            return self._breakdown(prior, raw_bf, 0.0, already, prior, integrity)

        logit_delta = _logit(posterior) - prior_logit
        root_cost = abs(logit_delta)
        event_kl = bernoulli_kl(posterior, prior)
        new_spent = min(ROOT_BUDGET, already + root_cost)
        if ROOT_BUDGET - new_spent <= 1e-12:
            new_spent = ROOT_BUDGET
        charged_root_cost = new_spent - already
        self._spent[root_experiment_id] = new_spent
        self.spend_log.append(
            SpendRecord(
                claim_id=claim_id,
                root_experiment_id=root_experiment_id,
                source_id=source_id,
                prior=prior,
                posterior=posterior,
                delta=delta,
                kl=event_kl,
                logit_delta=logit_delta,
                root_cost=charged_root_cost,
                integrity=integrity,
            )
        )
        return self._breakdown(prior, raw_bf, delta, new_spent, posterior, integrity)

    @staticmethod
    def _breakdown(
        prior: float,
        raw_bf: float,
        delta: float,
        root_spent: float,
        posterior: float,
        integrity: Integrity,
    ) -> EngineBreakdown:
        return EngineBreakdown(
            prior=prior,
            raw_bf=raw_bf,
            bounded_delta=delta,
            root_spent=root_spent,
            posterior=posterior,
            integrity=integrity,
        )

    def revert(
        self,
        claim_id: str,
        root_experiment_id: str,
        delta: float,
        root_cost: float,
    ) -> float:
        """Refund an explicitly retracted commit's recorded root cost."""
        del claim_id  # Budget scope is global per root, not per claim.
        _finite_real("delta", delta)
        root_cost = _finite_real("root_cost", root_cost)
        if root_cost < 0.0:
            raise ValueError("root_cost must be non-negative")
        with self._lock:
            new_spent = max(
                0.0, self._spent.get(root_experiment_id, 0.0) - root_cost
            )
            if new_spent <= 1e-12:
                new_spent = 0.0
            self._spent[root_experiment_id] = new_spent
            return new_spent

    def trust(self, source_id: str) -> float:
        with self._lock:
            return self._trust.get(source_id, TRUST_INIT)

    def record_escrow_survival(self, source_id: str) -> float:
        with self._lock:
            new = min(TRUST_MAX, self._trust.get(source_id, TRUST_INIT) + TRUST_STEP)
            self._trust[source_id] = new
            return new

    def record_fabrication(self, source_id: str) -> float:
        with self._lock:
            self._trust[source_id] = TRUST_FABRICATION
            return TRUST_FABRICATION

    def apply_reputation_epoch(self, adjustments: Mapping[str, float]) -> dict[str, float]:
        """Apply a finite, capped reputation epoch atomically."""
        validated = {
            source_id: _finite_real(f"adjustments[{source_id!r}]", requested)
            for source_id, requested in adjustments.items()
        }
        with self._lock:
            applied: dict[str, float] = {}
            for source_id in sorted(validated):
                requested = validated[source_id]
                capped = max(-MAX_CASCADE, min(MAX_CASCADE, requested))
                old = self._trust.get(source_id, TRUST_INIT)
                new = max(0.0, min(1.0, old + capped))
                self._trust[source_id] = new
                applied[source_id] = new - old
            return applied
