"""G04 acceptance tests: bounded update engine. Provider-free, deterministic."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.engine import (
    EPS,
    MAX_CASCADE,
    ROOT_BUDGET,
    TRUST_FABRICATION,
    TRUST_INIT,
    TRUST_MAX,
    Engine,
    bernoulli_kl,
    bound_update,
)
from core.types import Integrity

TOL = 1e-9


# ---------------------------------------------------------------------------
# bernoulli_kl
# ---------------------------------------------------------------------------


def test_kl_zero_when_equal() -> None:
    for p in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert bernoulli_kl(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_positive_and_asymmetric() -> None:
    assert bernoulli_kl(0.7, 0.4) > 0.0
    # (note: complementary pairs like (0.7, 0.3) ARE symmetric for Bernoulli)
    assert bernoulli_kl(0.7, 0.4) != pytest.approx(bernoulli_kl(0.4, 0.7))


def test_kl_safe_at_boundaries() -> None:
    # No math domain errors, no inf/nan at hard 0/1 inputs.
    for p, q in ((0.0, 1.0), (1.0, 0.0), (0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0)):
        v = bernoulli_kl(p, q)
        assert math.isfinite(v)
        assert v >= 0.0


def test_kl_known_value() -> None:
    # D_KL(Bern(0.5) || Bern(0.25)) = 0.5*ln(2) + 0.5*ln(2/3)
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert bernoulli_kl(0.5, 0.25) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------


def test_budget_constants() -> None:
    assert EPS[Integrity.L0_RAW] == 0.0
    assert EPS[Integrity.L1_PARSED] == 0.0
    assert EPS[Integrity.L2_VERIFIED] == 0.02
    assert EPS[Integrity.L3_REPLICATED] == 0.5
    assert ROOT_BUDGET == 0.6


# ---------------------------------------------------------------------------
# Acceptance: L0/L1 have exactly zero committed influence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("integrity", [Integrity.L0_RAW, Integrity.L1_PARSED])
@pytest.mark.parametrize("raw_bf", [1e6, 1e-6, 1.0, 1e12])
def test_l0_l1_zero_influence(integrity: Integrity, raw_bf: float) -> None:
    engine = Engine()
    prior = 0.5
    out = engine.propose("C17", "ROOT-001", "lab-alpha", prior, raw_bf, integrity)
    assert out["posterior"] == prior
    assert out["bounded_delta"] == 0.0
    assert out["root_spent"] == 0.0
    assert bound_update(prior, raw_bf, integrity, trust=1.0) == prior


def test_l0_flood_never_moves_belief() -> None:
    engine = Engine()
    prior = 0.5
    for i in range(50):
        out = engine.propose("C17", f"ROOT-{i}", f"src-{i}", prior, 1e6, Integrity.L0_RAW)
        assert out["posterior"] == prior
        assert out["bounded_delta"] == 0.0
    assert engine.spend_log == []  # zero-eps proposals commit nothing at all


# ---------------------------------------------------------------------------
# Acceptance: every L2/L3 update respects its KL budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("integrity", [Integrity.L2_VERIFIED, Integrity.L3_REPLICATED])
@pytest.mark.parametrize("prior", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
@pytest.mark.parametrize("raw_bf", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
def test_kl_budget_respected_sweep(integrity: Integrity, prior: float, raw_bf: float) -> None:
    engine = Engine()
    out = engine.propose("C1", "ROOT-X", "src", prior, raw_bf, integrity)
    kl = bernoulli_kl(out["posterior"], prior)
    assert kl <= EPS[integrity] + TOL
    # Delta direction matches the evidence direction.
    if raw_bf > 1.0:
        assert out["bounded_delta"] >= 0.0
    elif raw_bf < 1.0:
        assert out["bounded_delta"] <= 0.0
    # bound_update alone also respects the budget for any trust level.
    for trust in (0.0, 0.3, 0.9, 1.0):
        post = bound_update(prior, raw_bf, integrity, trust)
        assert bernoulli_kl(post, prior) <= EPS[integrity] + TOL


def test_bound_update_deterministic() -> None:
    a = bound_update(0.35, 50.0, Integrity.L3_REPLICATED, 0.7)
    b = bound_update(0.35, 50.0, Integrity.L3_REPLICATED, 0.7)
    assert a == b


def test_bound_update_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        bound_update(0.5, 0.0, Integrity.L2_VERIFIED, 0.5)
    with pytest.raises(ValueError):
        bound_update(0.5, -3.0, Integrity.L2_VERIFIED, 0.5)
    with pytest.raises(ValueError):
        bound_update(0.5, 2.0, Integrity.L2_VERIFIED, 1.5)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, False])
def test_nonfinite_and_bool_probabilities_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        bernoulli_kl(bad, 0.5)
    with pytest.raises(ValueError):
        bernoulli_kl(0.5, bad)
    with pytest.raises(ValueError):
        bound_update(bad, 2.0, Integrity.L2_VERIFIED, 0.5)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, False])
def test_nonfinite_and_bool_bayes_factors_rejected_before_integrity_short_circuit(
    bad: float,
) -> None:
    for integrity in Integrity:
        with pytest.raises(ValueError):
            bound_update(0.5, bad, integrity, 0.5)
        with pytest.raises(ValueError):
            Engine().propose("C", "ROOT", "src", 0.5, bad, integrity)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, False])
def test_nonfinite_and_bool_trust_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        bound_update(0.5, 2.0, Integrity.L2_VERIFIED, bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_probability_outside_unit_interval_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        bound_update(bad, 2.0, Integrity.L2_VERIFIED, 0.5)
    with pytest.raises(ValueError):
        Engine().propose("C", "ROOT", "src", bad, 2.0, Integrity.L2_VERIFIED)


# ---------------------------------------------------------------------------
# Acceptance: duplicate same-root evidence is conserved by ROOT_BUDGET
# ---------------------------------------------------------------------------


def test_same_root_flood_bounded_by_root_budget() -> None:
    """20 huge-BF articles from one root cannot spend past ROOT_BUDGET."""
    engine = Engine()
    prior = 0.5
    confidence = prior
    for i in range(20):
        out = engine.propose(
            "C17", "ROOT-001", f"outlet-{i}", confidence, 1e9, Integrity.L3_REPLICATED
        )
        assert out["root_spent"] <= ROOT_BUDGET + TOL
        confidence = out["posterior"]

    total_spent = engine.root_spent("C17", "ROOT-001")
    assert total_spent <= ROOT_BUDGET + TOL
    assert sum(r["root_cost"] for r in engine.proposals_for_root("ROOT-001")) == pytest.approx(
        total_spent, abs=1e-9
    )
    for record in engine.proposals_for_root("ROOT-001"):
        assert record["kl"] == pytest.approx(
            bernoulli_kl(record["posterior"], record["prior"]), abs=1e-12
        )
        assert record["root_cost"] == pytest.approx(abs(record["logit_delta"]), abs=1e-12)

    # Cumulative movement is bounded: once the budget is gone, later floods
    # are inert no matter how large the Bayes factor.
    frozen = confidence
    out = engine.propose("C17", "ROOT-001", "outlet-21", frozen, 1e12, Integrity.L3_REPLICATED)
    assert out["posterior"] == frozen
    assert out["bounded_delta"] == 0.0
    assert engine.root_spent("C17", "ROOT-001") <= ROOT_BUDGET + TOL


def test_single_root_not_starved() -> None:
    """A lone L3 root with the full budget still moves belief substantially."""
    engine = Engine()
    out = engine.propose("C17", "ROOT-FRESH", "lab", 0.5, 1e9, Integrity.L3_REPLICATED)
    # A single strong root gets the full global logit budget. The KL bound
    # remains an independent ceiling, but the tighter root cap wins here.
    assert out["bounded_delta"] > 0.1
    assert bernoulli_kl(out["posterior"], 0.5) <= EPS[Integrity.L3_REPLICATED]
    assert out["root_spent"] == pytest.approx(ROOT_BUDGET, abs=1e-9)
    assert out["root_spent"] <= ROOT_BUDGET + TOL


def test_flood_movement_stays_bounded_vs_unbudgeted() -> None:
    """The 20-article flood ends with total movement the budget allows, not
    the movement 20 unbounded huge-BF updates would produce (~1.0).

    Root spend is cumulative absolute applied logit movement, so chaining many
    small updates cannot salami-slice past the budget.
    """
    engine = Engine()
    confidence = 0.5
    for i in range(20):
        out = engine.propose(
            "C17", "ROOT-001", f"outlet-{i}", confidence, 1e9, Integrity.L3_REPLICATED
        )
        confidence = out["posterior"]
    # A 0.6 logit move from 0.5 ends near 0.646, well away from certainty.
    assert 0.5 < confidence < 0.7
    assert engine.root_spent("C17", "ROOT-001") <= ROOT_BUDGET + TOL


def test_same_root_budget_is_global_across_claims() -> None:
    engine = Engine()
    outputs = [
        engine.propose(claim, "ROOT-SHARED", f"src-{claim}", 0.5, 1e100, Integrity.L3_REPLICATED)
        for claim in ("C1", "C2", "C3")
    ]

    assert outputs[0]["bounded_delta"] > 0.0
    assert outputs[1]["bounded_delta"] == 0.0
    assert outputs[2]["bounded_delta"] == 0.0
    assert engine.spent_for_root("ROOT-SHARED") <= ROOT_BUDGET + TOL
    # Legacy two-argument reads now report the same global root spend.
    assert engine.root_spent("C1", "ROOT-SHARED") == engine.root_spent(
        "C3", "ROOT-SHARED"
    )


def test_contrary_evidence_never_refunds_ordinary_root_spend() -> None:
    engine = Engine()
    first = engine.propose("C", "ROOT", "src", 0.5, 3.0, Integrity.L3_REPLICATED)
    spent_before = first["root_spent"]
    second = engine.propose(
        "C", "ROOT", "src", first["posterior"], 1 / 3, Integrity.L3_REPLICATED
    )

    assert second["bounded_delta"] < 0.0
    assert second["root_spent"] >= spent_before
    assert all(record["root_cost"] >= 0.0 for record in engine.spend_log)


def test_sequential_cross_claim_updates_conserve_one_root_budget() -> None:
    engine = Engine()
    for index in range(100):
        engine.propose(
            f"C{index % 7}", "ROOT", f"src-{index}", 0.5, 1.5, Integrity.L2_VERIFIED
        )

    records = engine.proposals_for_root("ROOT")
    assert sum(record["root_cost"] for record in records) <= ROOT_BUDGET + TOL
    assert engine.spent_for_root("ROOT") == pytest.approx(
        sum(record["root_cost"] for record in records), abs=1e-12
    )


def test_concurrent_cross_claim_updates_conserve_one_root_budget() -> None:
    engine = Engine()

    def submit(index: int) -> None:
        engine.propose(
            f"C{index}", "ROOT", f"src-{index}", 0.5, 1e100, Integrity.L3_REPLICATED
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(submit, range(64)))

    records = engine.proposals_for_root("ROOT")
    assert engine.spent_for_root("ROOT") <= ROOT_BUDGET + TOL
    assert sum(record["root_cost"] for record in records) <= ROOT_BUDGET + TOL


def test_clone_isolated_for_shadow_execution() -> None:
    engine = Engine()
    first = engine.propose("C1", "ROOT", "src", 0.5, 2.0, Integrity.L2_VERIFIED)
    shadow = engine.clone()
    shadow.propose(
        "C2", "ROOT", "src", first["posterior"], 2.0, Integrity.L2_VERIFIED
    )

    assert shadow.spent_for_root("ROOT") > engine.spent_for_root("ROOT")
    assert len(shadow.spend_log) == len(engine.spend_log) + 1


# ---------------------------------------------------------------------------
# Acceptance: independent roots accumulate independently
# ---------------------------------------------------------------------------


def test_independent_roots_accumulate() -> None:
    prior = 0.5

    # One root alone.
    solo = Engine()
    one = solo.propose("C17", "ROOT-A", "lab-a", prior, 1e9, Integrity.L3_REPLICATED)

    # Two independent roots, chained.
    duo = Engine()
    first = duo.propose("C17", "ROOT-A", "lab-a", prior, 1e9, Integrity.L3_REPLICATED)
    assert first["posterior"] == one["posterior"]  # same first step, deterministic
    second = duo.propose(
        "C17", "ROOT-B", "lab-b", first["posterior"], 1e9, Integrity.L3_REPLICATED
    )

    # Each root has its own budget ledger...
    assert duo.root_spent("C17", "ROOT-A") <= ROOT_BUDGET + TOL
    assert duo.root_spent("C17", "ROOT-B") <= ROOT_BUDGET + TOL
    assert duo.root_spent("C17", "ROOT-B") > 0.0
    # ...and jointly they move confidence further than one root alone.
    assert second["posterior"] > one["posterior"]


# ---------------------------------------------------------------------------
# Acceptance: retraction deltas exactly negate prior committed deltas
# ---------------------------------------------------------------------------


def test_retraction_exactly_negates_deltas() -> None:
    engine = Engine()
    original_prior = 0.35  # deliberately not exactly representable / off-grid
    confidence = original_prior

    plan = [
        ("ROOT-001", "src-1", 40.0, Integrity.L3_REPLICATED),
        ("ROOT-001", "src-2", 3.0, Integrity.L2_VERIFIED),
        ("ROOT-002", "src-3", 0.05, Integrity.L3_REPLICATED),
        ("ROOT-002", "src-1", 7.0, Integrity.L2_VERIFIED),
    ]
    committed: list[tuple[str, float, float]] = []
    for root, src, bf, level in plan:
        out = engine.propose("C17", root, src, confidence, bf, level)
        # Engine invariant: posterior == prior + delta and posterior - delta
        # == prior, both float-exact.
        assert out["posterior"] == confidence + out["bounded_delta"]
        assert out["posterior"] - out["bounded_delta"] == confidence
        confidence = out["posterior"]
        committed.append((root, out["bounded_delta"], out["root_spent"]))

    assert confidence != original_prior  # something actually moved

    # Retract in reverse order by negating each recorded delta.
    for rec in reversed(engine.spend_log.copy()):
        confidence = confidence - rec["delta"]
        engine.revert(
            rec["claim_id"],
            rec["root_experiment_id"],
            rec["delta"],
            rec["root_cost"],
        )

    assert confidence == original_prior  # EXACT float equality


def test_revert_refunds_root_budget() -> None:
    engine = Engine()
    out = engine.propose("C17", "ROOT-001", "src", 0.5, 1e9, Integrity.L3_REPLICATED)
    rec = engine.proposals_for_root("ROOT-001")[0]
    assert engine.root_spent("C17", "ROOT-001") == rec["root_cost"] > 0.0

    remaining = engine.revert("C17", "ROOT-001", rec["delta"], rec["root_cost"])
    assert remaining == pytest.approx(0.0, abs=1e-12)
    assert engine.spent_for_root("ROOT-001") == pytest.approx(0.0, abs=1e-12)

    # Budget is genuinely reusable after the refund.
    prior_again = out["posterior"] - rec["delta"]
    again = engine.propose("C17", "ROOT-001", "src", prior_again, 1e9, Integrity.L3_REPLICATED)
    assert again["bounded_delta"] > 0.1


def test_revert_never_goes_negative() -> None:
    engine = Engine()
    assert engine.revert("C17", "ROOT-404", 0.1, 0.5) == 0.0
    assert engine.root_spent("C17", "ROOT-404") == 0.0


def test_revert_rejects_invalid_refund_values() -> None:
    engine = Engine()
    for bad in (math.nan, math.inf, -math.inf, True, False, -0.1):
        with pytest.raises(ValueError):
            engine.revert("C", "ROOT", 0.1, bad)


# ---------------------------------------------------------------------------
# Acceptance: source trust accounting
# ---------------------------------------------------------------------------


def test_new_source_starts_tight() -> None:
    engine = Engine()
    assert engine.trust("never-seen") == TRUST_INIT == 0.3


def test_escrow_survival_widens_trust_capped() -> None:
    engine = Engine()
    assert engine.record_escrow_survival("lab") == pytest.approx(0.4)
    assert engine.record_escrow_survival("lab") == pytest.approx(0.5)
    for _ in range(20):  # cannot widen past TRUST_MAX
        engine.record_escrow_survival("lab")
    assert engine.trust("lab") == pytest.approx(TRUST_MAX) == pytest.approx(0.9)


def test_fabrication_collapses_trust() -> None:
    engine = Engine()
    for _ in range(6):
        engine.record_escrow_survival("lab")
    assert engine.trust("lab") == pytest.approx(TRUST_MAX)
    engine.record_fabrication("lab")
    assert engine.trust("lab") == TRUST_FABRICATION == 0.05


def test_reputation_epoch_capped_per_call() -> None:
    engine = Engine()
    start = engine.trust("lab")  # 0.3

    applied = engine.apply_reputation_epoch({"lab": +0.5})  # asks for 5x the cap
    assert applied["lab"] == pytest.approx(MAX_CASCADE)
    assert engine.trust("lab") == pytest.approx(start + MAX_CASCADE)

    applied = engine.apply_reputation_epoch({"lab": -0.7})
    assert applied["lab"] == pytest.approx(-MAX_CASCADE)
    assert engine.trust("lab") == pytest.approx(start)

    # Within-cap requests apply as-is; results stay in [0, 1].
    applied = engine.apply_reputation_epoch({"lab": +0.05, "other": -0.9})
    assert applied["lab"] == pytest.approx(0.05)
    assert applied["other"] == pytest.approx(-MAX_CASCADE)
    assert 0.0 <= engine.trust("other") <= 1.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, False])
def test_reputation_epoch_rejects_nonfinite_and_bool_before_mutation(bad: float) -> None:
    engine = Engine()
    with pytest.raises(ValueError):
        engine.apply_reputation_epoch({"a-valid": 0.05, "z-invalid": bad})
    assert engine.trust("a-valid") == TRUST_INIT
    assert engine.trust("z-invalid") == TRUST_INIT


def test_trust_scales_influence() -> None:
    """Higher trust means the same Bayes factor moves belief further (within budget)."""
    low = bound_update(0.5, 3.0, Integrity.L3_REPLICATED, trust=0.1)
    high = bound_update(0.5, 3.0, Integrity.L3_REPLICATED, trust=0.9)
    assert high > low > 0.5
    zero = bound_update(0.5, 1e9, Integrity.L3_REPLICATED, trust=0.0)
    assert zero == 0.5  # zero trust neutralizes any evidence
