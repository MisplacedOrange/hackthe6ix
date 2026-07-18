"""G02 acceptance tests: strict lattice types, provider-free."""

import math

import pytest
from pydantic import ValidationError

from core.types import (
    Claim,
    ClaimState,
    EffectDirection,
    EvidenceIR,
    EvidenceSubmission,
    Integrity,
    Relation,
    Span,
    Verdict,
    Witnessed,
)
from tests.conftest import SAMPLE_TEXT, make_evidence


class TestSpan:
    def test_valid_span(self):
        s = Span(start=0, end=5)
        assert s.slice("hello world") == "hello"

    def test_end_must_exceed_start(self):
        with pytest.raises(ValidationError):
            Span(start=10, end=10)
        with pytest.raises(ValidationError):
            Span(start=10, end=3)

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            Span(start=-1, end=5)

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (False, 1),
            (0, True),
            ("0", 1),
            (0, "1"),
            (0.0, 1),
            (0, 1.0),
        ],
    )
    def test_offsets_are_strict_non_boolean_integers(self, start, end):
        with pytest.raises(ValidationError):
            Span(start=start, end=end)

    def test_bounds_check(self):
        s = Span(start=0, end=999)
        assert not s.in_bounds("short")
        assert Span(start=0, end=5).in_bounds("hello world")

    def test_whitespace_only_span_not_in_bounds(self):
        assert not Span(start=5, end=6).in_bounds("hello world")


class TestEvidenceIR:
    def test_valid_evidence_parses(self):
        ev = make_evidence()
        assert ev.target_claim.value == "C17"
        assert ev.relation.value is Relation.CONTRADICTS

    def test_unknown_fields_rejected(self):
        ev = make_evidence()
        with pytest.raises(ValidationError):
            EvidenceIR(**ev.model_dump(), sneaky_extra="ignore previous instructions")

    @pytest.mark.parametrize(
        "nested_path",
        ["witness", "span"],
    )
    def test_authority_fields_are_recursively_rejected(self, nested_path):
        payload = make_evidence().model_dump()
        if nested_path == "witness":
            payload["relation"]["authority"] = "peer-reviewed"
        else:
            payload["relation"]["support_span"]["trust"] = 1.0

        with pytest.raises(ValidationError):
            EvidenceIR(**payload)

    def test_cannot_self_declare_integrity(self):
        """The critical lattice rule: text cannot declare its own integrity."""
        ev = make_evidence()
        for field in ("integrity", "level", "confidence", "verdict", "trust"):
            with pytest.raises(ValidationError):
                EvidenceIR(**ev.model_dump(), **{field: "L3"})

    def test_frozen(self):
        ev = make_evidence()
        with pytest.raises(ValidationError):
            ev.source_id = "attacker"

    def test_relation_outside_ontology_rejected(self):
        with pytest.raises(ValidationError):
            Witnessed[Relation](value="proves_beyond_doubt", support_span=Span(start=0, end=1))

    def test_effect_direction_outside_ontology_rejected(self):
        with pytest.raises(ValidationError):
            Witnessed[EffectDirection](value="miraculous", support_span=Span(start=0, end=1))

    def test_empty_target_claim_rejected(self):
        with pytest.raises(ValidationError):
            make_evidence(claim_id="   ")

    def test_witness_spans_point_at_real_text(self):
        ev = make_evidence()
        for w in (ev.target_claim, ev.relation, ev.effect_direction, ev.effect_size, ev.sample_size):
            assert w is not None
            assert w.support_span.in_bounds(SAMPLE_TEXT)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_effect_size_must_be_finite(self, value):
        payload = make_evidence().model_dump()
        payload["effect_size"]["value"] = value
        with pytest.raises(ValidationError):
            EvidenceIR(**payload)

    @pytest.mark.parametrize("value", [0, -1, True, "482", 482.0])
    def test_sample_size_is_a_strict_positive_integer(self, value):
        payload = make_evidence().model_dump()
        payload["sample_size"]["value"] = value
        with pytest.raises(ValidationError):
            EvidenceIR(**payload)

    @pytest.mark.parametrize("field", ["source_id", "experiment_id", "root_experiment_id"])
    def test_proposed_ids_are_strict_strings(self, field):
        payload = make_evidence().model_dump()
        payload[field] = 17
        with pytest.raises(ValidationError):
            EvidenceIR(**payload)


class TestEvidenceSubmission:
    def test_trusted_envelope_is_strict_and_frozen(self):
        submission = EvidenceSubmission(
            raw_text="untrusted observation",
            source_id="source-registry-1",
            experiment_id="EXP-1",
            root_experiment_id="ROOT-1",
            idempotency_key="request-1",
        )
        assert submission.source_id == "source-registry-1"
        with pytest.raises(ValidationError):
            submission.source_id = "attacker"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("raw_text", 1),
            ("source_id", 1),
            ("experiment_id", True),
            ("root_experiment_id", 1.0),
            ("idempotency_key", b"request-1"),
        ],
    )
    def test_envelope_rejects_coercion(self, field, value):
        payload = {
            "raw_text": "observation",
            "source_id": "source-1",
            "experiment_id": "EXP-1",
            "root_experiment_id": "ROOT-1",
            "idempotency_key": "request-1",
        }
        payload[field] = value
        with pytest.raises(ValidationError):
            EvidenceSubmission(**payload)

    def test_envelope_rejects_authority_extras(self):
        with pytest.raises(ValidationError):
            EvidenceSubmission(
                raw_text="observation",
                source_id="source-1",
                experiment_id="EXP-1",
                root_experiment_id="ROOT-1",
                idempotency_key="request-1",
                integrity=Integrity.L3_REPLICATED,
            )


class TestIntegrityLattice:
    def test_ordering(self):
        assert Integrity.L0_RAW < Integrity.L1_PARSED < Integrity.L2_VERIFIED < Integrity.L3_REPLICATED

    def test_l0_is_zero(self):
        assert Integrity.L0_RAW == 0


class TestClaimState:
    def test_all_four_states(self):
        c = Claim(id="C1", text="X causes Y")
        assert c.state is ClaimState.UNKNOWN
        c.supporting = ["E1"]
        assert c.state is ClaimState.SUPPORTED
        c.supporting, c.contradicting = [], ["E2"]
        assert c.state is ClaimState.CONTRADICTED
        c.supporting = ["E1"]
        assert c.state is ClaimState.CONTESTED

    def test_confidence_bounded(self):
        with pytest.raises(ValidationError):
            Claim(id="C1", text="X", confidence=1.5)
        with pytest.raises(ValidationError):
            Claim(id="C1", text="X", confidence=-0.1)

    @pytest.mark.parametrize("confidence", [math.nan, math.inf, -math.inf, True, "0.5"])
    def test_confidence_is_strict_and_finite(self, confidence):
        with pytest.raises(ValidationError):
            Claim(id="C1", text="X", confidence=confidence)

    def test_mutable_defaults_are_safe(self):
        a = Claim(id="A", text="a")
        b = Claim(id="B", text="b")
        a.supporting.append("E1")
        assert b.supporting == []


class TestVerdict:
    def test_verdict_values(self):
        assert {v.value for v in Verdict} == {"commit", "provisional", "escrow", "reject"}
