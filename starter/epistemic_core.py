"""Pure, deterministic epistemic primitives for the scored policy.

This module deliberately has no process-global evidence memory.  Pending state is
encoded into strict, versioned identifiers stored by the official graph API, so
replay derives the same aggregate from :class:`~groundtruth.model.GraphView`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Iterable, Mapping, Sequence


MAX_COUNT = 8
MAX_PERCENT = 100
MAX_LOG_ODDS = 3.0
POLICY_OPERATIONS = frozenset(
    {
        "no_op",
        "revise_confidence",
        "set_scope",
        "drop_claim",
        "hold_pending",
        "propose_regime",
        "propose_axis",
    }
)
_SEMANTIC_KEY_RE = r"[0-9a-f]{16}"
_ORIGIN_HASH_RE = r"[0-9a-f]{12}"
_V3_RE = re.compile(
    rf"pending__v3__(?P<semantic>{_SEMANTIC_KEY_RE})"
    rf"__(?P<polarity>n)"
    rf"__g(?P<groups>[0-8])"
    rf"__r(?P<replications>[0-8])"
    rf"__d(?P<directness>0|[1-9][0-9]?|100)"
    rf"__e(?P<effect>0|[1-9][0-9]?|100)"
    rf"__m(?P<method>0|[1-9][0-9]?|100)"
    rf"__(?P<origin>{_ORIGIN_HASH_RE})"
)
_V2_RE = re.compile(
    rf"pending__v2__(?P<semantic>{_SEMANTIC_KEY_RE})__(?P<origin>{_ORIGIN_HASH_RE})"
)
_SAFE_RECEIPT_TOKEN = re.compile(r"[A-Za-z0-9_.,+\-]+")


EVIDENCE_CLASSES = frozenset(
    {
        "INJECTION",
        "INVALID",
        "NON_EVIDENCE",
        "NULL_REJECT",
        "INVALIDATION",
        "WEAK_EVIDENCE",
        "CONTRADICTION",
        "CONFIRMATION",
        "OOD",
        "SATURATED",
    }
)


def _bounded_integer(value: int, upper: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
        raise ValueError(f"{field} must be an integer in [0,{upper}]")


@dataclass(frozen=True)
class ClaimContract:
    """Closed authorization contract for one represented claim kind."""

    claim_kind: str
    mechanisms: frozenset[str]
    propositions: frozenset[str]
    min_groups: int
    min_accumulated_groups: int
    allow_scope_exception: bool
    scope_keys: frozenset[str]
    source_roles: frozenset[str] = frozenset()
    destination_roles: frozenset[str] = frozenset()
    allow_confirmation: bool = True
    allow_contradiction: bool = True
    min_quality: float = 0.60
    strong_groups: int = 3
    strong_quality: float = 0.82

    def __post_init__(self) -> None:
        if not self.claim_kind or re.fullmatch(r"[a-z][a-z0-9_]*", self.claim_kind) is None:
            raise ValueError("claim_kind must be a closed identifier")
        if not self.mechanisms or not self.propositions:
            raise ValueError("claim contract mechanism/proposition sets must not be empty")
        for collection in (
            self.mechanisms,
            self.propositions,
            self.scope_keys,
            self.source_roles,
            self.destination_roles,
        ):
            if not isinstance(collection, frozenset) or any(
                not isinstance(value, str)
                or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None
                for value in collection
            ):
                raise ValueError("claim contract sets must contain closed identifiers")
        _bounded_integer(self.min_groups, MAX_COUNT, "min_groups")
        _bounded_integer(
            self.min_accumulated_groups, MAX_COUNT, "min_accumulated_groups"
        )
        if self.min_groups < 1 or self.min_accumulated_groups < self.min_groups:
            raise ValueError("accumulated group threshold must be at least min_groups")
        _bounded_integer(self.strong_groups, MAX_COUNT, "strong_groups")
        if self.strong_groups < self.min_groups:
            raise ValueError("strong_groups must be at least min_groups")
        for name in ("min_quality", "strong_quality"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.strong_quality < self.min_quality:
            raise ValueError("strong_quality must be at least min_quality")
        if not all(
            isinstance(value, bool)
            for value in (
                self.allow_scope_exception,
                self.allow_confirmation,
                self.allow_contradiction,
            )
        ):
            raise ValueError("claim contract behavior flags must be boolean")


@dataclass(frozen=True)
class LabPolicy:
    """Machine-readable laboratory policy used by semantic preflight."""

    contracts: tuple[ClaimContract, ...]
    allowed_scope_values: frozenset[str]
    max_log_odds: float = MAX_LOG_ODDS
    allowed_axes: frozenset[str] = frozenset()
    allowed_regimes: frozenset[str] = frozenset()
    operations: frozenset[str] = POLICY_OPERATIONS
    evidence_classes: frozenset[str] = EVIDENCE_CLASSES

    def __post_init__(self) -> None:
        if not self.contracts or any(
            not isinstance(contract, ClaimContract) for contract in self.contracts
        ):
            raise ValueError("lab policy must contain claim contracts")
        kinds = tuple(contract.claim_kind for contract in self.contracts)
        if len(kinds) != len(set(kinds)):
            raise ValueError("lab policy claim kinds must be unique")
        for collection, allow_upper in (
            (self.allowed_scope_values, False),
            (self.allowed_axes, False),
            (self.allowed_regimes, False),
            (self.operations, False),
            (self.evidence_classes, True),
        ):
            if not isinstance(collection, frozenset):
                raise ValueError("lab policy registries must be frozensets")
            pattern = r"[A-Z][A-Z_]*" if allow_upper else r"[a-z][a-z0-9_]*"
            if any(
                not isinstance(value, str) or re.fullmatch(pattern, value) is None
                for value in collection
            ):
                raise ValueError("lab policy registry contains an invalid identifier")
        if not self.operations or not self.evidence_classes:
            raise ValueError("operation and evidence-class registries must not be empty")
        if (
            isinstance(self.max_log_odds, bool)
            or not isinstance(self.max_log_odds, (int, float))
            or not math.isfinite(float(self.max_log_odds))
            or not 0.0 < float(self.max_log_odds) <= MAX_LOG_ODDS
        ):
            raise ValueError("lab policy log-odds cap must be in (0,3]")

    @property
    def allowed_scope_keys(self) -> frozenset[str]:
        keys: set[str] = set()
        for contract in self.contracts:
            keys.update(contract.scope_keys)
        return frozenset(keys)

    def contract_for(self, claim_kind: str) -> ClaimContract | None:
        return next(
            (contract for contract in self.contracts if contract.claim_kind == claim_kind),
            None,
        )


_MECHANISMS = frozenset(
    {
        "defined_factor",
        "env_stress",
        "oocyte_nt",
        "spontaneous",
        "lineage_tracing",
        "randomized",
        "observational",
        "unspecified",
    }
)
_SCOPE_KEYS = frozenset(
    {"exception_under"} | {f"exception_under_{mechanism}" for mechanism in _MECHANISMS}
)
_BIOLOGICAL_ROLES = frozenset({"pluripotent", "progenitor", "terminal"})
DEFAULT_LAB_POLICY = LabPolicy(
    contracts=(
        ClaimContract(
            claim_kind="no_return",
            mechanisms=_MECHANISMS,
            propositions=frozenset({"potency_reversal"}),
            min_groups=2,
            min_accumulated_groups=4,
            allow_scope_exception=True,
            scope_keys=_SCOPE_KEYS,
            source_roles=_BIOLOGICAL_ROLES,
            destination_roles=_BIOLOGICAL_ROLES,
        ),
        ClaimContract(
            claim_kind="scoped_no_return",
            mechanisms=_MECHANISMS,
            propositions=frozenset({"potency_reversal"}),
            min_groups=2,
            min_accumulated_groups=4,
            allow_scope_exception=True,
            scope_keys=_SCOPE_KEYS,
            source_roles=_BIOLOGICAL_ROLES,
            destination_roles=_BIOLOGICAL_ROLES,
        ),
        ClaimContract(
            claim_kind="potency_monotonic",
            mechanisms=_MECHANISMS,
            propositions=frozenset({"potency_reversal"}),
            min_groups=2,
            min_accumulated_groups=4,
            allow_scope_exception=True,
            scope_keys=_SCOPE_KEYS,
            source_roles=_BIOLOGICAL_ROLES,
            destination_roles=_BIOLOGICAL_ROLES,
        ),
        ClaimContract(
            claim_kind="no_lateral",
            mechanisms=_MECHANISMS,
            propositions=frozenset({"cell_transition"}),
            min_groups=2,
            min_accumulated_groups=4,
            allow_scope_exception=False,
            scope_keys=frozenset(),
            source_roles=_BIOLOGICAL_ROLES,
            destination_roles=_BIOLOGICAL_ROLES,
        ),
        ClaimContract(
            claim_kind="differentiation",
            mechanisms=_MECHANISMS,
            propositions=frozenset({"differentiation"}),
            min_groups=2,
            min_accumulated_groups=4,
            allow_scope_exception=True,
            scope_keys=_SCOPE_KEYS,
            source_roles=_BIOLOGICAL_ROLES,
            destination_roles=_BIOLOGICAL_ROLES,
        ),
        ClaimContract(
            claim_kind="lineage_restriction",
            mechanisms=_MECHANISMS,
            propositions=frozenset({"differentiation"}),
            min_groups=2,
            min_accumulated_groups=4,
            allow_scope_exception=True,
            scope_keys=_SCOPE_KEYS,
            source_roles=_BIOLOGICAL_ROLES,
            destination_roles=_BIOLOGICAL_ROLES,
        ),
        ClaimContract(
            claim_kind="nuclear_potential",
            mechanisms=_MECHANISMS,
            propositions=frozenset({"nuclear_retention", "potency_reversal"}),
            min_groups=2,
            min_accumulated_groups=4,
            allow_scope_exception=False,
            scope_keys=frozenset(),
            # Nuclear-retention reports need not assert a cell-state transition,
            # so endpoint roles are optional for this claim kind.
            source_roles=frozenset(),
            destination_roles=frozenset(),
        ),
    ),
    allowed_scope_values=_MECHANISMS,
    allowed_axes=frozenset(
        {
            "biological_age",
            "cell_function_independent_of_identity",
            "cell_size",
            "chromatin",
            "epigenetic_state",
            "gene_expression",
            "metabolic_state",
            "morphology",
            "transcriptional_state",
        }
    ),
    allowed_regimes=frozenset(
        {"identity_preserving_state_change", "lateral_somatic_conversion"}
    ),
)


@dataclass(frozen=True)
class EvidenceWeight:
    """Bounded, quantized evidence factors accepted from structured provenance."""

    groups: int
    replications: int
    directness: int
    effect: int
    method_reliability: int

    def __post_init__(self) -> None:
        _bounded_integer(self.groups, MAX_COUNT, "groups")
        _bounded_integer(self.replications, MAX_COUNT, "replications")
        _bounded_integer(self.directness, MAX_PERCENT, "directness")
        _bounded_integer(self.effect, MAX_PERCENT, "effect")
        _bounded_integer(self.method_reliability, MAX_PERCENT, "method_reliability")


@dataclass(frozen=True)
class PendingEvidence:
    """One origin-distinct provisional report encoded in graph-visible state."""

    semantic_key: str
    polarity: str
    weight: EvidenceWeight
    origin_hash: str
    version: int = 3

    def __post_init__(self) -> None:
        if re.fullmatch(_SEMANTIC_KEY_RE, self.semantic_key) is None:
            raise ValueError("semantic_key must be 16 lowercase hexadecimal characters")
        if self.polarity != "n":
            raise ValueError("pending polarity marker must be the literal 'n'")
        if re.fullmatch(_ORIGIN_HASH_RE, self.origin_hash) is None:
            raise ValueError("origin_hash must be 12 lowercase hexadecimal characters")
        if self.version not in {2, 3}:
            raise ValueError("pending version must be 2 or 3")

    @property
    def pending_id(self) -> str:
        if self.version == 2:
            return f"pending__v2__{self.semantic_key}__{self.origin_hash}"
        return encode_pending(self)


@dataclass(frozen=True)
class EvidenceAggregate:
    """Conservative aggregate over distinct origins in one semantic family."""

    semantic_key: str
    polarity: str
    groups: int
    replications: int
    directness: int
    effect: int
    method_reliability: int
    origins: tuple[str, ...]
    active_origin: str | None = None
    contributing_pending_ids: tuple[str, ...] = ()
    legacy_origins: int = 0

    def __post_init__(self) -> None:
        if re.fullmatch(_SEMANTIC_KEY_RE, self.semantic_key) is None:
            raise ValueError("aggregate semantic key is invalid")
        if self.polarity != "n":
            raise ValueError("aggregate polarity marker must be 'n'")
        _bounded_integer(self.groups, MAX_COUNT, "groups")
        _bounded_integer(self.replications, MAX_COUNT, "replications")
        _bounded_integer(self.directness, MAX_PERCENT, "directness")
        _bounded_integer(self.effect, MAX_PERCENT, "effect")
        _bounded_integer(
            self.method_reliability, MAX_PERCENT, "method_reliability"
        )
        if (
            not self.origins
            or len(self.origins) > MAX_COUNT
            or len(self.origins) != len(set(self.origins))
            or tuple(sorted(self.origins)) != self.origins
            or any(re.fullmatch(_ORIGIN_HASH_RE, value) is None for value in self.origins)
        ):
            raise ValueError("aggregate origins must be unique canonical hashes")
        if self.active_origin is not None and self.active_origin not in self.origins:
            raise ValueError("active aggregate origin must be represented")
        pending_ids = tuple(self.contributing_pending_ids)
        if len(pending_ids) != len(set(pending_ids)) or any(
            decode_pending_id(value) is None for value in pending_ids
        ):
            raise ValueError("aggregate contributing pending IDs must be canonical")
        object.__setattr__(self, "contributing_pending_ids", pending_ids)
        if (
            isinstance(self.legacy_origins, bool)
            or not isinstance(self.legacy_origins, int)
            or not 0 <= self.legacy_origins <= len(self.origins)
        ):
            raise ValueError("legacy origin count is invalid")

    @property
    def origin_count(self) -> int:
        return len(self.origins)

    @property
    def weight(self) -> EvidenceWeight:
        return EvidenceWeight(
            self.groups,
            self.replications,
            self.directness,
            self.effect,
            self.method_reliability,
        )

    @property
    def quality(self) -> float:
        return evidence_quality(self.weight)


@dataclass(frozen=True)
class Revision:
    """Observable bounded confidence revision."""

    prior: float
    posterior: float
    delta_log_odds: float
    quality: float = 0.0
    authorized: bool = True
    reason: str = "authorized_revision"

    def __post_init__(self) -> None:
        for name in ("prior", "posterior", "delta_log_odds", "quality"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"revision {name} must be finite")
        if not 0.0 <= self.prior <= 1.0 or not 0.0 <= self.posterior <= 1.0:
            raise ValueError("revision confidence must be in [0,1]")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("revision quality must be in [0,1]")
        if abs(self.delta_log_odds) > MAX_LOG_ODDS + 1e-9:
            raise ValueError("revision exceeds the log-odds cap")
        actual = bounded_logit(self.posterior) - bounded_logit(self.prior)
        if abs(actual - self.delta_log_odds) > 1e-9:
            raise ValueError("revision movement does not match prior and posterior")
        if not isinstance(self.authorized, bool):
            raise ValueError("revision authorization flag must be boolean")
        if not self.authorized and (
            self.posterior != self.prior or self.delta_log_odds != 0.0
        ):
            raise ValueError("unauthorized revision must preserve the prior")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.reason) is None:
            raise ValueError("revision reason must be a closed code")


@dataclass(frozen=True)
class DecisionReceipt:
    """Compact receipt containing only observable authorization factors."""

    evidence_class: str
    event: str
    target: str
    quality: float
    current_groups: int
    accumulated_origins: int
    prior: float
    delta_log_odds: float
    posterior: float
    action: str
    provenance_status: str = "valid"
    source: str = "none"
    destination: str = "none"
    ood: bool = False
    reason: str = "authorized"

    def __post_init__(self) -> None:
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError("unknown evidence class")
        for name in ("event", "target", "action", "source", "destination", "reason"):
            value = getattr(self, name)
            if not value or _SAFE_RECEIPT_TOKEN.fullmatch(value) is None:
                raise ValueError(f"unsafe receipt {name}")
        if (
            isinstance(self.quality, bool)
            or not isinstance(self.quality, (int, float))
            or not 0.0 <= self.quality <= 1.0
            or not math.isfinite(self.quality)
        ):
            raise ValueError("quality must be finite and in [0,1]")
        if self.provenance_status not in {
            "valid",
            "invalid",
            "retracted",
            "unverified",
        }:
            raise ValueError("unknown provenance status")
        if not isinstance(self.ood, bool):
            raise ValueError("receipt OOD flag must be boolean")
        _bounded_integer(self.current_groups, MAX_COUNT, "current_groups")
        if (
            isinstance(self.accumulated_origins, bool)
            or not isinstance(self.accumulated_origins, int)
            or not 0 <= self.accumulated_origins <= MAX_COUNT
        ):
            raise ValueError("accumulated_origins must be an integer in [0,8]")
        for name in ("prior", "delta_log_odds", "posterior"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.prior <= 1.0 or not 0.0 <= self.posterior <= 1.0:
            raise ValueError("receipt confidence must be in [0,1]")
        if abs(self.delta_log_odds) > MAX_LOG_ODDS + 1e-9:
            raise ValueError("receipt log-odds exceeds the policy cap")
        actual = bounded_logit(self.posterior) - bounded_logit(self.prior)
        if abs(actual - self.delta_log_odds) > 2e-5:
            raise ValueError("receipt movement does not match prior and posterior")

    def render(self) -> str:
        return (
            "receipt{"
            f"class={self.evidence_class};"
            f"event={self.event};"
            f"target={self.target};"
            f"quality={self.quality:.6f};"
            f"provenance={self.provenance_status};"
            f"current_groups={self.current_groups};"
            f"accumulated_origins={self.accumulated_origins};"
            f"prior={self.prior:.6f};"
            f"delta_log_odds={self.delta_log_odds:.6f};"
            f"posterior={self.posterior:.6f};"
            f"source={self.source};"
            f"destination={self.destination};"
            f"ood={'true' if self.ood else 'false'};"
            f"action={self.action};"
            f"reason={self.reason}"
            "}"
        )


@dataclass(frozen=True)
class PreflightContext:
    """Read-only authorization facts for validating a proposed decision."""

    active_evidence_id: str
    claim_confidences: tuple[tuple[str, float], ...]
    pending_ids: frozenset[str]
    allowed_pending_drops: frozenset[str]
    allowed_scope_keys: frozenset[str]
    allowed_scope_values: frozenset[str]
    revisions_authorized: bool
    ood: bool
    allowed_regimes: frozenset[str] = frozenset()
    allowed_axes: frozenset[str] = frozenset()
    max_log_odds: float = MAX_LOG_ODDS


@dataclass(frozen=True)
class PreflightResult:
    valid: bool
    errors: tuple[str, ...]


def origin_fingerprint(evidence_id: str) -> str:
    """Return the bounded stable fingerprint used for origin de-duplication."""
    if not isinstance(evidence_id, str) or not evidence_id or len(evidence_id) > 1024:
        raise ValueError("evidence_id must be a nonempty bounded string")
    return hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:12]


def semantic_fingerprint(
    mechanism: str,
    claim_id: str,
    proposition: str,
    source: str,
    destination: str,
    qualifier: str = "none",
) -> str:
    """Hash only policy-resolved semantic fields into a stable family key."""
    values = (mechanism, claim_id, proposition, source, destination, qualifier)
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > 96
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is None
        for value in values
    ):
        raise ValueError("semantic fields must be bounded resolved identifiers")
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:16]


def encode_pending(pending: PendingEvidence) -> str:
    """Encode one pending record using the exact v3 grammar."""
    if not isinstance(pending, PendingEvidence) or pending.version != 3:
        raise ValueError("encode_pending accepts only a v3 PendingEvidence record")
    weight = pending.weight
    return (
        f"pending__v3__{pending.semantic_key}__{pending.polarity}"
        f"__g{weight.groups}__r{weight.replications}"
        f"__d{weight.directness}__e{weight.effect}"
        f"__m{weight.method_reliability}__{pending.origin_hash}"
    )


def decode_pending(token: str) -> PendingEvidence | None:
    """Strictly decode v3 state; malformed, legacy, or extended tokens fail closed."""

    if not isinstance(token, str):
        return None
    match = _V3_RE.fullmatch(token)
    if match is None:
        return None
    try:
        return PendingEvidence(
            semantic_key=match.group("semantic"),
            polarity=match.group("polarity"),
            weight=EvidenceWeight(
                groups=int(match.group("groups")),
                replications=int(match.group("replications")),
                directness=int(match.group("directness")),
                effect=int(match.group("effect")),
                method_reliability=int(match.group("method")),
            ),
            origin_hash=match.group("origin"),
        )
    except (TypeError, ValueError):
        return None


def decode_legacy_pending(token: str) -> PendingEvidence | None:
    """Decode v2 as one conservative migration-only provisional origin."""

    if not isinstance(token, str):
        return None
    match = _V2_RE.fullmatch(token)
    if match is None:
        return None
    return PendingEvidence(
        semantic_key=match.group("semantic"),
        polarity="n",
        # Conservative, but still capable of contributing after several newer
        # independent reports.  No value is recovered from untrusted prose.
        weight=EvidenceWeight(1, 1, 75, 65, 70),
        origin_hash=match.group("origin"),
        version=2,
    )


def encode_pending_v3(
    semantic_key: str,
    evidence_id: str,
    weight: EvidenceWeight,
    *,
    polarity: str = "n",
) -> str:
    """Convenience encoder that derives the origin from the trusted evidence ID."""
    return encode_pending(
        PendingEvidence(
            semantic_key=semantic_key,
            polarity=polarity,
            weight=weight,
            origin_hash=origin_fingerprint(evidence_id),
        )
    )


def decode_pending_id(token: object, *, allow_v2: bool = True) -> PendingEvidence | None:
    """Decode canonical v3, with an explicit conservative v2 migration option."""
    decoded = decode_pending(token) if isinstance(token, str) else None
    if decoded is not None:
        return decoded
    if isinstance(token, str) and token.startswith("pending__v3__"):
        return None
    return decode_legacy_pending(token) if allow_v2 and isinstance(token, str) else None


def aggregate_evidence(
    records: Iterable[PendingEvidence | str],
    *,
    active_origin: str | None = None,
    active_record: PendingEvidence | None = None,
    origin_cap: int = MAX_COUNT,
) -> EvidenceAggregate:
    """Aggregate exact origins using capped counts and conservative minima.

    Every prior origin contributes at most one group and one replication.  The
    explicitly identified active packet retains its validated packet counts.
    Duplicate records for one origin contribute once and use lower-bound
    factors, so input order and replay schedule cannot inflate quality.
    """
    _bounded_integer(origin_cap, MAX_COUNT, "origin_cap")
    if origin_cap < 1:
        raise ValueError("origin_cap must be positive")
    decoded_records: list[PendingEvidence] = []
    for candidate in records:
        decoded = (
            candidate
            if isinstance(candidate, PendingEvidence)
            else decode_pending_id(candidate)
        )
        if decoded is not None:
            decoded_records.append(decoded)
    if active_record is not None:
        if not isinstance(active_record, PendingEvidence) or active_record.version != 3:
            raise ValueError("active_record must be canonical v3 evidence")
        decoded_records.append(active_record)
        if active_origin is not None and active_origin != active_record.origin_hash:
            raise ValueError("active record and active origin disagree")
        active_origin = active_record.origin_hash
    if active_origin is not None and re.fullmatch(_ORIGIN_HASH_RE, active_origin) is None:
        raise ValueError("active_origin must be a canonical origin hash")

    by_origin: dict[str, list[PendingEvidence]] = {}
    semantic_key: str | None = None
    polarity: str | None = None
    for record in decoded_records:
        if semantic_key is None:
            semantic_key = record.semantic_key
            polarity = record.polarity
        elif record.semantic_key != semantic_key or record.polarity != polarity:
            raise ValueError("cannot aggregate different semantic families or polarities")
        by_origin.setdefault(record.origin_hash, []).append(record)
    if semantic_key is None or polarity is None or not by_origin:
        raise ValueError("at least one evidence record is required")
    if active_origin is not None and active_origin not in by_origin:
        raise ValueError("active origin is absent from aggregate records")

    ordered_origins = sorted(by_origin)
    if active_origin is None:
        selected_origins = ordered_origins[:origin_cap]
    else:
        selected_origins = [active_origin]
        selected_origins.extend(
            origin
            for origin in ordered_origins
            if origin != active_origin
        )
        selected_origins = selected_origins[:origin_cap]

    group_contributions: list[int] = []
    replication_contributions: list[int] = []
    directness: list[int] = []
    effects: list[int] = []
    methods: list[int] = []
    pending_ids: set[str] = set()
    legacy_origins = 0
    for origin in selected_origins:
        origin_records = by_origin[origin]
        minimum_groups = min(record.weight.groups for record in origin_records)
        minimum_replications = min(
            record.weight.replications for record in origin_records
        )
        if origin == active_origin:
            group_contributions.append(minimum_groups)
            replication_contributions.append(minimum_replications)
        else:
            group_contributions.append(min(1, minimum_groups))
            replication_contributions.append(min(1, minimum_replications))
            pending_ids.update(record.pending_id for record in origin_records)
            if any(record.version == 2 for record in origin_records):
                legacy_origins += 1
        directness.append(min(record.weight.directness for record in origin_records))
        effects.append(min(record.weight.effect for record in origin_records))
        methods.append(
            min(record.weight.method_reliability for record in origin_records)
        )

    return EvidenceAggregate(
        semantic_key=semantic_key,
        polarity=polarity,
        groups=min(MAX_COUNT, sum(group_contributions)),
        replications=min(MAX_COUNT, sum(replication_contributions)),
        directness=min(directness),
        effect=min(effects),
        method_reliability=min(methods),
        origins=tuple(sorted(selected_origins)),
        active_origin=active_origin,
        contributing_pending_ids=tuple(sorted(pending_ids)),
        legacy_origins=legacy_origins,
    )


def aggregate_with_active(
    active: PendingEvidence,
    pending: Iterable[PendingEvidence | str],
    *,
    origin_cap: int = MAX_COUNT,
) -> EvidenceAggregate:
    """Explicit active-packet API used by the scored sequential policy."""
    return aggregate_evidence(
        pending,
        active_record=active,
        origin_cap=origin_cap,
    )


def saturation(count: int) -> float:
    """Monotonic diminishing-returns curve for validated integer counts."""

    _bounded_integer(count, MAX_COUNT, "count")
    if count <= 1:
        return 0.0
    if count == 2:
        return 0.4
    if count == 3:
        return 0.75
    return 1.0


def evidence_quality(weight: EvidenceWeight) -> float:
    """Recompute quality from bounded aggregate factors; never average scores."""

    quality = (
        0.40 * saturation(weight.groups)
        + 0.15 * saturation(weight.replications)
        + 0.20 * (weight.directness / 100.0)
        + 0.15 * (weight.effect / 100.0)
        + 0.10 * (weight.method_reliability / 100.0)
    )
    return min(1.0, max(0.0, quality))


def contract_accepts_event(
    contract: ClaimContract,
    *,
    proposition: str,
    mechanism: str,
    source_role: str | None,
    destination_role: str | None,
    direction: int,
    ambiguous: bool = False,
) -> bool:
    """Authorize only resolved fields present in the closed claim contract."""
    if not isinstance(contract, ClaimContract) or ambiguous or direction not in {-1, 1}:
        return False
    if proposition not in contract.propositions or mechanism not in contract.mechanisms:
        return False
    if direction < 0 and not contract.allow_contradiction:
        return False
    if direction > 0 and not contract.allow_confirmation:
        return False
    if contract.source_roles and source_role not in contract.source_roles:
        return False
    if contract.destination_roles and destination_role not in contract.destination_roles:
        return False
    return True


def evidence_meets_contract(
    aggregate: EvidenceAggregate, contract: ClaimContract
) -> bool:
    """Use the stricter threshold when support accumulated across origins."""
    if not isinstance(aggregate, EvidenceAggregate) or not isinstance(
        contract, ClaimContract
    ):
        return False
    required_groups = (
        contract.min_accumulated_groups
        if aggregate.origin_count > 1
        else contract.min_groups
    )
    return (
        aggregate.groups >= required_groups
        and aggregate.replications >= 1
        and aggregate.quality >= contract.min_quality
    )


def bounded_logit(probability: float) -> float:
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
    ):
        raise ValueError("probability must be finite")
    bounded = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def bounded_sigmoid(log_odds: float) -> float:
    if (
        isinstance(log_odds, bool)
        or not isinstance(log_odds, (int, float))
        or not math.isfinite(float(log_odds))
    ):
        raise ValueError("log_odds must be finite")
    bounded = min(30.0, max(-30.0, float(log_odds)))
    if bounded >= 0.0:
        exponent = math.exp(-bounded)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(bounded)
    return exponent / (1.0 + exponent)


def compute_revision(
    prior: float,
    direction: int,
    aggregate: EvidenceAggregate,
    *,
    contract: ClaimContract | None = None,
    max_log_odds: float = MAX_LOG_ODDS,
) -> Revision:
    """Return a bounded log-odds revision or an explicit zero-movement abstention."""
    if (
        isinstance(prior, bool)
        or not isinstance(prior, (int, float))
        or not math.isfinite(float(prior))
        or not 0.0 <= float(prior) <= 1.0
    ):
        raise ValueError("prior must be finite and in [0,1]")
    prior_value = float(prior)
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if not isinstance(aggregate, EvidenceAggregate):
        raise TypeError("aggregate must be EvidenceAggregate")
    if (
        isinstance(max_log_odds, bool)
        or not isinstance(max_log_odds, (int, float))
        or not math.isfinite(float(max_log_odds))
        or not 0.0 < float(max_log_odds) <= MAX_LOG_ODDS
    ):
        raise ValueError("max_log_odds must be in (0,3]")

    def abstain(reason: str) -> Revision:
        return Revision(
            prior_value,
            prior_value,
            0.0,
            aggregate.quality,
            False,
            reason,
        )

    if contract is not None:
        if direction < 0 and not contract.allow_contradiction:
            return abstain("contradiction_disallowed")
        if direction > 0 and not contract.allow_confirmation:
            return abstain("confirmation_disallowed")
        if not evidence_meets_contract(aggregate, contract):
            return abstain("insufficient_evidence")
        movement_cap = float(max_log_odds)
        strong = (
            aggregate.groups >= contract.strong_groups
            and aggregate.quality >= contract.strong_quality
        )
    else:
        movement_cap = float(max_log_odds)
        strong = aggregate.groups >= 3 and aggregate.quality >= 0.82

    if direction < 0:
        movement = min(2.8, max(0.0, 5.0 * (aggregate.quality - 0.45)))
        movement *= 0.55 + 0.45 * (aggregate.effect / 100.0)
    elif prior_value < 0.70 and strong:
        movement = 0.65 + 0.75 * aggregate.quality
    else:
        movement = 0.15 + 0.30 * aggregate.quality
    movement = min(movement_cap, max(0.0, movement))
    if movement < 1e-12:
        return abstain("saturated_or_negligible")
    posterior = bounded_sigmoid(
        bounded_logit(prior_value) + direction * movement
    )
    actual = bounded_logit(posterior) - bounded_logit(prior_value)
    return Revision(
        prior_value,
        posterior,
        actual,
        aggregate.quality,
        True,
        "authorized_revision",
    )


def parse_receipt(text: str) -> DecisionReceipt | None:
    """Parse exactly one strict receipt embedded in a rationale."""

    if not isinstance(text, str):
        return None
    matches = re.findall(r"receipt\{([^{}]*)\}", text)
    if len(matches) != 1:
        return None
    parts = matches[0].split(";")
    if len(parts) != 15 or any("=" not in part for part in parts):
        return None
    pairs = [part.split("=", 1) for part in parts]
    expected = (
        "class",
        "event",
        "target",
        "quality",
        "provenance",
        "current_groups",
        "accumulated_origins",
        "prior",
        "delta_log_odds",
        "posterior",
        "source",
        "destination",
        "ood",
        "action",
        "reason",
    )
    if tuple(key for key, _ in pairs) != expected:
        return None
    values = dict(pairs)
    if values.get("ood") not in {"true", "false"}:
        return None
    try:
        return DecisionReceipt(
            evidence_class=values["class"],
            event=values["event"],
            target=values["target"],
            quality=float(values["quality"]),
            current_groups=int(values["current_groups"]),
            accumulated_origins=int(values["accumulated_origins"]),
            prior=float(values["prior"]),
            delta_log_odds=float(values["delta_log_odds"]),
            posterior=float(values["posterior"]),
            action=values["action"],
            provenance_status=values["provenance"],
            source=values["source"],
            destination=values["destination"],
            ood=values["ood"] == "true",
            reason=values["reason"],
        )
    except (TypeError, ValueError):
        return None


def build_receipt(
    *,
    evidence_class: str,
    event: str,
    target: str,
    current_groups: int,
    action: str,
    aggregate: EvidenceAggregate | None = None,
    revision: Revision | None = None,
    prior: float = 0.5,
    provenance_status: str = "valid",
    source: str = "none",
    destination: str = "none",
    ood: bool = False,
    reason: str = "authorized",
) -> DecisionReceipt:
    """Build one receipt from the exact factors used by authorization."""
    if revision is None:
        prior_value = float(prior)
        posterior = prior_value
        movement = 0.0
    else:
        prior_value = revision.prior
        posterior = revision.posterior
        movement = revision.delta_log_odds
    return DecisionReceipt(
        evidence_class=evidence_class,
        event=event,
        target=target,
        quality=aggregate.quality if aggregate is not None else 0.0,
        current_groups=current_groups,
        accumulated_origins=aggregate.origin_count if aggregate is not None else 0,
        prior=prior_value,
        delta_log_odds=movement,
        posterior=posterior,
        action=action,
        provenance_status=provenance_status,
        source=source,
        destination=destination,
        ood=ood,
        reason=reason,
    )


def render_receipt(
    receipt: DecisionReceipt, policy: LabPolicy = DEFAULT_LAB_POLICY
) -> str:
    """Render only receipts whose class/actions are in the closed lab policy."""
    if not isinstance(receipt, DecisionReceipt) or not isinstance(policy, LabPolicy):
        raise TypeError("receipt and policy must have the declared immutable types")
    if receipt.evidence_class not in policy.evidence_classes:
        raise ValueError("receipt evidence class is outside policy")
    actions = receipt.action.split("+")
    if not actions or any(action not in policy.operations for action in actions):
        raise ValueError("receipt action is outside policy")
    return receipt.render()


def _logit(probability: float) -> float:
    bounded = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def _payload_has_exact_keys(payload: Mapping[str, object], keys: set[str]) -> bool:
    return set(payload) == keys


def semantic_preflight(
    deltas: Sequence[Any], context: PreflightContext
) -> PreflightResult:
    """Validate a complete proposed decision before it reaches the official API."""

    errors: list[str] = []
    if not deltas:
        errors.append("decision must contain at least one delta")
        return PreflightResult(False, tuple(errors))
    claim_confidences = dict(context.claim_confidences)
    operations = [
        delta.op
        for delta in deltas
        if isinstance(getattr(delta, "op", None), str)
        and isinstance(getattr(delta, "evidence_id", None), str)
        and isinstance(getattr(delta, "payload", None), dict)
    ]
    if len(operations) != len(deltas):
        errors.append("every proposed operation must be a Delta")
    if "no_op" in operations and len(operations) != 1:
        errors.append("no_op cannot be mixed with mutations")
    if context.ood and any(
        operation not in {"no_op", "propose_regime", "propose_axis"}
        for operation in operations
    ):
        errors.append("OOD decision cannot mutate in-model state")
    if not context.ood and any(
        operation in {"propose_regime", "propose_axis"}
        for operation in operations
    ):
        errors.append("in-model decision cannot expand the OOD ontology")
    revised: set[str] = set()

    for delta in deltas:
        if not (
            isinstance(getattr(delta, "op", None), str)
            and isinstance(getattr(delta, "evidence_id", None), str)
            and isinstance(getattr(delta, "payload", None), dict)
        ):
            continue
        if delta.op not in POLICY_OPERATIONS:
            errors.append(f"unknown closed-vocabulary operation: {delta.op}")
            continue
        if delta.evidence_id != context.active_evidence_id:
            errors.append("delta has incorrect evidence attribution")
        if not isinstance(delta.payload, dict):
            errors.append(f"{delta.op} payload must be a dictionary")
            continue

        payload = delta.payload
        if delta.op == "no_op":
            if payload:
                errors.append("no_op payload must be empty")
        elif delta.op == "revise_confidence":
            if not _payload_has_exact_keys(payload, {"claim_id", "new_confidence"}):
                errors.append("revise_confidence payload has unexpected fields")
                continue
            claim_id = payload.get("claim_id")
            proposed = payload.get("new_confidence")
            if not context.revisions_authorized:
                errors.append("confidence revision is not authorized")
            if context.ood:
                errors.append("OOD decision cannot revise an in-model claim")
            if not isinstance(claim_id, str) or claim_id not in claim_confidences:
                errors.append("confidence revision targets an unknown claim")
                continue
            if claim_id in revised:
                errors.append("second confidence revision to the same claim")
            revised.add(claim_id)
            if (
                isinstance(proposed, bool)
                or not isinstance(proposed, (int, float))
                or not math.isfinite(float(proposed))
            ):
                errors.append("revision requires finite confidence")
                continue
            proposed_float = float(proposed)
            if not 0.0 <= proposed_float <= 1.0:
                errors.append("revision confidence is outside [0,1]")
                continue
            movement = abs(
                _logit(proposed_float) - _logit(claim_confidences[claim_id])
            )
            if movement > context.max_log_odds + 1e-9:
                errors.append("revision exceeds the bounded log-odds cap")
        elif delta.op == "set_scope":
            if not _payload_has_exact_keys(payload, {"claim_id", "scope"}):
                errors.append("set_scope payload has unexpected fields")
                continue
            claim_id = payload.get("claim_id")
            scope = payload.get("scope")
            if not context.revisions_authorized:
                errors.append("scope revision is not authorized")
            if context.ood:
                errors.append("OOD decision cannot scope an in-model claim")
            if not isinstance(claim_id, str) or claim_id not in claim_confidences:
                errors.append("scope revision targets an unknown claim")
            if not isinstance(scope, dict) or not scope:
                errors.append("scope must be a nonempty dictionary")
                continue
            for key, value in scope.items():
                if key not in context.allowed_scope_keys:
                    errors.append(f"scope key is outside the lab contract: {key}")
                elif key == "exception_under":
                    if (
                        not isinstance(value, str)
                        or value not in context.allowed_scope_values
                    ):
                        errors.append("scope value is outside the lab ontology")
                elif value is not True:
                    errors.append("boolean scope marker must be exactly true")
        elif delta.op == "drop_claim":
            if not _payload_has_exact_keys(payload, {"claim_id"}):
                errors.append("drop_claim payload has unexpected fields")
                continue
            claim_id = payload.get("claim_id")
            if not isinstance(claim_id, str):
                errors.append("pending drop requires a string identifier")
                continue
            if claim_id not in context.pending_ids:
                errors.append("pending drop targets nonexistent state")
            if claim_id not in context.allowed_pending_drops:
                errors.append("pending drop does not match the exact semantic family")
            if decode_pending(claim_id) is None and decode_legacy_pending(claim_id) is None:
                errors.append("pending drop targets a malformed versioned token")
        elif delta.op == "hold_pending":
            if not _payload_has_exact_keys(payload, {"claim_id", "note"}):
                errors.append("hold_pending payload has unexpected fields")
                continue
            claim_id = payload.get("claim_id")
            note = payload.get("note")
            record = decode_pending(claim_id) if isinstance(claim_id, str) else None
            if record is None:
                errors.append("pending write requires a well-formed v3 token")
            elif record.origin_hash != origin_fingerprint(context.active_evidence_id):
                errors.append("pending token does not match the active evidence origin")
            if not isinstance(note, str) or not note or len(note) > 512:
                errors.append("pending note must be a bounded nonempty string")
        elif delta.op == "propose_regime":
            if not _payload_has_exact_keys(payload, {"regime"}):
                errors.append("propose_regime payload has unexpected fields")
            elif (
                not isinstance(payload.get("regime"), str)
                or payload.get("regime") not in context.allowed_regimes
            ):
                errors.append("proposed regime is outside the closed domain contract")
        elif delta.op == "propose_axis":
            if not _payload_has_exact_keys(payload, {"axis"}):
                errors.append("propose_axis payload has unexpected fields")
            elif (
                not isinstance(payload.get("axis"), str)
                or payload.get("axis") not in context.allowed_axes
            ):
                errors.append("proposed axis is outside the closed domain contract")
        else:
            # The official vocabulary is larger than this policy's authorization
            # surface.  New graph objects/status writes require a separate contract.
            errors.append(f"operation is not authorized by the scored policy: {delta.op}")

    return PreflightResult(not errors, tuple(errors))


__all__ = [
    "ClaimContract",
    "DEFAULT_LAB_POLICY",
    "DecisionReceipt",
    "EvidenceAggregate",
    "EvidenceWeight",
    "LabPolicy",
    "PendingEvidence",
    "PreflightContext",
    "PreflightResult",
    "Revision",
    "aggregate_evidence",
    "aggregate_with_active",
    "bounded_logit",
    "bounded_sigmoid",
    "build_receipt",
    "compute_revision",
    "contract_accepts_event",
    "decode_legacy_pending",
    "decode_pending",
    "decode_pending_id",
    "encode_pending",
    "encode_pending_v3",
    "evidence_meets_contract",
    "evidence_quality",
    "origin_fingerprint",
    "parse_receipt",
    "render_receipt",
    "saturation",
    "semantic_fingerprint",
    "semantic_preflight",
]
