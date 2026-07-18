"""Input normalization (G06) -- HYGIENE, NOT THE SECURITY BOUNDARY.

Per the threat model (EVIDENCE_FIREWALL_V2.md section 2): there is no reliable
detector for arbitrarily encoded hidden instructions. Normalization here
(NFKC, zero-width/bidi stripping, size limits, suspicious-encoding flags) is
useful hygiene that shrinks the attack surface a little; the actual security
boundary is the integrity lattice, the strict EvidenceIR compiler, and the
deterministic reference monitor. The system must remain safe even when this
module fails to notice an injection.

Determinism contract: ``normalize`` is a pure function of its input. Witness
spans elsewhere in the pipeline (core.types.Span) refer to character offsets
in the *normalized* text produced here, so the same raw input must always
yield the same normalized text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Audit identifier for the exact ordered normalization algorithm below.
#:
#: Any change that can alter ``NormalizedInput.text`` MUST introduce a new
#: version. Ledger/API consumers can then prove which algorithm produced the
#: text addressed by witness spans instead of assuming today's implementation.
NORMALIZATION_VERSION = "nfkc-zws-bidi-crlf-v1"

#: Hard input size limit (characters, applied to the raw input).
MAX_INPUT_CHARS = 20_000

#: Zero-width / invisible characters: U+200B..U+200F, U+FEFF (BOM/ZWNBSP),
#: U+2060 (word joiner). Classic carriers for hidden-channel payloads.
_ZERO_WIDTH_CHARS = "".join(chr(c) for c in range(0x200B, 0x2010)) + chr(0xFEFF) + chr(0x2060)
_ZERO_WIDTH_RE = re.compile("[" + re.escape(_ZERO_WIDTH_CHARS) + "]")

#: Bidirectional control characters: U+202A..U+202E (embedding/override),
#: U+2066..U+2069 (isolates). Used for visual-reordering spoofing.
_BIDI_CHARS = "".join(chr(c) for c in range(0x202A, 0x202F)) + "".join(
    chr(c) for c in range(0x2066, 0x206A)
)
_BIDI_RE = re.compile("[" + re.escape(_BIDI_CHARS) + "]")

#: A contiguous run of base64-alphabet characters long enough to look like an
#: encoded payload rather than a word. Flagged, never rejected: base64 content
#: is still just low-integrity text under the lattice.
_BASE64_RUN_RE = re.compile(r"[A-Za-z0-9+/=]{120,}")

#: Fraction of non-ASCII characters above which we flag the input.
_NON_ASCII_FLAG_RATIO = 0.30


class InputTooLarge(ValueError):
    """Raised when the raw input exceeds MAX_INPUT_CHARS."""


class NormalizedInput(BaseModel):
    """Result of hygiene normalization.

    ``text`` is the canonical normalized text that all downstream witness
    spans index into. ``text_sha256`` binds audit records to those exact UTF-8
    bytes, while ``normalization_version`` identifies the algorithm that
    produced them. ``flags`` are advisory hygiene signals for escalation
    heuristics and audit -- they never gate acceptance by themselves.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str
    flags: list[str]
    original_length: int = Field(ge=0)
    normalization_version: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _audit_context_matches_text(self) -> "NormalizedInput":
        if self.normalization_version != NORMALIZATION_VERSION:
            raise ValueError(
                f"normalization_version must be {NORMALIZATION_VERSION!r}"
            )
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_sha256 != expected:
            raise ValueError("text_sha256 digest does not match normalized text")
        return self


def normalize(raw: str) -> NormalizedInput:
    """Deterministically normalize untrusted raw text.

    Pipeline (fixed order, pure function of ``raw``):
      1. Reject inputs longer than MAX_INPUT_CHARS (InputTooLarge).
      2. Unicode NFKC normalization (folds ligatures, fullwidth forms, ...).
      3. Strip zero-width characters (flag ``zero_width_stripped``).
      4. Strip bidirectional control characters (flag ``bidi_stripped``).
      5. Collapse CRLF line endings to LF.
      6. Flag -- never reject -- likely-encoded payloads:
         ``possible_base64`` for contiguous base64-looking runs >= 120 chars,
         ``high_non_ascii`` when > 30% of characters are non-ASCII.
    """
    if not isinstance(raw, str):
        raise TypeError("raw input must be a string")

    original_length = len(raw)
    if original_length > MAX_INPUT_CHARS:
        raise InputTooLarge(
            f"input is {original_length} chars; limit is {MAX_INPUT_CHARS}"
        )

    flags: list[str] = []

    text = unicodedata.normalize("NFKC", raw)

    stripped = _ZERO_WIDTH_RE.sub("", text)
    if stripped != text:
        flags.append("zero_width_stripped")
    text = stripped

    stripped = _BIDI_RE.sub("", text)
    if stripped != text:
        flags.append("bidi_stripped")
    text = stripped

    text = text.replace("\r\n", "\n")

    if _BASE64_RUN_RE.search(text):
        flags.append("possible_base64")

    if text:
        non_ascii = sum(1 for ch in text if ord(ch) > 0x7F)
        if non_ascii / len(text) > _NON_ASCII_FLAG_RATIO:
            flags.append("high_non_ascii")

    return NormalizedInput(
        text=text,
        flags=flags,
        original_length=original_length,
        normalization_version=NORMALIZATION_VERSION,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
