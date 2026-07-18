"""Compiler adapters (G06): untrusted raw text -> typed EvidenceIR.

This is the L0 -> L1 seam of the integrity lattice. A compiler may return
ONLY an ``EvidenceIR`` or raise ``CompileError`` -- there is no other output
channel, so raw text can never directly become a privileged action. The
model proposes an observation; deterministic code downstream decides what,
if anything, it is allowed to influence.

Local revalidation is non-negotiable: even when the provider offers
structured output (Gemini's response_schema), the returned JSON is ALWAYS
re-parsed through ``EvidenceIR.model_validate_json`` locally before it is
returned. ``EvidenceIR`` is strict (``extra="forbid"``, no integrity /
confidence / verdict fields), so schema-level self-promotion is blocked at
parse time.

Failure policy: validation errors become L0/rejected events upstream and are
NEVER retried with a looser schema. Loosening the schema on failure would
hand the attacker a downgrade lever; the only retry-shaped behavior allowed
in the system is the escalation lane's *stricter* reparse (G07), never a
laxer one.
"""

from __future__ import annotations

import os
import re
from typing import Any

from pydantic import ValidationError

from core.types import EvidenceIR
from llm.normalize import normalize

#: Env var naming the fast-lane model; see EVIDENCE_FIREWALL_V2.md section 6.
FAST_MODEL_ENV = "EPISTEMICOS_FAST_MODEL"
# Pin the documented stable model. The ``-latest`` alias can hot-swap to a
# materially different release, which is undesirable for an auditable demo.
DEFAULT_FAST_MODEL = "gemini-3.1-flash-lite"

#: Fixture convention (shared with G07/G08/G10 red-team fixtures): the fake
#: compiler extracts the JSON object between these markers in the NORMALIZED
#: input text and validates it exactly like the real compiler validates
#: provider output.
IR_OPEN = "<<IR>>"
IR_CLOSE = "<</IR>>"

_IR_BLOCK_RE = re.compile(re.escape(IR_OPEN) + r"(.*?)" + re.escape(IR_CLOSE), re.DOTALL)

_EXTRACTION_PROMPT = (
    "You are an evidence extraction compiler. Read the observation text and "
    "emit a single JSON object matching the EvidenceIR schema. Every "
    "support_span must give [start, end) character offsets into the exact "
    "input text you were given. Use \"UNMODELED\" for identifiers the text "
    "does not establish. Emit JSON only."
)

# Public, attacker-safe failure classifications. These values cross the L0
# boundary into API responses and the ledger, so they must never contain
# provider messages, validation paths, input values, or other exception text.
REASON_IR_BLOCK_MISSING = "ir_block_missing"
REASON_EVIDENCE_IR_JSON_INVALID = "evidence_ir_json_invalid"
REASON_EVIDENCE_IR_SCHEMA_INVALID = "evidence_ir_schema_invalid"
REASON_PROVIDER_FAILURE = "provider_failure"
REASON_PROVIDER_EMPTY_RESPONSE = "provider_empty_response"


class CompileError(Exception):
    """Raised with a stable, attacker-safe public failure reason.

    Upstream, a CompileError becomes an L0/rejected ledger event. It is never
    retried with a looser schema. Technical exceptions remain available via
    exception chaining for internal diagnostics, but ``reason`` is safe to
    serialize into responses and the ledger.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FakeCompiler:
    """Deterministic compiler for tests and demos (no provider, no network).

    Normalizes the input, then extracts the JSON object between ``<<IR>>``
    and ``<</IR>>`` markers in the normalized text and strictly validates it
    as EvidenceIR -- the exact same local revalidation path GeminiCompiler
    applies to provider output. This lets fixtures fully control compiler
    output while keeping the L1 gate honest.
    """

    def __init__(self) -> None:
        self.last_usage: dict[str, Any] | None = None

    async def compile(self, raw_text: str) -> EvidenceIR:
        normalized = normalize(raw_text)
        match = _IR_BLOCK_RE.search(normalized.text)
        if match is None:
            raise CompileError(REASON_IR_BLOCK_MISSING)
        payload = match.group(1).strip()
        try:
            # Identical local revalidation call to GeminiCompiler's.
            ir = EvidenceIR.model_validate_json(payload)
        except ValidationError as exc:
            raise CompileError(_validation_reason(exc)) from exc
        self.last_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
        return ir


class GeminiCompiler:
    """Gemini-backed compiler (fast lane).

    The google-genai client is imported and constructed lazily inside
    ``compile()``, so importing this module (and constructing this class)
    works with no GEMINI_API_KEY and without google-genai being importable.

    ``last_usage`` exposes the provider usage metadata of the most recent
    call (input/output/thinking/cached/total token counts) -- the
    usage-metadata seam consumed by G07/G08 metrics.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get(FAST_MODEL_ENV, DEFAULT_FAST_MODEL)
        self.last_usage: dict[str, Any] | None = None
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily import google-genai and build the client (cached)."""
        if self._client is None:
            from google import genai  # deferred: no import cost / key needed at module load

            self._client = genai.Client()
        return self._client

    async def compile(self, raw_text: str) -> EvidenceIR:
        normalized = normalize(raw_text)
        try:
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=[_EXTRACTION_PROMPT, normalized.text],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": EvidenceIR.model_json_schema(),
                },
            )
        except Exception as exc:  # provider/transport failure -> CompileError
            raise CompileError(REASON_PROVIDER_FAILURE) from exc

        self.last_usage = _usage_dict(response)

        text = getattr(response, "text", None)
        if not text:
            raise CompileError(REASON_PROVIDER_EMPTY_RESPONSE)
        try:
            # ALWAYS revalidate locally; never trust provider-side validation.
            return EvidenceIR.model_validate_json(text)
        except ValidationError as exc:
            raise CompileError(_validation_reason(exc)) from exc


def _validation_reason(exc: ValidationError) -> str:
    """Classify validation failure without exposing attacker-controlled details."""
    if any(error.get("type") == "json_invalid" for error in exc.errors()):
        return REASON_EVIDENCE_IR_JSON_INVALID
    return REASON_EVIDENCE_IR_SCHEMA_INVALID


def _usage_dict(response: Any) -> dict[str, Any] | None:
    """Extract token usage metadata from a google-genai response, if present."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    return {
        "input_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
        "thinking_tokens": getattr(meta, "thoughts_token_count", None),
        "cached_tokens": getattr(meta, "cached_content_token_count", None),
        "total_tokens": getattr(meta, "total_token_count", None),
    }
