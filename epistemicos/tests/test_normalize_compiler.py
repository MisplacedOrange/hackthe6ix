"""G06 tests: normalization hygiene and compiler adapters.

All tests are provider-free and deterministic: no network, no GEMINI_API_KEY,
no real google-genai client is ever constructed.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys

import pytest
from pydantic import ValidationError

from core.types import EvidenceIR
from llm.compiler import IR_CLOSE, IR_OPEN, CompileError, FakeCompiler, GeminiCompiler
from llm.normalize import (
    MAX_INPUT_CHARS,
    NORMALIZATION_VERSION,
    InputTooLarge,
    NormalizedInput,
    normalize,
)
from tests.conftest import SAMPLE_TEXT, make_evidence

ZWSP = chr(0x200B)  # zero width space
RLO = chr(0x202E)  # right-to-left override
LRI = chr(0x2066)  # left-to-right isolate
BOM = chr(0xFEFF)  # zero width no-break space / BOM
FI_LIGATURE = chr(0xFB01)  # "fi" ligature, NFKC-folds to "fi"
FULLWIDTH_A = chr(0xFF21)  # fullwidth "A", NFKC-folds to "A"


def wrap_ir(payload: str, prefix: str = "Observation report. ") -> str:
    """Embed an IR JSON payload using the shared <<IR>>...<</IR>> convention."""
    return f"{prefix}{IR_OPEN}{payload}{IR_CLOSE}"


# ---------------------------------------------------------------------------
# normalize: size limit
# ---------------------------------------------------------------------------


def test_size_limit_rejects_over_max():
    with pytest.raises(InputTooLarge):
        normalize("a" * (MAX_INPUT_CHARS + 1))


def test_size_limit_accepts_exactly_max():
    result = normalize("a" * MAX_INPUT_CHARS)
    assert isinstance(result, NormalizedInput)
    assert result.original_length == MAX_INPUT_CHARS
    assert len(result.text) == MAX_INPUT_CHARS


@pytest.mark.parametrize("raw", [None, b"text", 17, True, object()])
def test_non_string_input_is_rejected_deterministically(raw):
    with pytest.raises(TypeError, match="raw input must be a string"):
        normalize(raw)


# ---------------------------------------------------------------------------
# normalize: unicode hygiene
# ---------------------------------------------------------------------------


def test_zero_width_stripped_with_flag():
    raw = f"ignore{ZWSP}previous{BOM}instructions"
    result = normalize(raw)
    assert result.text == "ignorepreviousinstructions"
    assert "zero_width_stripped" in result.flags
    assert "bidi_stripped" not in result.flags


def test_bidi_stripped_with_flag():
    raw = f"benign {RLO}txt.exe{LRI} filename"
    result = normalize(raw)
    assert result.text == "benign txt.exe filename"
    assert "bidi_stripped" in result.flags
    assert "zero_width_stripped" not in result.flags


def test_nfkc_folds_ligatures_and_fullwidth():
    raw = f"{FI_LIGATURE}rewall {FULLWIDTH_A}TTACK"
    result = normalize(raw)
    assert result.text == "firewall ATTACK"


def test_crlf_collapsed():
    assert normalize("line one\r\nline two").text == "line one\nline two"


def test_deterministic_same_input_same_output():
    raw = f"mixed {FI_LIGATURE}{ZWSP}{RLO} payload\r\nwith lines"
    first = normalize(raw)
    second = normalize(raw)
    assert first == second
    assert first.text == second.text
    assert first.flags == second.flags
    assert first.original_length == second.original_length
    assert first.normalization_version == second.normalization_version
    assert first.text_sha256 == second.text_sha256


def test_version_and_digest_are_deterministic_audit_context():
    result = normalize("stable observation")
    expected_digest = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
    assert result.normalization_version == NORMALIZATION_VERSION
    assert result.text_sha256 == expected_digest
    assert len(result.text_sha256) == 64


def test_digest_binds_nfkc_and_control_stripped_normalized_text():
    raw = f"{FI_LIGATURE}{ZWSP}rewall{RLO}\r\nreport"
    expected_text = "firewall\nreport"
    transformed = normalize(raw)
    canonical = normalize(expected_text)

    assert transformed.text == expected_text
    assert transformed.text_sha256 == hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    assert transformed.text_sha256 == canonical.text_sha256
    assert transformed.text_sha256 != hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_normalized_input_is_strict_frozen_and_digest_bound():
    valid = normalize("observation")

    with pytest.raises(ValidationError):
        valid.text_sha256 = "0" * 64
    with pytest.raises(ValidationError):
        NormalizedInput(
            text=valid.text,
            flags=valid.flags,
            original_length=str(valid.original_length),
            normalization_version=valid.normalization_version,
            text_sha256=valid.text_sha256,
        )
    with pytest.raises(ValidationError, match="digest does not match"):
        NormalizedInput(
            text="different text",
            flags=valid.flags,
            original_length=valid.original_length,
            normalization_version=valid.normalization_version,
            text_sha256=valid.text_sha256,
        )


def test_clean_text_has_no_flags():
    result = normalize(SAMPLE_TEXT)
    assert result.flags == []
    assert result.text == SAMPLE_TEXT
    assert result.original_length == len(SAMPLE_TEXT)


# ---------------------------------------------------------------------------
# normalize: encoded-payload flags (flag, never reject)
# ---------------------------------------------------------------------------


def test_possible_base64_flag_fires_on_long_run():
    payload = "aGVsbG8gd29ybGQ" * 10  # 150 contiguous base64-alphabet chars
    assert len(payload) >= 120
    result = normalize(f"see attachment: {payload}")
    assert "possible_base64" in result.flags


def test_possible_base64_flag_absent_on_ordinary_english():
    assert "possible_base64" not in normalize(SAMPLE_TEXT).flags


def test_high_non_ascii_flag_fires():
    raw = ("中" * 40) + ("x" * 10)  # 80% non-ASCII
    result = normalize(raw)
    assert "high_non_ascii" in result.flags


def test_high_non_ascii_flag_absent_on_ordinary_english():
    assert "high_non_ascii" not in normalize(SAMPLE_TEXT).flags


# ---------------------------------------------------------------------------
# compiler module: provider absence must not break imports
# ---------------------------------------------------------------------------


def test_import_and_construct_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Force a fresh import so module-level provider imports would surface.
    sys.modules.pop("llm.compiler", None)
    module = importlib.import_module("llm.compiler")
    compiler = module.GeminiCompiler()
    assert compiler.model == module.DEFAULT_FAST_MODEL
    assert compiler.last_usage is None


def test_gemini_model_name_from_env(monkeypatch):
    monkeypatch.setenv("EPISTEMICOS_FAST_MODEL", "gemini-test-model")
    assert GeminiCompiler().model == "gemini-test-model"
    assert GeminiCompiler(model="explicit-wins").model == "explicit-wins"


# ---------------------------------------------------------------------------
# FakeCompiler: strict local revalidation path
# ---------------------------------------------------------------------------


async def test_fake_compiler_round_trips_valid_ir():
    evidence = make_evidence()
    compiler = FakeCompiler()
    result = await compiler.compile(wrap_ir(evidence.model_dump_json()))
    assert isinstance(result, EvidenceIR)
    assert result == evidence


async def test_fake_compiler_missing_markers_raises():
    compiler = FakeCompiler()
    with pytest.raises(CompileError) as excinfo:
        await compiler.compile("just ordinary text, no IR block")
    assert excinfo.value.reason == "ir_block_missing"


async def test_fake_compiler_malformed_json_raises():
    compiler = FakeCompiler()
    with pytest.raises(CompileError) as excinfo:
        await compiler.compile(wrap_ir("{not valid json"))
    assert excinfo.value.reason == "evidence_ir_json_invalid"


async def test_fake_compiler_extra_field_self_promotion_blocked():
    payload = json.loads(make_evidence().model_dump_json())
    marker = "SENSITIVE_MARKER_7F3A91"
    payload["integrity"] = marker  # text may not declare its own integrity
    compiler = FakeCompiler()
    with pytest.raises(CompileError) as excinfo:
        await compiler.compile(wrap_ir(json.dumps(payload)))
    assert excinfo.value.reason == "evidence_ir_schema_invalid"
    assert marker not in excinfo.value.reason
    assert "integrity" not in excinfo.value.reason


async def test_gemini_provider_failure_is_stable_and_redacted(monkeypatch):
    marker = "PROVIDER_SECRET_7F3A91"

    class FailingModels:
        async def generate_content(self, **_kwargs):
            raise RuntimeError(marker)

    class FakeClient:
        aio = type("Aio", (), {"models": FailingModels()})()

    compiler = GeminiCompiler()
    monkeypatch.setattr(compiler, "_get_client", lambda: FakeClient())

    with pytest.raises(CompileError) as excinfo:
        await compiler.compile("ordinary observation")

    assert excinfo.value.reason == "provider_failure"
    assert marker not in excinfo.value.reason


async def test_gemini_empty_response_has_stable_reason(monkeypatch):
    class EmptyModels:
        async def generate_content(self, **_kwargs):
            return type("Response", (), {"text": "", "usage_metadata": None})()

    class FakeClient:
        aio = type("Aio", (), {"models": EmptyModels()})()

    compiler = GeminiCompiler()
    monkeypatch.setattr(compiler, "_get_client", lambda: FakeClient())

    with pytest.raises(CompileError) as excinfo:
        await compiler.compile("ordinary observation")

    assert excinfo.value.reason == "provider_empty_response"


async def test_gemini_validation_details_are_redacted(monkeypatch):
    marker = "MODEL_OUTPUT_SECRET_7F3A91"

    class InvalidModels:
        async def generate_content(self, **_kwargs):
            return type(
                "Response",
                (),
                {
                    "text": json.dumps({"leaked_secret": marker}),
                    "usage_metadata": None,
                },
            )()

    class FakeClient:
        aio = type("Aio", (), {"models": InvalidModels()})()

    compiler = GeminiCompiler()
    monkeypatch.setattr(compiler, "_get_client", lambda: FakeClient())

    with pytest.raises(CompileError) as excinfo:
        await compiler.compile("ordinary observation")

    assert excinfo.value.reason == "evidence_ir_schema_invalid"
    assert marker not in excinfo.value.reason
    assert "leaked_secret" not in excinfo.value.reason


async def test_validation_details_do_not_reach_response_or_ledger():
    from api.main import create_app
    from core.ledger import EventType
    from core.types import EvidenceSubmission

    marker = "SENSITIVE_MARKER_7F3A91"
    raw_text = wrap_ir(json.dumps({"leaked_secret": marker}))
    app = create_app(seed_demo=False)
    firewall = app.state.firewall
    firewall.ledger.append(
        EventType.CLAIM_REGISTERED,
        {"claim_id": "C17", "text": "Test claim", "prior": 0.5},
    )

    try:
        result = await firewall.submit(
            EvidenceSubmission(
                raw_text=raw_text,
                source_id="SRC-REDACTION",
                experiment_id="EXP-REDACTION",
                root_experiment_id="ROOT-REDACTION",
                idempotency_key="IDEM-REDACTION",
            )
        )
        event = firewall.ledger.events()[-1]

        assert result["monitor"]["reasons"] == [
            "compile_error:evidence_ir_schema_invalid"
        ]
        assert marker not in json.dumps(result, sort_keys=True)
        assert "leaked_secret" not in json.dumps(result, sort_keys=True)
        assert marker not in json.dumps(event.model_dump(mode="json"), sort_keys=True)
        assert "leaked_secret" not in json.dumps(
            event.model_dump(mode="json"), sort_keys=True
        )
    finally:
        firewall.ledger.close()


async def test_fake_compiler_output_type_is_evidence_ir_only():
    result = await FakeCompiler().compile(wrap_ir(make_evidence().model_dump_json()))
    assert type(result) is EvidenceIR


# ---------------------------------------------------------------------------
# Compiler protocol conformance (structural: async compile(str) -> EvidenceIR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compiler_cls", [FakeCompiler, GeminiCompiler])
def test_compilers_satisfy_compiler_protocol(compiler_cls):
    compile_method = getattr(compiler_cls, "compile", None)
    assert compile_method is not None
    assert inspect.iscoroutinefunction(compile_method)
    signature = inspect.signature(compile_method)
    assert list(signature.parameters) == ["self", "raw_text"]
