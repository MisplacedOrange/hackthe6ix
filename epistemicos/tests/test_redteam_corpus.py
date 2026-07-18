"""Provider-free contracts for the deterministic G10 red-team corpus."""

from __future__ import annotations

import asyncio
import copy
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.engine import ROOT_BUDGET
from llm.compiler import CompileError, FakeCompiler
from redteam.generate import (
    CATEGORIES,
    CORPUS_PATH,
    CorpusValidationError,
    generate_corpus,
    load_corpus,
    serialize_corpus,
    validate_corpus,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STABLE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def test_seeded_regeneration_is_byte_for_byte_stable() -> None:
    generated = serialize_corpus(generate_corpus())
    checked_in = CORPUS_PATH.read_text(encoding="utf-8")
    assert generated == checked_in


def test_all_required_categories_and_stable_unique_ids_are_present() -> None:
    cases = load_corpus()["cases"]
    assert {case["category"] for case in cases} == CATEGORIES

    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    assert all(STABLE_ID.fullmatch(case_id) for case_id in ids)


def test_corpus_explicitly_disclaims_perfect_detection() -> None:
    corpus = load_corpus()
    assert "not perfect injection detection" in corpus["disclaimer"]
    assert all(
        case["expected_containment"]["detection_required"] is False
        for case in corpus["cases"]
    )


def test_all_structured_cases_pass_the_strict_fake_compiler() -> None:
    async def compile_all() -> None:
        compiler = FakeCompiler()
        structured = [
            case for case in load_corpus()["cases"] if "<<IR>>" in case["raw_text"]
        ]
        assert structured
        for case in structured:
            result = await compiler.compile(case["raw_text"])
            assert result.experiment_id == case["submission"]["experiment_id"]
            assert result.root_experiment_id == case["submission"]["root_experiment_id"]

    asyncio.run(compile_all())


def test_raw_t1_cases_fail_closed_under_fake_compiler() -> None:
    async def compile_all() -> None:
        compiler = FakeCompiler()
        t1_cases = [
            case
            for case in load_corpus()["cases"]
            if case["expected_containment"]["threat_tier"] == "T1"
        ]
        assert t1_cases
        for case in t1_cases:
            assert case["expected_containment"]["containment"] == (
                "ZERO_COMMITTED_INFLUENCE"
            )
            assert "COMMIT" not in case["expected_containment"]["allowed_verdicts"]
            with pytest.raises(CompileError):
                await compiler.compile(case["raw_text"])

    asyncio.run(compile_all())


def test_duplicate_flood_shares_one_root_and_declares_engine_budget() -> None:
    duplicates = [
        case for case in load_corpus()["cases"] if case["category"] == "duplicate_flood"
    ]
    roots = {case["submission"]["root_experiment_id"] for case in duplicates}
    assert roots == {"ROOT-DUPLICATE-SHARED"}

    limits = [case["expected_containment"]["limits"] for case in duplicates]
    assert [limit["sequence_index"] for limit in limits] == list(
        range(1, len(duplicates) + 1)
    )
    assert all(limit["sequence_length"] == len(duplicates) for limit in limits)
    assert all(limit["max_cumulative_root_kl"] == ROOT_BUDGET for limit in limits)


def test_slow_drip_is_an_ordered_same_source_sequence() -> None:
    cases = [case for case in load_corpus()["cases"] if case["category"] == "slow_drip"]
    limits = [case["expected_containment"]["limits"] for case in cases]

    assert len(cases) >= 2
    assert {case["submission"]["source_id"] for case in cases} == {
        "source-repeat-offender"
    }
    assert len({case["submission"]["root_experiment_id"] for case in cases}) == len(cases)
    assert [limit["sequence_index"] for limit in limits] == list(range(1, len(cases) + 1))
    assert all(limit["sequence_id"] == "slow-drip" for limit in limits)
    assert all(limit["sequence_length"] == len(cases) for limit in limits)


@pytest.mark.parametrize("mutation", ["duplicate_id", "permit_t1_commit", "drop_category"])
def test_invalid_corpus_is_rejected(mutation: str) -> None:
    invalid = copy.deepcopy(load_corpus())
    if mutation == "duplicate_id":
        invalid["cases"][1]["id"] = invalid["cases"][0]["id"]
    elif mutation == "permit_t1_commit":
        invalid["cases"][0]["expected_containment"]["allowed_verdicts"].append("COMMIT")
    elif mutation == "drop_category":
        invalid["cases"] = [
            case for case in invalid["cases"] if case["category"] != "base64"
        ]
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    with pytest.raises(CorpusValidationError):
        validate_corpus(invalid)


def test_cli_check_accepts_checked_in_corpus() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "redteam.generate", "--check"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "validated 27 cases" in completed.stdout


def test_cli_check_rejects_valid_but_drifted_corpus(tmp_path: Path) -> None:
    drifted = copy.deepcopy(load_corpus())
    drifted["cases"][0]["raw_text"] += " Drifted fixture."
    drifted_path = tmp_path / "corpus.json"
    drifted_path.write_text(serialize_corpus(drifted), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "redteam.generate",
            "--check",
            "--output",
            str(drifted_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "differs from seed" in completed.stderr
