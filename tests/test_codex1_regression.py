"""Behavior-freeze tests for the trusted deterministic policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from groundtruth.harness import run
from groundtruth.ingest import EvidenceItem
from groundtruth.loader import (
    load_practice_seed,
    load_practice_stream,
    load_seed,
    load_stream,
)
from starter.my_solution import ingest
import starter.my_solution as solution


ROOT = Path(__file__).resolve().parents[1]


class DeterministicRegressionTests(unittest.TestCase):
    def test_practice_decisions_are_frozen(self) -> None:
        log = run(load_practice_stream(), ingest, load_practice_seed())
        self.assertEqual(
            [(record.evidence_id, record.applied_ops, record.ood_flag) for record in log.records],
            [
                ("PR01", ["no_op"], False),
                ("PR02", ["revise_confidence", "set_scope"], False),
                ("PR03", ["hold_pending"], False),
                ("PR04", ["no_op"], False),
                ("PR05", ["propose_regime"], True),
                ("PR06", ["revise_confidence", "set_scope"], False),
            ],
        )
        self.assertFalse(log.structural_violations)

    def test_predicted_stream_targets_and_posteriors_are_frozen(self) -> None:
        graph = load_seed(str(ROOT / "groundtruth" / "data" / "seed.json"))
        stream = load_stream(str(ROOT / "adversarial" / "predicted_hidden_stream.json"))
        log = run(stream, ingest, graph)
        expected_ops = {
            "H01": ["revise_confidence"],
            "H02": ["no_op"],
            "H03": ["revise_confidence", "set_scope"],
            "H04": ["revise_confidence", "set_scope"],
            "H05": ["revise_confidence", "set_scope", "revise_confidence"],
            "H06": ["hold_pending"],
            "H07": ["no_op"],
            "H08": ["no_op"],
            "H09": ["no_op"],
            "H10": ["no_op"],
            "H11": ["propose_regime"],
            "H12": ["propose_axis"],
            "H13": ["revise_confidence", "set_scope"],
            "H14": ["revise_confidence"],
            "H15": ["revise_confidence", "revise_confidence"],
            "H16": ["no_op"],
            "H17": ["revise_confidence", "set_scope"],
            "H18": ["revise_confidence"],
            "H19": ["no_op"],
            "H20": ["no_op"],
        }
        self.assertEqual(
            {record.evidence_id: record.applied_ops for record in log.records},
            expected_ops,
        )
        self.assertEqual(
            {record.evidence_id for record in log.records if record.ood_flag},
            {"H11", "H12"},
        )
        self.assertEqual(
            {claim_id: round(graph.claims[claim_id].confidence, 6) for claim_id in (
                "C1", "C3a", "C3b", "C3c", "C3d", "C4", "C5", "C6"
            )},
            {
                "C1": 0.802413,
                "C3a": 0.85715,
                "C3b": 0.371036,
                "C3c": 0.429803,
                "C3d": 0.429803,
                "C4": 0.767737,
                "C5": 0.997102,
                "C6": 0.981244,
            },
        )
        self.assertFalse(log.structural_violations)

    def test_solution_loads_by_direct_file_path(self) -> None:
        path = ROOT / "starter" / "my_solution.py"
        name = "codex1_direct_path_solution"
        spec = importlib.util.spec_from_file_location(name, path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader if spec is not None else None)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules.pop(name, None)
        spec.loader.exec_module(module)
        result = module.ingest(load_practice_stream()[0], load_practice_seed_view())
        self.assertEqual([delta.op for delta in result.deltas], ["no_op"])

    def test_body_is_assessed_once_and_provenance_rederived_by_the_monitor(self) -> None:
        # The untrusted body must be parsed exactly once (this is what keeps the
        # optional canary from observing two different witnesses). The trusted
        # structured provenance, by contrast, is deterministically re-derived a
        # second time by the Phase-1 authorization monitor as an independent
        # authorization record -- that re-derivation is safe and intentional.
        item = load_practice_stream()[1]
        view = load_practice_seed_view()
        original_assess = solution._assess_body
        original_provenance = solution._provenance
        with patch.object(
            solution, "_assess_body", wraps=original_assess
        ) as assess_mock, patch.object(
            solution, "_provenance", wraps=original_provenance
        ) as provenance_mock:
            result = solution.ingest(item, view)
        self.assertIn("revise_confidence", [delta.op for delta in result.deltas])
        assess_mock.assert_called_once()
        self.assertEqual(provenance_mock.call_count, 2)  # decide + authorize

    def test_weak_independent_origins_accumulate_then_clear_exact_pending_family(self) -> None:
        provenance = {
            "replication_count": 1,
            "independent_groups": 1,
            "method_class": "environmental_stress",
            "method_directness": "direct",
            "effect_strength": "strong",
            "retraction_status": "none",
        }
        body = (
            "An environmental stress protocol returned Fibroblasts to a "
            "pluripotent-like state."
        )
        graph = load_seed()
        log = run(
            [EvidenceItem(f"A{index}", "", body, provenance) for index in range(1, 5)],
            ingest,
            graph,
        )
        self.assertEqual(
            [record.applied_ops for record in log.records],
            [
                ["hold_pending"],
                ["hold_pending"],
                ["hold_pending"],
                [
                    "revise_confidence",
                    "set_scope",
                    "drop_claim",
                    "drop_claim",
                    "drop_claim",
                ],
            ],
        )
        self.assertEqual(graph.pending, {})
        self.assertEqual(round(graph.claims["C3d"].confidence, 6), 0.429803)

    def test_retraction_and_failed_replication_drop_only_exact_pending_family(self) -> None:
        weak = {
            "replication_count": 1,
            "independent_groups": 1,
            "method_class": "environmental_stress",
            "method_directness": "direct",
            "effect_strength": "strong",
            "retraction_status": "none",
        }
        body = (
            "An environmental stress protocol returned Fibroblasts to a "
            "pluripotent-like state."
        )
        cases = (
            (
                body,
                {**weak, "retraction_status": "retracted"},
            ),
            (
                "Independent groups failed to replicate that Fibroblasts returned "
                "to a pluripotent-like state.",
                {
                    **weak,
                    "replication_count": 3,
                    "independent_groups": 3,
                    "effect_strength": "weak",
                },
            ),
        )
        for invalidating_body, invalidating_provenance in cases:
            with self.subTest(body=invalidating_body):
                graph = load_seed()
                log = run(
                    [
                        EvidenceItem("W1", "", body, weak),
                        EvidenceItem(
                            "I1", "", invalidating_body, invalidating_provenance
                        ),
                    ],
                    ingest,
                    graph,
                )
                self.assertEqual(
                    [record.applied_ops for record in log.records],
                    [["hold_pending"], ["drop_claim"]],
                )
                self.assertEqual(graph.pending, {})
                self.assertEqual(graph.claims["C3d"].confidence, 0.92)


def load_practice_seed_view():
    from groundtruth.model import GraphView

    return GraphView(load_practice_seed())


if __name__ == "__main__":
    unittest.main()
