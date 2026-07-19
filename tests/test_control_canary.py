"""Release gate: the offline control-canary check must report zero drift.

Run as a subprocess so the harness's offline-forcing environment (GT_LLM_MODE
= off, no key) cannot leak into the rest of the suite -- in particular the
opt-in live-canary tests, which key off GT_LLM_MODE.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "adversarial" / "control_canary.py"


def test_control_canary_reports_zero_unexpected_mutations() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True, text=True, timeout=120,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"control canary failed the release gate:\n{output}"
    assert "unexpected mutations: 0" in output, output
    assert "repeatability failures: 0" in output, output
