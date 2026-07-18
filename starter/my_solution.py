"""Submission entrypoint for the semantic GROUND TRUTH policy.

The normal import works in the repository.  The local-file fallback also works
when a scorer loads this module by path from another working directory, provided
``semantic_policy.py`` is submitted beside it.
"""
from __future__ import annotations

try:
    from starter.semantic_policy import ingest
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path
    import sys

    policy_path = Path(__file__).with_name("semantic_policy.py")
    repository_root = str(policy_path.parent.parent)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    module_name = "ground_truth_submission_semantic_policy"
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load semantic policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    ingest = module.ingest

__all__ = ["ingest"]
