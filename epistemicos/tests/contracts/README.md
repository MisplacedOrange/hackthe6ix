# Cross-goal contract tests

Tests in this directory exercise the seams **between** goals (the §9
interfaces in `core/contracts.py`) rather than a single module. A change is
only safe when every contract test passes.

Conventions:

- Contract tests are provider-free: they must pass without `GEMINI_API_KEY`
  and without network access. Use the fake compiler/embedder implementations.
- Name files `test_contract_<seam>.py`, e.g. `test_contract_pipeline.py`.
- Shared fixtures live in `tests/conftest.py`. Fixture JSON (red-team corpus,
  seeded scenarios) lives in `tests/fixtures/` and must be seeded /
  deterministic.
- If a contract test fails after an interface change, fix the interface
  consumer(s), not the test, unless §9 of EVIDENCE_FIREWALL_V2.md was
  deliberately updated first.
