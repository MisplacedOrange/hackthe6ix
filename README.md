# Paladin

Paladin is a provenance-aware scientific belief-revision engine. It processes evidence streams, evaluates the strength and direction of each result, and updates a structured knowledge graph only through validated deltas.

The system is designed around a strict firewall: incoming text is untrusted, while structured provenance controls evidence weight and the graph remains read-only to the policy. Paladin also detects out-of-distribution evidence, tracks weak results as pending, and rejects control-plane instructions embedded in reports.

## Gemini integration

Paladin optionally uses Gemini through the `google-genai` SDK as a sacrificial canary for malformed or ambiguous evidence. Gemini receives only the report body and returns a closed JSON verdict (`benign`, `injection`, or `abstain`) with supporting quotes. The response is grounded against the original text, and Gemini cannot create deltas, choose confidence values, access graph state, or bypass the deterministic policy.

Without `GEMINI_API_KEY`, the optional SDK, or a valid response, Paladin fails closed and follows the deterministic path unchanged.

## Running locally

```bash
python -m py_compile starter/my_solution.py
python -m pytest -q
```

The Gemini canary is disabled by default. To enable it locally, install the optional dependency and configure `GEMINI_API_KEY` in `starter/.env`:

```bash
pip install "google-genai>=1.0"
```

# Disclaimer
This project was made for the CORTEX Biosciences track, see [challenge.md](challenge.md) for the original challenge specification and [DESIGN.md](DESIGN.md) for the policy and safety guarantees.
