# The Evidence Firewall

The Evidence Firewall is a provider-optional demo of bounded authority for
LLM-mediated evidence ingestion. Untrusted prose is compiled into a strict IR,
checked against witnessed spans and trusted provenance, shadow-executed, and
only then allowed to make a KL-bounded, root-budgeted, append-only update.

The security claim is intentionally narrow: the system does not promise to
detect every malicious sentence. It prevents raw or merely parsed text from
owning deterministic security decisions, limits authorized influence, and can
reverse every committed delta associated with a retracted root.

## Run locally

Backend (Python 3.12+ and `uv`):

```powershell
cd epistemicos
uv sync --locked
$env:EPISTEMICOS_SEED_DEMO="1"
$env:EPISTEMICOS_DB_PATH="demo.db"
uv run uvicorn api.main:app --reload --port 8000
```

Frontend (Node 20+):

```powershell
cd epistemicos/ui
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite development server proxies API and SSE
requests to `http://127.0.0.1:8000`.

The shipped app defaults to deterministic fake providers, a persistent SQLite
ledger, and demo seeding disabled. The two environment variables above opt in
to trusted demo fixtures. No API key or network call is required. Set
`EPISTEMICOS_PROVIDER_MODE=gemini` with `GEMINI_API_KEY` to select the
separately configured live-provider path; fake mode remains the safe default.

This MVP intentionally has no auth system. Keep it bound to the default
loopback interface; do not expose its ingestion or retraction routes publicly.

## Verify

```powershell
cd epistemicos
uv run pytest -q
uv run --with pytest-cov pytest -q --cov=core --cov=llm --cov=api --cov=demo --cov=redteam --cov-report=term-missing

cd ui
npm run lint
npm run build
```

See [the demo runbook](epistemicos/DEMO_RUNBOOK.md), [test report](epistemicos/TEST_REPORT.md),
and [red-team report](epistemicos/REDTEAM_REPORT.md) for the validated release snapshot.
