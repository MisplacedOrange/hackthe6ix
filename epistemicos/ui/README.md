# Evidence Firewall UI

React observability view for the deterministic Evidence Firewall. It connects to
`/stream` through one `EventSource` hook and falls back to the complete seeded
nine-event replay when the API is unavailable.

```bash
npm install
npm run dev
```

Set `VITE_API_BASE_URL` when the API is served from another origin. No value is
needed when Vite and the API share an origin.

The live view replaces the fallback ledger as soon as the first evidence event
arrives. Displayed confidence is reduced only from applying ledger
`payload.delta` values (`EVIDENCE_COMMITTED`, `ESCROW_RELEASED`, and
`REVERSAL`); shadow engine proposals remain observability-only. Aggregate
security and reversal figures come from `/metrics` when connected.

## G11 stage-demo checklist

- Confirm the header reports **Live ledger** with the API running, or **Seeded fallback** offline.
- Click **Replay all** and narrate all nine numbered events in sequence.
- At event 3, show the schema-valid semantic attack remains provisional with zero committed influence.
- At event 5, select the escrow row and show its high-shock monitor reason.
- At event 6, show independent replication earning L3 and moving confidence.
- At event 7, show the same-root derivative clipped by the shared root budget.
- At event 9, point to the red timeline segment and the exact −15 pp root reversal.
- Select several ledger rows to show reasons, bounded engine arithmetic, latency, calls, and token metrics.
- Verify an L0 event displays the locked explainer message and never renders or requests raw input.
- Finish on **0 unauthorized transitions** and **100% reversal completeness**.
