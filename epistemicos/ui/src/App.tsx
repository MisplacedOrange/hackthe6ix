import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { CLAIM, DEMO_EVENTS } from './demo'
import type {
  ClaimState,
  EventKind,
  Integrity,
  LedgerEvent,
  TimelinePoint,
  Verdict,
} from './types'
import { useEventStream, type StreamMessage } from './useEventStream'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''
const STATE_ORDER: ClaimState[] = ['SUPPORTED', 'CONTRADICTED', 'CONTESTED', 'UNKNOWN']

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {}
}

function asNumber(value: unknown, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function asOptionalNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function integrityLabel(value: unknown): Integrity {
  if (typeof value === 'number') {
    return ['L0 · RAW', 'L1 · PARSED', 'L2 · VERIFIED', 'L3 · REPLICATED'][value] as Integrity
  }
  const normalized = String(value ?? '').toUpperCase()
  if (normalized.includes('L3') || normalized.includes('REPLICATED')) return 'L3 · REPLICATED'
  if (normalized.includes('L2') || normalized.includes('VERIFIED')) return 'L2 · VERIFIED'
  if (normalized.includes('L1') || normalized.includes('PARSED')) return 'L1 · PARSED'
  return 'L0 · RAW'
}

function eventKind(value: unknown): EventKind | null {
  const normalized = String(value ?? '').toUpperCase()
  const known: EventKind[] = [
    'EVIDENCE_COMMITTED',
    'EVIDENCE_PROVISIONAL',
    'EVIDENCE_ESCROWED',
    'EVIDENCE_REJECTED',
    'ESCROW_RELEASED',
    'ESCROW_REJECTED',
    'RETRACTION',
    'REVERSAL',
  ]
  return known.find((kind) => kind === normalized) ?? null
}

function verdictFor(kind: EventKind): Verdict {
  if (kind === 'EVIDENCE_PROVISIONAL') return 'PROVISIONAL'
  if (kind === 'EVIDENCE_ESCROWED') return 'ESCROW'
  if (kind === 'EVIDENCE_REJECTED') return 'REJECT'
  if (kind === 'ESCROW_RELEASED') return 'RELEASE'
  if (kind === 'ESCROW_REJECTED') return 'ESCROW_REJECTED'
  if (kind === 'RETRACTION') return 'RETRACTION'
  if (kind === 'REVERSAL') return 'REVERSAL'
  return 'COMMIT'
}

function appliesToConfidence(kind: EventKind) {
  return kind === 'EVIDENCE_COMMITTED' || kind === 'ESCROW_RELEASED' || kind === 'REVERSAL'
}

/**
 * Converts an SSE ledger envelope to the UI's metadata-only shape. Notice
 * that raw_text/raw input is never copied, even if a server payload has it.
 */
function sanitizeStreamEvent(message: StreamMessage): LedgerEvent | null {
  const seq = asNumber(message.seq, 0)
  if (seq < 1) return null
  if (message.event_type === 'CLAIM_REGISTERED') return null
  const payload = asRecord(message.payload)
  const monitor = asRecord(payload.monitor ?? message.monitor)
  const engineData = asRecord(payload.engine ?? message.engine)
  const metricData = asRecord(payload.metrics ?? message.metrics)
  const kind = eventKind(message.event_type ?? message.type)
  if (!kind) return null
  const integrity = integrityLabel(monitor.integrity ?? payload.integrity)
  // Only the append-only ledger's committed payload delta can move displayed
  // belief. A shadow engine proposal may be non-zero while escrowed/provisional.
  const delta = appliesToConfidence(kind) ? asNumber(payload.delta, 0) : 0
  const enginePrior = asNumber(engineData.prior, CLAIM.prior)
  const boundedDelta = asNumber(engineData.bounded_delta, 0)
  const candidatePosterior = asNumber(engineData.posterior, Math.max(0, Math.min(1, enginePrior + boundedDelta)))
  const reasonsValue = monitor.reasons ?? payload.reasons
  const reasons = Array.isArray(reasonsValue)
    ? reasonsValue.filter((reason): reason is string => typeof reason === 'string')
    : ['ledger_event_received']

  return {
    seq,
    stepId: String(payload.step_id ?? `live-event-${seq}`),
    title: String(payload.title ?? String(message.event_type ?? 'Ledger event').replaceAll('_', ' ')),
    eventType: kind,
    verdict: verdictFor(kind),
    integrity,
    delta,
    rootId: String(payload.root_experiment_id ?? 'UNMODELED'),
    relation: typeof payload.relation === 'string' ? payload.relation : null,
    reversesSeqs: typeof payload.reverses_seq === 'number' ? [payload.reverses_seq] : [],
    reasons,
    tags: ['live ledger'],
    engine: Object.keys(engineData).length
      ? {
          prior: enginePrior,
          rawBf: asNumber(engineData.raw_bf, 0),
          boundedDelta,
          rootSpent: asNumber(engineData.root_spent, 0),
          posterior: candidatePosterior,
          integrity,
        }
      : null,
    metrics: {
      latencyMs: asNumber(metricData.latency_ms, 0),
      geminiCalls: asNumber(metricData.gemini_calls, 0),
      escalated: Boolean(metricData.escalated),
      inputTokens: asNumber(metricData.input_tokens, 0),
      outputTokens: asNumber(metricData.output_tokens, 0),
      thinkingTokens: asNumber(metricData.thinking_tokens, 0),
      cachedTokens: asNumber(metricData.cached_tokens, 0),
    },
    explanation: 'Deterministic ledger metadata received. Select the event to inspect its monitor and bounded-update trace.',
  }
}

function reduceLedger(events: LedgerEvent[]) {
  let confidence: number = CLAIM.prior
  const activeEvidence = new Map<number, number>()
  const timeline: Array<TimelinePoint & { reversalConfidence?: number }> = [
    { seq: 0, label: 'Prior', confidence, delta: 0, kind: 'BASELINE' },
  ]

  for (const event of [...events].sort((left, right) => left.seq - right.seq)) {
    if (event.eventType === 'EVIDENCE_COMMITTED' || event.eventType === 'ESCROW_RELEASED') {
      confidence = Math.max(0, Math.min(1, confidence + event.delta))
      if (event.delta !== 0) {
        const direction = event.relation === 'contradicts' ? -1 : event.delta > 0 ? 1 : -1
        activeEvidence.set(event.seq, direction)
      }
    } else if (event.eventType === 'REVERSAL') {
      confidence = Math.max(0, Math.min(1, confidence + event.delta))
      for (const reversedSeq of event.reversesSeqs) activeEvidence.delete(reversedSeq)
    }

    timeline.push({
      seq: event.seq,
      label: `E${event.seq}`,
      confidence,
      delta: event.delta,
      kind: event.eventType,
    })
  }

  for (let index = 1; index < timeline.length; index += 1) {
    if (timeline[index].kind !== 'REVERSAL') continue
    timeline[index - 1].reversalConfidence = timeline[index - 1].confidence
    timeline[index].reversalConfidence = timeline[index].confidence
  }

  const activeDirections = [...activeEvidence.values()]
  const hasPositive = activeDirections.some((direction) => direction > 0)
  const hasNegative = activeDirections.some((direction) => direction < 0)
  let state: ClaimState = 'UNKNOWN'
  if (hasPositive && hasNegative) state = 'CONTESTED'
  else if (hasPositive) state = 'SUPPORTED'
  else if (hasNegative) state = 'CONTRADICTED'
  return { confidence, state, timeline }
}

interface BackendMetrics {
  unauthorizedTransitions: number | null
  reversalPercentage: number
  reversalReversed: number
  reversalCommitted: number
  escrowReleased: number
  escrowRejected: number
  p50: number
  p95: number
  escalationRate: number
  callsPerEvent: number
  tokens: number
  cacheFraction: number
}

function sanitizeBackendMetrics(value: unknown): BackendMetrics {
  const body = asRecord(value)
  const security = asRecord(body.security)
  const reversal = asRecord(body.reversal_completeness)
  const escrow = asRecord(body.escrow_outcomes)
  const latency = asRecord(body.fast_lane_latency_ms)
  const tokens = asRecord(body.token_totals)
  return {
    unauthorizedTransitions: asOptionalNumber(body.unauthorized_transition_count ?? security.unauthorized_transition_count),
    reversalPercentage: asNumber(reversal.percentage, 0),
    reversalReversed: asNumber(reversal.reversed_count, 0),
    reversalCommitted: asNumber(reversal.committed_count, 0),
    escrowReleased: asNumber(escrow.release, 0),
    escrowRejected: asNumber(escrow.reject, 0),
    p50: asNumber(body.p50_latency_ms ?? latency.p50, 0),
    p95: asNumber(body.p95_latency_ms ?? latency.p95, 0),
    escalationRate: asNumber(body.escalation_percentage, 0) / 100,
    callsPerEvent: asNumber(body.average_gemini_calls_per_event, 0),
    tokens: asNumber(tokens.total_tokens, 0),
    cacheFraction: asNumber(body.cache_hit_token_fraction, 0),
  }
}

function localReversalCoverage(events: LedgerEvent[]) {
  const reversedRoots = new Set(
    events
      .filter((event) => event.eventType === 'RETRACTION' || event.eventType === 'REVERSAL')
      .map((event) => event.rootId),
  )
  if (reversedRoots.size === 0) return null
  const expected = new Set(
    events
      .filter((event) =>
        (event.eventType === 'EVIDENCE_COMMITTED' || event.eventType === 'ESCROW_RELEASED')
        && reversedRoots.has(event.rootId),
      )
      .map((event) => event.seq),
  )
  const reversed = new Set(events.flatMap((event) => event.reversesSeqs))
  const matched = [...expected].filter((seq) => reversed.has(seq)).length
  return expected.size ? matched / expected.size : null
}

function formatPercent(value: number, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`
}

function formatReason(reason: string) {
  return reason.replaceAll('_', ' ').replace(':', ' · ')
}

function formatVerdict(verdict: Verdict) {
  return verdict.replaceAll('_', ' ')
}

function App() {
  const { status, message } = useEventStream(`${API_BASE}/stream`)
  const [liveEvents, setLiveEvents] = useState<LedgerEvent[]>([])
  const [visibleCount, setVisibleCount] = useState(DEMO_EVENTS.length)
  const [selectedSeq, setSelectedSeq] = useState<number>(DEMO_EVENTS.length)
  const [playing, setPlaying] = useState(false)
  const [remoteExplanation, setRemoteExplanation] = useState<string | null>(null)
  const [backendMetrics, setBackendMetrics] = useState<BackendMetrics | null>(null)
  const [chainValid, setChainValid] = useState<boolean | null>(null)

  const dataMode = liveEvents.length > 0 ? 'live' : 'demo'
  const events = dataMode === 'live' ? liveEvents : DEMO_EVENTS

  useEffect(() => {
    if (!message) return
    const incoming = sanitizeStreamEvent(message)
    if (!incoming) return
    setPlaying(false)
    setSelectedSeq(incoming.seq)
    setLiveEvents((current) => {
      const withoutSeq = current.filter((event) => event.seq !== incoming.seq)
      return [...withoutSeq, incoming].sort((a, b) => a.seq - b.seq)
    })
  }, [message])

  useEffect(() => {
    if (status !== 'live') return
    const controller = new AbortController()
    void fetch(`${API_BASE}/metrics`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('metrics unavailable')
        return response.json() as Promise<unknown>
      })
      .then((metricsBody) => setBackendMetrics(sanitizeBackendMetrics(metricsBody)))
      .catch(() => undefined)
    void fetch(`${API_BASE}/claims`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('claims unavailable')
        return response.json() as Promise<unknown>
      })
      .then((claimsBody) => {
        const claims = asRecord(claimsBody)
        setChainValid(typeof claims.chain_valid === 'boolean' ? claims.chain_valid : null)
      })
      .catch(() => undefined)
    return () => controller.abort()
  }, [message, status])

  useEffect(() => {
    if (!playing || dataMode !== 'demo') return
    if (visibleCount >= DEMO_EVENTS.length) {
      setPlaying(false)
      return
    }
    const timer = window.setTimeout(() => {
      const nextCount = visibleCount + 1
      setVisibleCount(nextCount)
      setSelectedSeq(DEMO_EVENTS[nextCount - 1]?.seq ?? 0)
    }, 720)
    return () => window.clearTimeout(timer)
  }, [dataMode, playing, visibleCount])

  const visibleEvents = dataMode === 'live' ? events : events.slice(0, visibleCount)
  const selected = visibleEvents.find((event) => event.seq === selectedSeq) ?? visibleEvents.at(-1)
  const reduced = useMemo(() => reduceLedger(visibleEvents), [visibleEvents])
  const { confidence, state, timeline } = reduced

  useEffect(() => {
    setRemoteExplanation(null)
    if (dataMode !== 'live' || !selected || selected.integrity === 'L0 · RAW') return
    const controller = new AbortController()
    void fetch(`${API_BASE}/explain/${selected.seq}`, { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error('explain unavailable'))))
      .then((body: unknown) => {
        const safeBody = asRecord(body)
        const explanation = safeBody.explanation ?? safeBody.summary
        if (typeof explanation === 'string') setRemoteExplanation(explanation)
      })
      .catch(() => undefined)
    return () => controller.abort()
  }, [dataMode, selected])

  const aggregate = useMemo(() => {
    const latencies = visibleEvents.map((event) => event.metrics.latencyMs).sort((a, b) => a - b)
    const percentile = (p: number) => latencies[Math.max(0, Math.ceil(latencies.length * p) - 1)] ?? 0
    const totalTokens = visibleEvents.reduce(
      (sum, event) => sum + event.metrics.inputTokens + event.metrics.outputTokens + event.metrics.thinkingTokens,
      0,
    )
    const cachedTokens = visibleEvents.reduce((sum, event) => sum + event.metrics.cachedTokens, 0)
    const totalCalls = visibleEvents.reduce((sum, event) => sum + event.metrics.geminiCalls, 0)
    const escalated = visibleEvents.filter((event) => event.metrics.escalated).length
    return {
      p50: percentile(0.5),
      p95: percentile(0.95),
      callsPerEvent: visibleEvents.length ? totalCalls / visibleEvents.length : 0,
      escalationRate: visibleEvents.length ? escalated / visibleEvents.length : 0,
      cacheFraction: totalTokens ? cachedTokens / totalTokens : 0,
      tokens: totalTokens,
    }
  }, [visibleEvents])

  const localCoverage = useMemo(() => localReversalCoverage(visibleEvents), [visibleEvents])
  const telemetry = dataMode === 'live' && backendMetrics ? backendMetrics : aggregate
  const reversalDelta = visibleEvents
    .filter((event) => event.eventType === 'REVERSAL')
    .reduce((sum, event) => sum + event.delta, 0)

  function replay() {
    if (dataMode !== 'demo') return
    setVisibleCount(0)
    setSelectedSeq(0)
    setPlaying(true)
  }

  function stepForward() {
    if (dataMode !== 'demo') return
    setPlaying(false)
    const next = Math.min(DEMO_EVENTS.length, visibleCount + 1)
    setVisibleCount(next)
    setSelectedSeq(DEMO_EVENTS[next - 1]?.seq ?? 0)
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="#top" aria-label="Epistemic OS observatory home">
          <span className="wordmark-mark" aria-hidden="true">EF</span>
          <span>Epistemic OS</span>
          <span className="wordmark-edition">Evidence Firewall / v2</span>
        </a>
        <div className="system-status" aria-live="polite">
          <span className={`status-light status-${status}`} />
          <span>{dataMode === 'live'
            ? status === 'live' ? 'Live ledger' : 'Live snapshot · reconnecting'
            : status === 'live' ? 'Connected · awaiting ledger' : status === 'connecting' ? 'Probing API' : 'Seeded fallback'}</span>
          <span className="hash-label">{dataMode === 'live'
            ? chainValid === true ? 'chain verified' : chainValid === false ? 'chain invalid' : 'chain checking'
            : 'seeded scenario'}</span>
        </div>
      </header>

      <section id="top" className="briefing">
        <div className="briefing-copy">
          <p className="eyebrow">Deterministic belief control room</p>
          <h1>Evidence may earn influence.<br /><em>Text never earns authority.</em></h1>
        </div>
        <div className="briefing-invariant">
          <span>North-star invariant</span>
          <p>All input arrives at L0. Every mutation is witnessed, budgeted, append-only and reversible.</p>
        </div>
      </section>

      <section className="overview-grid" aria-label="Claim overview">
        <article className="claim-card panel">
          <div className="panel-kicker">
            <span>Primary claim</span>
            <span>{CLAIM.id} / oncology</span>
          </div>
          <div className="claim-main">
            <div>
              <h2>{CLAIM.text}</h2>
              <p className="claim-meta">Seed prior {formatPercent(CLAIM.prior)} · {visibleEvents.length} {dataMode === 'live' ? 'ledger' : 'replay'} events observed</p>
            </div>
            <div className="confidence-orbit" aria-label={`Current confidence ${formatPercent(confidence)}`}>
              <span>{formatPercent(confidence)}</span>
              <small>confidence</small>
            </div>
          </div>
          <div className="state-row" aria-label={`Current claim state ${state}`}>
            {STATE_ORDER.map((item) => (
              <span key={item} className={`state-badge state-${item.toLowerCase()} ${item === state ? 'is-current' : ''}`}>
                {item}
              </span>
            ))}
          </div>
          <div className="confidence-meter" aria-hidden="true">
            <span style={{ width: `${confidence * 100}%` }} />
            <i style={{ left: `${CLAIM.prior * 100}%` }} />
          </div>
          <div className="claim-foot">
            <span>prior marker {formatPercent(CLAIM.prior)}</span>
            <span>net movement {confidence - CLAIM.prior >= 0 ? '+' : ''}{formatPercent(confidence - CLAIM.prior, 1)}</span>
          </div>
        </article>

        <aside className="lattice-card panel">
          <div className="panel-kicker"><span>Integrity lattice</span><span>earned only</span></div>
          <ol className="lattice-list">
            <li><b>L0</b><span><strong>Raw</strong><small>inert provenance</small></span><i>ε = 0</i></li>
            <li><b>L1</b><span><strong>Parsed</strong><small>provisional only</small></span><i>schema</i></li>
            <li><b>L2</b><span><strong>Verified</strong><small>bounded update</small></span><i>witness</i></li>
            <li><b>L3</b><span><strong>Replicated</strong><small>full weighting</small></span><i>lineage</i></li>
          </ol>
        </aside>
      </section>

      <section className="replay-panel panel" aria-label={dataMode === 'demo' ? 'Nine-event deterministic replay' : 'Live deterministic ledger timeline'}>
        <div className="section-heading">
          <div>
            <p className="eyebrow">{dataMode === 'demo' ? 'Seeded scenario / 09 events' : `Live ledger / ${events.length} events`}</p>
            <h2>{dataMode === 'demo' ? 'Confidence replay' : 'Committed confidence'}</h2>
          </div>
          {dataMode === 'demo' ? (
            <div className="replay-controls">
              <button type="button" onClick={replay}>{playing ? 'Restart replay' : 'Replay all'}</button>
              <button type="button" className="button-secondary" onClick={stepForward} disabled={visibleCount >= events.length}>Step +1</button>
              <span>{String(visibleCount).padStart(2, '0')} / {String(events.length).padStart(2, '0')}</span>
            </div>
          ) : <span className="chain-ok"><i /> payload deltas only</span>}
        </div>

        <div className="timeline-wrap">
          <ResponsiveContainer width="100%" height={290}>
            <LineChart data={timeline} margin={{ top: 22, right: 20, left: -10, bottom: 4 }}>
              <CartesianGrid stroke="rgba(179, 194, 184, 0.12)" vertical={false} />
              <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: '#84948a', fontSize: 11 }} />
              <YAxis domain={[0.3, 0.75]} ticks={[0.35, 0.45, 0.55, 0.65, 0.75]} axisLine={false} tickLine={false} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fill: '#84948a', fontSize: 11 }} />
              <Tooltip
                cursor={{ stroke: '#a5ffcc', strokeDasharray: '2 4' }}
                contentStyle={{ background: '#101914', border: '1px solid #345044', borderRadius: 0, fontSize: 12 }}
                labelStyle={{ color: '#dbe9df', marginBottom: 5 }}
                formatter={(value) => [formatPercent(Number(value)), 'confidence']}
              />
              <ReferenceLine y={CLAIM.prior} stroke="#7f8b84" strokeDasharray="4 5" label={{ value: 'PRIOR', fill: '#84948a', fontSize: 10, position: 'insideTopRight' }} />
              <Line dataKey="confidence" type="stepAfter" stroke="#a5ffcc" strokeWidth={2.5} dot={{ r: 3, fill: '#0c1510', stroke: '#a5ffcc', strokeWidth: 2 }} activeDot={{ r: 5 }} isAnimationActive={false} />
              <Line dataKey="reversalConfidence" type="linear" connectNulls={false} stroke="#ff735c" strokeWidth={4} dot={{ r: 4, fill: '#ff735c', strokeWidth: 0 }} isAnimationActive />
            </LineChart>
          </ResponsiveContainer>
          {reversalDelta !== 0 && (
            <div className="retraction-callout">
              <span>Reversal</span>
              <strong>{reversalDelta > 0 ? '+' : '−'}{Math.abs(reversalDelta * 100).toFixed(1)} pp</strong>
              <small>committed payload delta</small>
            </div>
          )}
        </div>

        <div className="event-rail" role="list" aria-label="Replay steps">
          {events.map((event, index) => {
            const visible = index < visibleCount
            return (
              <button
                type="button"
                role="listitem"
                key={event.stepId}
                className={`rail-step verdict-${event.verdict.toLowerCase()} ${visible ? 'is-visible' : ''} ${selected?.seq === event.seq ? 'is-selected' : ''}`}
                onClick={() => {
                  if (!visible) return
                  setSelectedSeq(event.seq)
                  setPlaying(false)
                }}
                disabled={!visible}
                aria-label={`Event ${event.seq}: ${event.title}`}
              >
                <span>{String(event.seq).padStart(2, '0')}</span>
                <i />
                <small>{formatVerdict(event.verdict)}</small>
              </button>
            )
          })}
        </div>
      </section>

      <section className="observability-grid">
        <article className="ledger-panel panel">
          <div className="section-heading compact-heading">
            <div>
              <p className="eyebrow">Append-only chain</p>
              <h2>Live ledger</h2>
            </div>
            <span className={`chain-ok ${chainValid === false ? 'chain-bad' : ''}`}><i /> {dataMode === 'demo' ? 'seeded chain' : chainValid === true ? 'chain valid' : chainValid === false ? 'chain invalid' : 'verifying chain'}</span>
          </div>
          <div className="ledger-list">
            {[...visibleEvents].reverse().map((event) => (
              <button
                type="button"
                key={event.seq}
                className={`ledger-row ${selected?.seq === event.seq ? 'is-selected' : ''}`}
                onClick={() => setSelectedSeq(event.seq)}
              >
                <span className="ledger-seq">#{String(event.seq).padStart(4, '0')}</span>
                <span className={`verdict-chip verdict-${event.verdict.toLowerCase()}`}>{formatVerdict(event.verdict)}</span>
                <span className="ledger-copy">
                  <strong>{event.title}</strong>
                  <small>{event.rootId}</small>
                </span>
                <span className={`delta ${event.delta > 0 ? 'positive' : event.delta < 0 ? 'negative' : ''}`}>
                  {event.delta > 0 ? '+' : ''}{(event.delta * 100).toFixed(1)} pp
                </span>
              </button>
            ))}
          </div>
        </article>

        <aside className="trace-panel panel" aria-live="polite">
          {selected ? (
            <>
              <div className="section-heading compact-heading trace-heading">
                <div>
                  <p className="eyebrow">Selected transaction / #{String(selected.seq).padStart(4, '0')}</p>
                  <h2>{selected.title}</h2>
                </div>
                <span className={`integrity-chip integrity-${selected.integrity.slice(1, 2)}`}>{selected.integrity}</span>
              </div>

              <div className="trace-section">
                <h3>Reference monitor</h3>
                <ul className="reason-list">
                  {selected.reasons.map((reason) => <li key={reason}><span>✓</span>{formatReason(reason)}</li>)}
                </ul>
              </div>

              <div className="trace-section">
                <h3>Engine breakdown</h3>
                {selected.engine ? (
                  <div className="engine-grid">
                    <div><span>Prior</span><strong>{formatPercent(selected.engine.prior)}</strong></div>
                    <div><span>Raw BF</span><strong>{selected.engine.rawBf.toFixed(2)}×</strong></div>
                    <div><span>Candidate Δ</span><strong className={selected.engine.boundedDelta < 0 ? 'negative' : ''}>{selected.engine.boundedDelta > 0 ? '+' : ''}{(selected.engine.boundedDelta * 100).toFixed(1)} pp</strong></div>
                    <div><span>Root spent</span><strong>{formatPercent(selected.engine.rootSpent)}</strong></div>
                    <div className="engine-posterior"><span>Candidate post.</span><strong>{formatPercent(selected.engine.posterior)}</strong></div>
                  </div>
                ) : <p className="muted-copy">Engine not invoked. The monitor rejected this action at the boundary.</p>}
              </div>

              <div className="trace-section explainer">
                <div className="explainer-title"><h3>Safe explainer</h3><span>lazy / metadata only</span></div>
                {selected.integrity === 'L0 · RAW'
                  ? <p className="locked-copy"><span>Locked</span> Raw L0 content is never sent to or rendered by the explainer.</p>
                  : <p>{remoteExplanation ?? selected.explanation}</p>}
              </div>
            </>
          ) : (
            <div className="empty-trace"><span>Awaiting event</span><p>Start the replay to inspect deterministic transaction traces.</p></div>
          )}
        </aside>
      </section>

      <section className="metrics-section panel" aria-label="Security and performance metrics">
        <div className="section-heading compact-heading">
          <div>
            <p className="eyebrow">Observability / current replay</p>
            <h2>Firewall telemetry</h2>
          </div>
          <p className="metrics-note">{dataMode === 'live' && backendMetrics ? 'Backend /metrics snapshot' : 'Provider-free seeded values'} · refreshed per event</p>
        </div>
        <div className="metric-groups">
          <div className="metric-group security-metrics">
            <h3>Security</h3>
            <div className="metric-grid">
              <div>
                <span>Unauthorized transitions</span>
                <strong>{dataMode === 'live' ? backendMetrics?.unauthorizedTransitions ?? '—' : 0}</strong>
                <small>{dataMode === 'live' ? backendMetrics?.unauthorizedTransitions == null ? 'not reported by API' : 'backend security report' : 'seeded demo assertion'}</small>
              </div>
              <div>
                <span>Reversal completeness</span>
                <strong>{dataMode === 'live' && backendMetrics
                  ? `${backendMetrics.reversalPercentage.toFixed(0)}%`
                  : localCoverage === null ? '—' : formatPercent(localCoverage)}</strong>
                <small>{dataMode === 'live' && backendMetrics
                  ? `${backendMetrics.reversalReversed} / ${backendMetrics.reversalCommitted} committed IDs`
                  : localCoverage === null ? 'no retracted root' : 'referenced commits reversed'}</small>
              </div>
              <div>
                <span>Escrow release / reject</span>
                <strong>{dataMode === 'live' && backendMetrics
                  ? `${backendMetrics.escrowReleased} / ${backendMetrics.escrowRejected}`
                  : `${visibleEvents.filter((event) => event.eventType === 'ESCROW_RELEASED').length} / ${visibleEvents.filter((event) => event.eventType === 'ESCROW_REJECTED').length}`}</strong>
                <small>{dataMode === 'live' && backendMetrics ? 'backend outcomes' : 'ledger events'}</small>
              </div>
              <div><span>Zero-influence entries</span><strong>{visibleEvents.filter((event) => event.delta === 0).length}</strong><small>non-applying ledger events</small></div>
            </div>
          </div>
          <div className="metric-group performance-metrics">
            <h3>Performance & cost</h3>
            <div className="metric-grid">
              <div><span>Fast-lane p50 / p95</span><strong>{telemetry.p50.toFixed(0)} / {telemetry.p95.toFixed(0)} ms</strong><small>end-to-end</small></div>
              <div><span>Escalation rate</span><strong>{formatPercent(telemetry.escalationRate)}</strong><small>risk triggered</small></div>
              <div><span>Gemini calls / event</span><strong>{telemetry.callsPerEvent.toFixed(1)}</strong><small>compile + embed</small></div>
              <div><span>Token mix</span><strong>{telemetry.tokens.toLocaleString()}</strong><small>{formatPercent(telemetry.cacheFraction)} cached fraction</small></div>
            </div>
          </div>
        </div>
      </section>

      <footer>
        <span>Evidence Firewall / coordination build</span>
        <span>Safety from bounded authority — not perfect detection.</span>
      </footer>
    </main>
  )
}

export default App
