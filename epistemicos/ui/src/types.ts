export type ClaimState = 'SUPPORTED' | 'CONTRADICTED' | 'CONTESTED' | 'UNKNOWN'

export type Verdict =
  | 'COMMIT'
  | 'PROVISIONAL'
  | 'ESCROW'
  | 'REJECT'
  | 'RELEASE'
  | 'ESCROW_REJECTED'
  | 'RETRACTION'
  | 'REVERSAL'

export type Integrity = 'L0 · RAW' | 'L1 · PARSED' | 'L2 · VERIFIED' | 'L3 · REPLICATED'

export type EventKind =
  | 'EVIDENCE_COMMITTED'
  | 'EVIDENCE_PROVISIONAL'
  | 'EVIDENCE_ESCROWED'
  | 'EVIDENCE_REJECTED'
  | 'ESCROW_RELEASED'
  | 'ESCROW_REJECTED'
  | 'RETRACTION'
  | 'REVERSAL'

export interface EngineBreakdown {
  prior: number
  rawBf: number
  boundedDelta: number
  rootSpent: number
  posterior: number
  integrity: Integrity
}

export interface EventMetrics {
  latencyMs: number
  geminiCalls: number
  escalated: boolean
  inputTokens: number
  outputTokens: number
  thinkingTokens: number
  cachedTokens: number
}

/**
 * Deliberately contains no raw input field. L0 bytes never enter the view model,
 * which keeps the observability/explainer surface provenance-only.
 */
export interface LedgerEvent {
  seq: number
  stepId: string
  title: string
  eventType: EventKind
  verdict: Verdict
  integrity: Integrity
  /** Committed ledger delta. Zero for provisional, escrow, reject, and retraction. */
  delta: number
  rootId: string
  relation: string | null
  reversesSeqs: number[]
  reasons: string[]
  tags: string[]
  engine: EngineBreakdown | null
  metrics: EventMetrics
  explanation: string
}

export interface TimelinePoint {
  seq: number
  label: string
  confidence: number
  delta: number
  kind: EventKind | 'BASELINE'
}
