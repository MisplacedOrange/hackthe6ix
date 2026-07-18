import { useEffect, useState } from 'react'

export type StreamStatus = 'connecting' | 'live' | 'fallback'

export interface StreamMessage {
  seq?: number
  event_type?: string
  payload?: Record<string, unknown>
  [key: string]: unknown
}

/** The application's single EventSource seam. */
export function useEventStream(url: string) {
  const [status, setStatus] = useState<StreamStatus>('connecting')
  const [message, setMessage] = useState<StreamMessage | null>(null)

  useEffect(() => {
    if (typeof EventSource === 'undefined') {
      setStatus('fallback')
      return
    }

    const source = new EventSource(url)
    source.onopen = () => setStatus('live')
    const receive = (event: MessageEvent<string>) => {
      try {
        setMessage(JSON.parse(event.data) as StreamMessage)
      } catch {
        // Malformed SSE stays inert; it cannot become a display or state action.
      }
    }
    source.onmessage = receive
    const ledgerEventTypes = [
      'CLAIM_REGISTERED',
      'EVIDENCE_COMMITTED',
      'EVIDENCE_PROVISIONAL',
      'EVIDENCE_ESCROWED',
      'EVIDENCE_REJECTED',
      'ESCROW_RELEASED',
      'ESCROW_REJECTED',
      'RETRACTION',
      'REVERSAL',
    ]
    ledgerEventTypes.forEach((eventType) => source.addEventListener(eventType, receive as EventListener))
    source.onerror = () => setStatus('fallback')

    return () => {
      ledgerEventTypes.forEach((eventType) => source.removeEventListener(eventType, receive as EventListener))
      source.close()
    }
  }, [url])

  return { status, message }
}
