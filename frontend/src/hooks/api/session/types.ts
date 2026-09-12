import type { MutableRefObject } from 'react'

export interface ActiveSessionLoad {
  generation: number
  controller: AbortController
  /** Normalized owner installed when the load started. */
  ownerKey: string | null
  /** Session id the load is responsible for installing. */
  requestedSessionId: string
}

export interface ActiveSessionHistory {
  sessionId: string
  generation: number
  controller: AbortController
  /** Normalized `<session-id>,<branch-id>` captured when paging started. */
  ownerKey?: string | null
}

export interface SessionLoadingOwnership {
  loadGenerationRef: MutableRefObject<number>
  activeLoadRef: MutableRefObject<ActiveSessionLoad | null>
  activeHistoryRef: MutableRefObject<ActiveSessionHistory | null>
}
