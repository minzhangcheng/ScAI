import type { MutableRefObject } from 'react'

export interface ActiveSessionLoad {
  generation: number
  controller: AbortController
}

export interface ActiveSessionHistory {
  sessionId: string
  generation: number
  controller: AbortController
}

export interface SessionLoadingOwnership {
  loadGenerationRef: MutableRefObject<number>
  activeLoadRef: MutableRefObject<ActiveSessionLoad | null>
  activeHistoryRef: MutableRefObject<ActiveSessionHistory | null>
}
