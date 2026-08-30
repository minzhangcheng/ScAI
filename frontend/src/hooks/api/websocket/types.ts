import type { Dispatch, MutableRefObject, SetStateAction } from 'react'
import type { Message, Session } from '../../../stores/appStore'

export type RestoreCommitRefreshes = { keys: Set<string>; order: string[] }

export type ConnectionRefs = {
  wsRef: MutableRefObject<WebSocket | null>
  connectionAttemptRef: MutableRefObject<number>
  connectTimeoutRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  reconnectCountRef: MutableRefObject<number>
  reconnectTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  restoreCommitRefreshesRef: MutableRefObject<RestoreCommitRefreshes>
  stateSyncEpochRef: MutableRefObject<number>
}

export type ConnectionContext = ConnectionRefs & {
  setConnected: Dispatch<SetStateAction<boolean>>
  setWsConnected: (connected: boolean) => void
  addMessage: (message: Message) => void
}

export type MessageHandlerContext = {
  session: Session
  attempt: number
  ws: WebSocket
  isCurrentAttempt: () => boolean
  ownsSocketSession: () => boolean
  restoreCommitRefreshesRef: MutableRefObject<RestoreCommitRefreshes>
  stateSyncEpochRef: MutableRefObject<number>
  addMessage: (message: Message) => void
}
