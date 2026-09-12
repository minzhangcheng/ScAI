// WebSocket connection and message dispatch.

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAppStore } from '../../stores/appStore'
import { chatOwnerKey } from '../../utils/chatOwnership'
import {
  RECONNECTABLE_SESSION_STATUSES,
  connectWebSocket,
  disconnectWebSocket,
} from './websocket/connection'
import { loadRealtimeMessageHistory } from './realtimeHistory'
import type { ConnectionRefs } from './websocket/types'

export function useWebSocket() {
  const { currentSession, addMessage, setWsConnected } = useAppStore()
  const wsRef = useRef<WebSocket | null>(null)
  const connectionAttemptRef = useRef(0)
  const stateSyncEpochRef = useRef(0)
  const connectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectCountRef = useRef<number>(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [connected, setConnected] = useState(false)

  const refs: ConnectionRefs = {
    wsRef, connectionAttemptRef, connectTimeoutRef, reconnectCountRef,
    reconnectTimerRef, stateSyncEpochRef,
  }

  const ownerKey = chatOwnerKey(
    currentSession?.id, currentSession?.restoreBranchId,
  )
  const previousSessionIdRef = useRef(currentSession?.id)

  const connect = useCallback(() => {
    connectWebSocket({
      ...refs, setConnected, setWsConnected, addMessage,
    })
  }, [addMessage, setWsConnected])

  const disconnect = useCallback(() => {
    disconnectWebSocket(refs, setConnected, setWsConnected)
  }, [setWsConnected])

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  const previousOwnerRef = useRef(ownerKey)
  useEffect(() => {
    if (previousOwnerRef.current === ownerKey) return
    const previousSessionId = previousSessionIdRef.current
    previousOwnerRef.current = ownerKey
    previousSessionIdRef.current = currentSession?.id
    // Every owner change, including a same-id branch change, is retired and
    // replaced here. AnalysisPanel deliberately does not compete for owner
    // transitions, so this hook establishes exactly one replacement socket.
    disconnect()
    // Reconnect state belongs to the retired owner. Carrying it into the new
    // owner would make connectWebSocket run a second reload/history recovery
    // after the branch transition already started its authoritative pager.
    reconnectCountRef.current = 0
    const session = useAppStore.getState().currentSession
    // Normal session selection is already owned by useSessionLoading, which
    // installs the authoritative payload and starts its one history pager.
    // A branch-only update has no load hook to do that work, so this effect is
    // the sole history owner for that transition.
    if (session && previousSessionId === session.id) {
      // The new owner's message window is already empty (the atomic session
      // transition cleared it); load exactly one authoritative history page
      // set for this owner.
      const reportHistoryFailure = (error: unknown) => {
        const current = useAppStore.getState().currentSession
        if (
          current
          && chatOwnerKey(current.id, current.restoreBranchId) === ownerKey
        ) {
          useAppStore.getState().setRealtimeHistoryError(
            error instanceof Error ? error.message : 'Failed to load realtime history',
          )
        }
      }
      void loadRealtimeMessageHistory(session.id, {
        ownerKey,
        ownsResponse: () => (
          chatOwnerKey(
            useAppStore.getState().currentSession?.id,
            useAppStore.getState().currentSession?.restoreBranchId,
          ) === ownerKey
        ),
        onFailure: reportHistoryFailure,
      })
    }
    if (session && RECONNECTABLE_SESSION_STATUSES.has(session.status)) {
      connect()
    }
  }, [ownerKey, currentSession?.id, connect, disconnect])

  return { connect, disconnect, connected }
}
