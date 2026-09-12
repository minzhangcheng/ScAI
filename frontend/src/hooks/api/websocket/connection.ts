import { useAppStore } from '../../../stores/appStore'
import { restoreAgentErrors } from '../../../utils/messageTransform'
import { devLog } from '../../../utils/logger'
import { getToken } from '../../../utils/authStorage'
import { WS_AUTH_SUBPROTOCOL } from '../../../constants/auth'
import { WS_CONNECT_TIMEOUT_MS, ApiError } from '../client'
import { chatOwnerKey } from '../../../utils/chatOwnership'
import { loadRealtimeMessageHistory, mergeRealtimeMessages } from '../realtimeHistory'
import { notifySnapshotListingChanged } from '../snapshots'
import { handleWebSocketMessage } from './messageHandlers'
import { reloadSessionState } from './sessionReload'
import type { ConnectionContext, ConnectionRefs } from './types'

export const RECONNECTABLE_SESSION_STATUSES = new Set([
  'running', 'waiting_input', 'waiting_annotation', 'waiting_plan_review',
  'waiting_species_input', 'generating_report',
])

export function connectWebSocket(ctx: ConnectionContext): void {
  const {
    wsRef, connectionAttemptRef, connectTimeoutRef, reconnectCountRef,
    reconnectTimerRef, stateSyncEpochRef, setConnected,
    setWsConnected, addMessage,
  } = ctx
  // Use fresh state to avoid stale closure references
  const session = useAppStore.getState().currentSession
  if (!session) {
    devLog('WebSocket: No current session, skipping connect')
    return
  }

  // Each call owns a distinct generation. Retiring a CONNECTING socket must
  // not leave the next call behind the previous generation's 10-second
  // timeout, and callbacks from the retired socket must become inert.
  const attempt = connectionAttemptRef.current + 1
  connectionAttemptRef.current = attempt
  if (reconnectTimerRef.current) {
    clearTimeout(reconnectTimerRef.current)
    reconnectTimerRef.current = null
  }
  if (connectTimeoutRef.current) {
    clearTimeout(connectTimeoutRef.current)
    connectTimeoutRef.current = null
  }
  const previousSocket = wsRef.current
  wsRef.current = null
  if (previousSocket) {
    const state = previousSocket.readyState
    if (state === WebSocket.CONNECTING || state === WebSocket.OPEN) {
      devLog('WebSocket: Closing existing connection before reconnecting')
      previousSocket.close()
    }
  }
  setConnected(false)
  setWsConnected(false)
  const isReconnect = reconnectCountRef.current > 0
  const ownerKey = chatOwnerKey(session.id, session.restoreBranchId)

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/ws/${session.id}`
  const token = getToken()
  const wsProtocols = token ? [WS_AUTH_SUBPROTOCOL, token] : undefined

  devLog(`WebSocket: Connecting to ${wsUrl}${isReconnect ? ' (reconnect #' + reconnectCountRef.current + ')' : ''}`)

  const ws = new WebSocket(wsUrl, wsProtocols)
  wsRef.current = ws
  const isCurrentAttempt = () => (
    connectionAttemptRef.current === attempt && wsRef.current === ws
  )
  const ownsSocketSession = () => {
    const current = useAppStore.getState().currentSession
    return isCurrentAttempt()
      && current?.id === session.id
      && chatOwnerKey(current.id, current.restoreBranchId) === ownerKey
  }
  const clearCurrentConnectTimeout = () => {
    if (!isCurrentAttempt()) return
    if (connectTimeoutRef.current) {
      clearTimeout(connectTimeoutRef.current)
      connectTimeoutRef.current = null
    }
  }

  // Connection timeout: close and retry if onopen doesn't fire within 10 seconds
  const connectTimeout = setTimeout(() => {
    if (!isCurrentAttempt()) return
    connectTimeoutRef.current = null
    if (ws.readyState !== WebSocket.OPEN) {
      devLog('WebSocket: Connection timeout (10s), closing')
      ws.close()
    }
  }, WS_CONNECT_TIMEOUT_MS)
  connectTimeoutRef.current = connectTimeout

  ws.onopen = async () => {
    if (!isCurrentAttempt()) return
    if (!ownsSocketSession()) {
      clearCurrentConnectTimeout()
      wsRef.current = null
      ws.close()
      return
    }
    clearCurrentConnectTimeout()
    setConnected(true)
    setWsConnected(true)
    devLog('WebSocket: Connected successfully')

    // Capture invalidations are intentionally not durable messages. Refresh
    // immediately on reconnect so an unrelated state/history reload failure
    // cannot leave a missed snapshot event stale until the tab is hidden.
    if (isReconnect) notifySnapshotListingChanged(session.id)

    // Use fresh state to avoid stale closure
    const freshSession = useAppStore.getState().currentSession

    // On reconnect, reload authoritative session state before paging only the
    // disconnected realtime gap.
    if (isReconnect && freshSession) {
      const loadingSessionId = freshSession.id
      const reconnectLoadEpoch = stateSyncEpochRef.current
      const ownsReconnectStateLoad = () => {
        const current = useAppStore.getState().currentSession
        return isCurrentAttempt()
          && current?.id === loadingSessionId
          && chatOwnerKey(current.id, current.restoreBranchId) === ownerKey
          && stateSyncEpochRef.current === reconnectLoadEpoch
      }
      const ownsReconnectSocket = () => {
        const current = useAppStore.getState().currentSession
        return isCurrentAttempt()
          && current?.id === loadingSessionId
          && chatOwnerKey(current.id, current.restoreBranchId) === ownerKey
      }
      try {
        devLog('WebSocket: Reconnect detected, reloading session state...')
        const data = await reloadSessionState(
          loadingSessionId,
          ownsReconnectStateLoad,
        )

        if (!ownsReconnectSocket()) return
        if (data && Array.isArray(data.agent_errors)) {
          restoreAgentErrors(data.agent_errors)
        }
        if (data && Array.isArray(data.messages)) {
          mergeRealtimeMessages(data.messages)
        }
        useAppStore.getState().setRealtimeHistoryError(null)
        const reportHistoryFailure = (error: unknown) => {
          if (ownsReconnectSocket()) {
            useAppStore.getState().setRealtimeHistoryError(
              error instanceof Error ? error.message : 'Failed to load realtime history',
            )
          }
        }
        await loadRealtimeMessageHistory(loadingSessionId, {
          ownerKey,
          ownsResponse: ownsReconnectSocket,
          stopWhenOverlapping: true,
          onInitialFailure: reportHistoryFailure,
          onFailure: reportHistoryFailure,
        })
      } catch (e: unknown) {
        if (!isCurrentAttempt()) return
        // apiFetch already applies the lifecycle contract. Keep this guard so
        // the socket also closes immediately, but key only on stable codes.
        if (
          e instanceof ApiError &&
          (e.code === 'SESSION_NOT_FOUND' || e.code === 'SESSION_GONE') &&
          (
            useAppStore.getState().currentSession?.id === loadingSessionId
            || useAppStore.getState().currentSession == null
          )
        ) {
          console.warn(`WebSocket: ${e.code}, closing stale connection`)
          ws.close()
          return
        }
        if (ownsReconnectSocket()) {
          useAppStore.getState().setRealtimeHistoryError(
            e instanceof Error ? e.message : 'Failed to reload session state/history',
          )
        }
        console.error('WebSocket: Failed to restore session state/history:', e)
        return
      }
    }

    // Reset reconnect count on successful connection
    if (isCurrentAttempt()) reconnectCountRef.current = 0
  }

  ws.onmessage = (event) => handleWebSocketMessage(event, {
    session, attempt, ws, isCurrentAttempt, ownsSocketSession, ownerKey,
    stateSyncEpochRef, addMessage,
  })

  ws.onerror = (error) => {
    if (!ownsSocketSession()) return
    clearCurrentConnectTimeout()
    console.error('WebSocket: Error:', error)
  }

  ws.onclose = (event) => {
    if (!isCurrentAttempt()) return
    clearCurrentConnectTimeout()
    setConnected(false)
    setWsConnected(false)
    wsRef.current = null
    devLog('WebSocket: Disconnected, code:', event.code, 'reason:', event.reason)

    // The selected owner may have changed before this old socket closed. A
    // same-id branch switch is a different owner: do not clear or reconnect
    // the new owner's socket from an old attempt.
    const ownerAtClose = chatOwnerKey(
      useAppStore.getState().currentSession?.id,
      useAppStore.getState().currentSession?.restoreBranchId,
    )
    if (ownerAtClose !== ownerKey) return

    if (event.code === 4404 || event.code === 4410) {
      const store = useAppStore.getState()
      const currentOwner = chatOwnerKey(
        store.currentSession?.id, store.currentSession?.restoreBranchId,
      )
      if (currentOwner === ownerKey) {
        store.setCurrentSession(null)
        store.clearMessages()
      }
      return
    }

    // Don't reconnect on auth failure (4401) or if no token
    if (event.code === 4401 || !getToken()) {
      devLog('WebSocket: Auth failure or no token, not reconnecting')
      return
    }

    reconnectCountRef.current += 1  // Track reconnect for history recovery

    // Auto-reconnect if session is still running (not manually disconnected)
    const latestSession = useAppStore.getState().currentSession
    if (latestSession &&
        RECONNECTABLE_SESSION_STATUSES.has(latestSession.status) &&
        event.code !== 1000) {  // Not a normal close
      if (reconnectCountRef.current < 5) {  // Max 5 reconnect attempts
        const delay = Math.min(1000 * Math.pow(2, reconnectCountRef.current - 1), 10000)  // Exponential backoff, max 10s
        devLog(`WebSocket: Auto-reconnecting in ${delay}ms (attempt ${reconnectCountRef.current})...`)
        const reconnectTimer = setTimeout(() => {
          if (
            connectionAttemptRef.current !== attempt
            || reconnectTimerRef.current !== reconnectTimer
          ) return
          reconnectTimerRef.current = null
          const reconnectOwner = chatOwnerKey(
            useAppStore.getState().currentSession?.id,
            useAppStore.getState().currentSession?.restoreBranchId,
          )
          if (reconnectOwner === ownerKey && !wsRef.current) {
            connectWebSocket(ctx)
          }
        }, delay)
        reconnectTimerRef.current = reconnectTimer
      } else {
        // Max reconnect attempts reached - notify user
        console.error('WebSocket: Max reconnection attempts reached, giving up')
        addMessage({
          id: `ws-error-${Date.now()}`,
          type: 'error',
          agent: undefined,
          content: 'WebSocket connection lost. Please refresh the page or click reconnect.',
          timestamp: new Date().toISOString()
        })
      }
    }
  }
}

export function disconnectWebSocket(
  refs: ConnectionRefs,
  setConnected: (connected: boolean) => void,
  setWsConnected: (connected: boolean) => void,
): void {
  const { wsRef, connectionAttemptRef, connectTimeoutRef, reconnectTimerRef } = refs
  connectionAttemptRef.current += 1
  if (reconnectTimerRef.current) {
    clearTimeout(reconnectTimerRef.current)
    reconnectTimerRef.current = null
  }
  if (connectTimeoutRef.current) {
    clearTimeout(connectTimeoutRef.current)
    connectTimeoutRef.current = null
  }
  const currentSocket = wsRef.current
  wsRef.current = null
  if (currentSocket) {
    devLog('WebSocket: Disconnecting manually')
    currentSocket.close()
  }
  setConnected(false)
  setWsConnected(false)
}
