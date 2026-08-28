import { useAppStore } from '../../../stores/appStore'
import { restoreAgentErrors } from '../../../utils/messageTransform'
import { transformLoadResponse } from '../../../utils/transformLoadResponse'
import { devLog } from '../../../utils/logger'
import { getToken } from '../../../utils/authStorage'
import { WS_AUTH_SUBPROTOCOL } from '../../../constants/auth'
import { WS_CONNECT_TIMEOUT_MS, ApiError, apiFetch, applyStoreUpdates } from '../client'
import { loadRealtimeMessageHistory, mergeRealtimeMessages } from '../realtimeHistory'
import { handleWebSocketMessage } from './messageHandlers'
import type { ConnectionContext, ConnectionRefs } from './types'

const RECONNECTABLE_SESSION_STATUSES = new Set([
  'running', 'waiting_input', 'waiting_annotation', 'waiting_plan_review',
  'waiting_species_input', 'generating_report',
])

export function connectWebSocket(ctx: ConnectionContext): void {
  const {
    wsRef, connectionAttemptRef, connectTimeoutRef, reconnectCountRef,
    reconnectTimerRef, restoreCommitRefreshesRef, setConnected,
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
  const ownsSocketSession = () => (
    isCurrentAttempt()
    && useAppStore.getState().currentSession?.id === session.id
  )
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

    // Use fresh state to avoid stale closure
    const freshSession = useAppStore.getState().currentSession

    // On reconnect, reload authoritative session state before paging only the
    // disconnected realtime gap.
    if (isReconnect && freshSession) {
      const loadingSessionId = freshSession.id
      const ownsReconnectLoad = () => (
        isCurrentAttempt()
        && useAppStore.getState().currentSession?.id === loadingSessionId
      )
      try {
        devLog('WebSocket: Reconnect detected, reloading session state...')
        const data = await apiFetch<any>(
          `/session/${freshSession.id}/load`,
          undefined,
          ownsReconnectLoad,
        )

        if (!ownsReconnectLoad()) return
        const { sessionUpdates, storeUpdates } = transformLoadResponse(data)
        useAppStore.getState().updateSession(sessionUpdates)
        applyStoreUpdates(storeUpdates)
        if (data.agent_errors && Array.isArray(data.agent_errors)) {
          restoreAgentErrors(data.agent_errors)
        }
        if (data.messages && Array.isArray(data.messages)) {
          mergeRealtimeMessages(data.messages)
        }
        useAppStore.getState().setRealtimeHistoryError(null)
        const reportHistoryFailure = (error: unknown) => {
          if (ownsReconnectLoad()) {
            useAppStore.getState().setRealtimeHistoryError(
              error instanceof Error ? error.message : 'Failed to load realtime history',
            )
          }
        }
        await loadRealtimeMessageHistory(loadingSessionId, {
          ownsResponse: ownsReconnectLoad,
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
          useAppStore.getState().currentSession?.id === loadingSessionId
        ) {
          console.warn(`WebSocket: ${e.code}, closing stale connection`)
          ws.close()
          return
        }
        if (ownsReconnectLoad()) {
          useAppStore.getState().setRealtimeHistoryError(
            e instanceof Error ? e.message : 'Failed to reload session state/history',
          )
        }
        console.error('WebSocket: Failed to restore session state/history:', e)
      }
    }

    // Reset reconnect count on successful connection
    if (isCurrentAttempt()) reconnectCountRef.current = 0
  }

  ws.onmessage = (event) => handleWebSocketMessage(event, {
    session, attempt, ws, isCurrentAttempt, ownsSocketSession,
    restoreCommitRefreshesRef, addMessage,
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

    // The selected session may have changed before this old socket closed.
    // Do not let A schedule a reconnect whose fresh connect() targets B.
    if (useAppStore.getState().currentSession?.id !== session.id) return

    if (event.code === 4404 || event.code === 4410) {
      const store = useAppStore.getState()
      if (store.currentSession?.id === session.id) {
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
          if (!wsRef.current) {
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
