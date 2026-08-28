// WebSocket connection, message dispatch, and snapshot reminders.

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAppStore } from '../../stores/appStore'
import { connectWebSocket, disconnectWebSocket } from './websocket/connection'
import type { ConnectionRefs } from './websocket/types'

export function useWebSocket() {
  const { currentSession, addMessage, updateSession, setWsConnected, addAgentError, setStateVersion } = useAppStore()
  const wsRef = useRef<WebSocket | null>(null)
  const connectionAttemptRef = useRef(0)
  const connectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectCountRef = useRef<number>(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const restoreCommitRefreshesRef = useRef<{ keys: Set<string>; order: string[] }>({
    keys: new Set(),
    order: [],
  })
  const [connected, setConnected] = useState(false)

  const refs: ConnectionRefs = {
    wsRef, connectionAttemptRef, connectTimeoutRef, reconnectCountRef,
    reconnectTimerRef, restoreCommitRefreshesRef,
  }

  const connect = useCallback(() => {
    connectWebSocket({
      ...refs, setConnected, setWsConnected, addMessage,
    })
  }, [currentSession, addMessage, updateSession, setWsConnected, addAgentError, setStateVersion])

  const disconnect = useCallback(() => {
    disconnectWebSocket(refs, setConnected, setWsConnected)
  }, [setWsConnected])

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return { connect, disconnect, connected }
}
