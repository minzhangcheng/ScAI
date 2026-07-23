// WebSocket connection, message dispatch, and snapshot reminders.
// Split out of frontend/src/hooks/useApi.ts verbatim.

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAppStore, Message, Session, AgentType, RoundResult } from '../../stores/appStore'
import { transformRawMessage, restoreAgentErrors } from '../../utils/messageTransform'
import { transformLoadResponse, transformRoundResults } from '../../utils/transformLoadResponse'
import { devLog } from '../../utils/logger'
import { roundKey } from '../../utils/roundKey'
import { getToken } from '../../utils/authStorage'
import { WS_AUTH_SUBPROTOCOL } from '../../constants/auth'
import type { ChatMessage } from '../../types'
import { resolveSnapshotFailureActions } from '../../utils/snapshotV2'
import { API_BASE, WS_CONNECT_TIMEOUT_MS, apiFetch, applyStoreUpdates, authHeaders, generateId } from './client'

const SNAPSHOT_CLEANUP_THRESHOLD = 20
const SNAPSHOT_REMINDER_INTERVAL = 5
export function useWebSocket() {
  const { currentSession, addMessage, updateSession, setWsConnected, setMessages, addAgentError, setStateVersion } = useAppStore()
  const wsRef = useRef<WebSocket | null>(null)
  const connectingRef = useRef<boolean>(false)  // Prevent multiple simultaneous connect attempts
  const reconnectCountRef = useRef<number>(0)  // Track reconnection attempts for history recovery
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [connected, setConnected] = useState(false)

  const connect = useCallback(() => {
    // Use fresh state to avoid stale closure references
    const session = useAppStore.getState().currentSession
    if (!session) {
      devLog('WebSocket: No current session, skipping connect')
      return
    }

    // Close existing connection before opening a new one
    if (wsRef.current) {
      const state = wsRef.current.readyState
      if (state === WebSocket.CONNECTING || state === WebSocket.OPEN) {
        devLog('WebSocket: Closing existing connection before reconnecting')
        wsRef.current.close()
      }
      wsRef.current = null
    }

    if (connectingRef.current) {
      devLog('WebSocket: Connection already in progress, skipping')
      return
    }

    connectingRef.current = true
    const isReconnect = reconnectCountRef.current > 0

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/${session.id}`
    const token = getToken()
    const wsProtocols = token ? [WS_AUTH_SUBPROTOCOL, token] : undefined

    devLog(`WebSocket: Connecting to ${wsUrl}${isReconnect ? ' (reconnect #' + reconnectCountRef.current + ')' : ''}`)

    const ws = new WebSocket(wsUrl, wsProtocols)
    wsRef.current = ws

    // Connection timeout: close and retry if onopen doesn't fire within 10 seconds
    const connectTimeout = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        devLog('WebSocket: Connection timeout (10s), closing')
        connectingRef.current = false
        ws.close()
      }
    }, WS_CONNECT_TIMEOUT_MS)

    ws.onopen = async () => {
      if (wsRef.current !== ws) return
      clearTimeout(connectTimeout)
      connectingRef.current = false
      setConnected(true)
      setWsConnected(true)
      devLog('WebSocket: Connected successfully')

      // Use fresh state to avoid stale closure
      const freshSession = useAppStore.getState().currentSession

      // On reconnect, restore message history from server
      if (isReconnect && freshSession) {
        try {
          devLog('WebSocket: Reconnect detected, loading message history...')
          // Mark that we're loading history to prevent race conditions
          const loadingSessionId = freshSession.id
          const data = await apiFetch<any>(`/session/${freshSession.id}/load`)

          // Verify session hasn't changed during fetch (race condition protection)
          const currentState = useAppStore.getState()
          if (currentState.currentSession?.id !== loadingSessionId) {
            devLog('WebSocket: Session changed during reconnect, skipping history merge')
            reconnectCountRef.current = 0
            return
          }

          if (data) {

            if (data.messages && Array.isArray(data.messages)) {
              // Convert server messages to frontend format
              const serverMessages: Message[] = data.messages.map(transformRawMessage)

              // Transactional merge: read+write inside setState so messages that
              // arrived via WebSocket during the async fetch are preserved.
              let mergedCount = 0
              let localOnlyCount = 0
              useAppStore.setState((state) => {
                const serverMsgIdSet = new Set(serverMessages.map(m => m.id))
                localOnlyCount = state.messages.filter(m => !serverMsgIdSet.has(m.id)).length

                const currentIdSet = new Set(state.messages.map((m: Message) => m.id))
                const newFromServer = serverMessages.filter((m) => !currentIdSet.has(m.id))
                if (newFromServer.length === 0) {
                  return state
                }
                mergedCount = newFromServer.length
                const merged = [...state.messages, ...newFromServer]
                  .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
                return { ...state, messages: merged }
              })
              devLog(`WebSocket: Restored ${serverMessages.length} messages (${mergedCount} new merged), ${localOnlyCount} local-only`)
            }

            // Also restore agent errors if present
            if (data.agent_errors && Array.isArray(data.agent_errors)) {
              restoreAgentErrors(data.agent_errors)
            }

            // Sync full session state from server using shared transform
            const { sessionUpdates, storeUpdates } = transformLoadResponse(data)
            const currentSess = useAppStore.getState().currentSession
            if (currentSess && currentSess.id === loadingSessionId) {
              useAppStore.getState().updateSession(sessionUpdates)
              applyStoreUpdates(storeUpdates)
            }
          }
        } catch (e: any) {
          // Session no longer exists on server - clear stale session to stop reconnect loop
          if (e?.message?.includes('not found') || e?.message?.includes('404')) {
            console.warn('WebSocket: Session not found on server (404), clearing stale session')
            useAppStore.getState().setCurrentSession(null)
            useAppStore.getState().clearMessages()
            ws.close()
            return
          }
          console.error('WebSocket: Failed to restore message history:', e)
        }
      }

      // Reset reconnect count on successful connection
      reconnectCountRef.current = 0
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        // Server heartbeat ping — ignore silently
        if (data.type === 'ping') return

        // Handle error messages that indicate session/config issues
        if (data.error) {
          console.error('WebSocket: Server error:', data.error)
          // If session not found, clear stale session to stop reconnect loop
          // Match specific backend error pattern: "Session {id} not found" or "Session not found"
          if (typeof data.error === 'string' && /session\b.*\bnot found/i.test(data.error)) {
            const currentSessionId = useAppStore.getState().currentSession?.id
            if (currentSessionId) {
              console.warn(`WebSocket: Session ${currentSessionId} not found on server, clearing stale session`)
              useAppStore.getState().setCurrentSession(null)
              useAppStore.getState().clearMessages()
            }
          }
          return
        }

        // Handle state_sync messages - version control and state synchronization
        if (data.type === 'state_sync') {
          // Normalise: some backend paths nest fields under `content`
          // while others place them at the top level of the message.
          const syncContent = data.content ?? data
          const currentVersion = useAppStore.getState().stateVersion

          // Check for version conflict - reload full session state from server
          if (syncContent.version < currentVersion) {
            console.warn('WebSocket: Received outdated state sync, current:', currentVersion, 'received:', syncContent.version, '. Reloading session state.')
            const sessionId = useAppStore.getState().currentSession?.id
            if (sessionId) {
              fetch(`${API_BASE}/session/${sessionId}/load`, { headers: authHeaders() })
                .then(resp => resp.ok ? resp.json() : null)
                .then(reloadData => {
                  if (reloadData) {
                    const { sessionUpdates, storeUpdates } = transformLoadResponse(reloadData)
                    useAppStore.getState().updateSession(sessionUpdates)
                    applyStoreUpdates(storeUpdates)
                  }
                })
                .catch(reloadErr => console.error('WebSocket: Failed to reload session state:', reloadErr))
            }
          } else {
            // Update state version and session status
            useAppStore.getState().setStateVersion(syncContent.version)

            // Handle LLM failure state (single source of truth — state_sync only)
            if (syncContent.change_type === 'llm_failure' && syncContent.llm_failure_info) {
              const info = syncContent.llm_failure_info
              const store = useAppStore.getState()
              store.setLlmFailureInfo({
                agent: info.agent,
                model: info.model,
                llmServer: info.llm_server,
                error: info.error,
                nodeName: info.node_name,
                generation: info.generation,
                reason: info.reason,
              })
              store.setShowLlmTimeoutDialog(true)
            } else if (
              syncContent.change_type === 'agent_disabled' &&
              syncContent.agent_disabled_info
            ) {
              const info = syncContent.agent_disabled_info
              const store = useAppStore.getState()
              store.setLlmFailureInfo({
                agent: info.agent,
                model: '',
                llmServer: '',
                error: `Agent "${info.agent}" is disabled but required by step "${info.node_name}".`,
                nodeName: info.node_name,
                generation: info.generation,
                reason: 'agent_disabled',
              })
              store.setShowLlmTimeoutDialog(true)
            } else if (syncContent.change_type === 'llm_retry') {
              useAppStore.getState().clearLlmFailureState()
            } else if (syncContent.change_type === 'agent_resumed') {
              useAppStore.getState().clearLlmFailureState()
            } else if (
              syncContent.change_type === 'species_input' ||
              syncContent.change_type === 'gene_species_mismatch'
            ) {
              if (syncContent.species_input_info) {
                const store = useAppStore.getState()
                const info = syncContent.species_input_info
                store.setSpeciesInputInfo({
                  source: info.source,
                  suggestions: Array.isArray(info.suggestions) ? info.suggestions : [],
                  context: info.context || info.pi_assessment || '',
                  species: info.species,
                  sampleGenes: info.sample_genes,
                  piAssessment: info.pi_assessment,
                  likelySpecies: info.likely_species,
                })
                store.setShowSpeciesInputDialog(true)
              }
            } else if (syncContent.change_type === 'snapshot_transaction') {
              const info = syncContent.snapshot_transaction
              if (info) {
                const store = useAppStore.getState()
                store.setSnapshotTransaction(info)

                // Terminal committed events may trigger reconnect if backend directs
                if (info.status === 'committed' && info.reconnect === true) {
                  const { wsConnect: connectWs } = useAppStore.getState()
                  if (connectWs) setTimeout(() => connectWs(), 100)
                }
              }
            } else if (syncContent.change_type === 'snapshot_failure') {
              const failure = syncContent.snapshot_failure
              if (failure) {
                const store = useAppStore.getState()
                store.setSnapshotFailure({
                  failure,
                  allowed_actions: resolveSnapshotFailureActions(failure),
                })
              }
              const txn = syncContent.snapshot_transaction
              if (txn) {
                useAppStore.getState().setSnapshotTransaction(txn)
              }
            } else if (
              syncContent.change_type === 'species_resolved' ||
              syncContent.change_type === 'species_auto_skipped'
            ) {
              useAppStore.getState().clearSpeciesInputState()
            }

            useAppStore.getState().updateSession({
              status: syncContent.status,
              currentRound: syncContent.current_round,
              ...(syncContent.analysis_phase ? { analysisPhase: syncContent.analysis_phase } : {}),
            })
          }
          return  // Don't add state_sync to message stream
        }

        // Handle message_history - batch restore messages (e.g. reconnect to completed session)
        if (data.type === 'message_history') {
          if (data.content?.messages && Array.isArray(data.content.messages)) {
            const serverMessages = data.content.messages.map(transformRawMessage)
            useAppStore.getState().setMessages(serverMessages)
            devLog(`WebSocket: Restored ${serverMessages.length} messages from history`)
          }
          return  // Don't add message_history itself to the message stream
        }

        // Handle parallel_errors messages - structured error tracking
        if (data.type === 'parallel_errors') {
          const errorContent = data.content
          if (errorContent.errors && Array.isArray(errorContent.errors)) {
            for (const err of errorContent.errors) {
              useAppStore.getState().addAgentError({
                id: err.id,
                agent: err.agent as AgentType,
                errorType: err.error_type || 'execution_error',
                message: err.message,
                details: err.details,
                round: err.round,
                timestamp: data.timestamp || new Date().toISOString(),
                resolved: false
              })
            }
          }
          // Also add to message stream for visibility
        }

        if (data.type === 'snapshot_count') {
          const count = data.content?.count ?? 0
          useAppStore.getState().setSnapshotCount(count)

          if (count >= SNAPSHOT_CLEANUP_THRESHOLD && count % SNAPSHOT_REMINDER_INTERVAL === 0) {
            useAppStore.getState().setShowSnapshotCleanupReminder(true)
          }
          return  // Don't add to MessageStream
        }

        // Handle chat_result - final response from chat delegation
        if (data.type === 'chat_result') {
          // Refresh full chat history from backend to ensure consistency
          const sessionId = useAppStore.getState().currentSession?.id
          if (sessionId) {
            apiFetch<{ messages: ChatMessage[] }>(`/session/${sessionId}/chat`)
              .then((res) => {
                useAppStore.getState().setChatMessages(res.messages || [])
              })
              .catch((e) => { devLog('Chat refresh after delegation failed:', e) })
          }
          useAppStore.getState().setChatLoading(false)
          useAppStore.getState().setChatDelegationActive(false)
          useAppStore.getState().setChatDelegationPlan(null)
          return  // Don't add to MessageStream
        }

        // Handle chat_cancelled - delegation was stopped by user
        if (data.type === 'chat_cancelled') {
          useAppStore.getState().setChatLoading(false)
          useAppStore.getState().setChatDelegationActive(false)
          useAppStore.getState().setChatDelegationPlan(null)
          useAppStore.getState().setChatAbortController(null)
          return  // Don't add to MessageStream
        }

        // Handle chat_error - delegation failed
        if (data.type === 'chat_error') {
          useAppStore.getState().addChatMessage({
            role: 'assistant',
            content: `Error: ${data.content?.error || 'Unknown error during analysis'}`,
            timestamp: new Date().toISOString(),
          })
          useAppStore.getState().setChatLoading(false)
          useAppStore.getState().setChatDelegationActive(false)
          useAppStore.getState().setChatDelegationPlan(null)
          return  // Don't add to MessageStream
        }

        // Handle chat_delegation progress - update delegation status AND show in MessageStream
        if (data.type === 'chat_delegation') {
          const delegContent = data.content
          if (delegContent?.status === 'started') {
            useAppStore.getState().setChatDelegationActive(true)
            useAppStore.getState().setChatDelegationPlan(delegContent.plan || null)
          }
          // Fall through to addMessage() so it shows in MessageStream
        }

        // Handle exploration_failed - exploration (Stage 1) did not complete,
        // most likely a data problem. Surface an actionable dialog so the user
        // can review/re-upload data and retry Stage 1, rather than silently
        // advancing rounds toward max_rounds.
        if (data.type === 'exploration_failed') {
          const c = data.content || {}
          const store = useAppStore.getState()
          store.setExplorationFailedInfo({
            round: c.round,
            reason: c.reason || 'incomplete',
            message: c.message || '',
          })
          store.setShowExplorationFailedDialog(true)
          return  // Don't add to MessageStream — the dialog is the UX
        }

        const message: Message = {
          id: data.id || generateId(),
          type: data.type || 'message',
          agent: data.agent as AgentType | undefined,
          content: data.content,
          timestamp: data.timestamp || new Date().toISOString(),
        }

        useAppStore.getState().addMessage(message)
        
        // Update session status based on message
        if (data.type === 'completed') {
          const updates: Partial<Session> = {
            status: 'completed' as const,
            finalSummary: data.content?.final_summary,
          }
          // Extract round_results if provided (for completed session reconnect)
          const roundResults = data.content?.round_results
          if (roundResults && Array.isArray(roundResults)) {
            updates.roundResults = transformRoundResults(roundResults)
          }
          useAppStore.getState().updateSession(updates)
        } else if (data.type === 'error' && !data.agent) {
          // Only mark as failed if it's a system error, not an agent error
          useAppStore.getState().updateSession({ status: 'failed' })
        }
        
        // Update currentRound from status messages
        if (data.type === 'status' && typeof data.content === 'string') {
          // Only match the analysis round marker ("Starting round X analysis").
          // Must NOT match other status text that mentions "round", e.g. the
          // planning dialogue's "Planning dialogue: round 3/5", which would
          // otherwise overwrite currentRound with the dialogue iteration count.
          const roundMatch = data.content.match(/Starting round\s+(\d+)/i)
          if (roundMatch) {
            const round = parseInt(roundMatch[1], 10)
            if (!isNaN(round)) {
              useAppStore.getState().updateSession({ currentRound: round })
            }
          }
        }
        
        // Update from result messages that contain round info
        if (data.type === 'result' && data.content?.round) {
          useAppStore.getState().updateSession({ currentRound: data.content.round })
        }

        // Handle round_summary messages to update roundResults
        if (data.type === 'round_summary' && data.content) {
          const roundData = data.content
          const latestSession = useAppStore.getState().currentSession
          if (latestSession) {
            const existingRounds = latestSession.roundResults || []
            const incomingKey = roundKey(roundData.phase || '', roundData.round)
            const existingIndex = existingRounds.findIndex((r: RoundResult) => (
              roundKey(r.phase, r.round) === incomingKey
            ))

            const newRoundResult = {
              round: roundData.round,
              summary: roundData.summary || '',
              phase: roundData.phase || '',
              pathologyFindings: roundData.pathology_findings || [],
              bioinformaticsFindings: roundData.bioinformatics_findings || [],
              biochemistryFindings: roundData.biochemistry_findings || [],
              literatureFindings: roundData.literature_findings || [],
              crossModalCandidates: roundData.cross_modal_candidates || [],
              integrationCandidates: roundData.integration_candidates || [],
              integrationReportSummary: roundData.integration_report_summary || '',
            }

            let updatedRounds
            if (existingIndex >= 0) {
              // Update existing round
              updatedRounds = [...existingRounds]
              updatedRounds[existingIndex] = newRoundResult
            } else {
              // Add new round
              updatedRounds = [...existingRounds, newRoundResult]
            }

            useAppStore.getState().updateSession({
              roundResults: updatedRounds,
              currentRound: roundData.round,
              status: roundData.status === 'waiting_input' ? 'waiting_input' : latestSession.status,
            })
          }
        }
      } catch (e) {
        console.error('WebSocket: Failed to parse message:', e)
      }
    }
    
    ws.onerror = (error) => {
      clearTimeout(connectTimeout)
      connectingRef.current = false
      console.error('WebSocket: Error:', error)
    }

    ws.onclose = (event) => {
      if (wsRef.current !== ws) return
      clearTimeout(connectTimeout)
      connectingRef.current = false
      setConnected(false)
      setWsConnected(false)
      wsRef.current = null
      devLog('WebSocket: Disconnected, code:', event.code, 'reason:', event.reason)

      // Don't reconnect on auth failure (4401) or if no token
      if (event.code === 4401 || !getToken()) {
        devLog('WebSocket: Auth failure or no token, not reconnecting')
        return
      }

      reconnectCountRef.current += 1  // Track reconnect for history recovery

      // Auto-reconnect if session is still running (not manually disconnected)
      const latestSession = useAppStore.getState().currentSession
      if (latestSession &&
          (latestSession.status === 'running' || latestSession.status === 'waiting_input' || latestSession.status === 'waiting_annotation' || latestSession.status === 'waiting_plan_review') &&
          event.code !== 1000) {  // Not a normal close
        if (reconnectCountRef.current < 5) {  // Max 5 reconnect attempts
          const delay = Math.min(1000 * Math.pow(2, reconnectCountRef.current - 1), 10000)  // Exponential backoff, max 10s
          devLog(`WebSocket: Auto-reconnecting in ${delay}ms (attempt ${reconnectCountRef.current})...`)
          reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null
            if (!wsRef.current && !connectingRef.current) {
              connect()
            }
          }, delay)
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
  }, [currentSession, addMessage, updateSession, setWsConnected, setMessages, addAgentError, setStateVersion])
  
  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    if (wsRef.current) {
      devLog('WebSocket: Disconnecting manually')
      wsRef.current.close()
      wsRef.current = null
    }
    connectingRef.current = false
  }, [])
  
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])
  
  return { connect, disconnect, connected }
}
