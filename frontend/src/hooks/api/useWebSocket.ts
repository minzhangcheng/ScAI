// WebSocket connection, message dispatch, and snapshot reminders.
// Split out of frontend/src/hooks/useApi.ts verbatim.

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAppStore, Message, Session, AgentType, RoundResult } from '../../stores/appStore'
import { transformRawMessage, restoreAgentErrors } from '../../utils/messageTransform'
import { transformLoadResponse, transformRoundResults } from '../../utils/transformLoadResponse'
import { devLog } from '../../utils/logger'
import { resultIdentityKey } from '../../utils/roundKey'
import { getToken } from '../../utils/authStorage'
import { WS_AUTH_SUBPROTOCOL } from '../../constants/auth'
import type { ChatMessage } from '../../types'
import { resolveSnapshotFailureActions } from '../../utils/snapshotV2'
import { WS_CONNECT_TIMEOUT_MS, ApiError, apiFetch, applyStoreUpdates, generateId } from './client'

const SNAPSHOT_CLEANUP_THRESHOLD = 20
const SNAPSHOT_REMINDER_INTERVAL = 5
const MAX_RESTORE_COMMIT_REFRESH_KEYS = 64
const RECONNECTABLE_SESSION_STATUSES = new Set<Session['status']>([
  'running',
  'waiting_input',
  'waiting_annotation',
  'waiting_plan_review',
  'waiting_species_input',
  'generating_report',
])

/**
 * Refetch `/session/{id}/load` and apply it to the store.
 *
 * The response owner must still own both the session and WebSocket generation
 * when the request completes. This also guards lifecycle side effects inside
 * `apiFetch` itself.
 */
function reloadSessionState(
  sessionId: string,
  errorLabel: string,
  ownsResponse: () => boolean,
): Promise<boolean> {
  return apiFetch<any>(
    `/session/${sessionId}/load`,
    undefined,
    ownsResponse,
  )
    .then(reloadData => {
      if (!reloadData || typeof reloadData !== 'object') return false
      if (
        !ownsResponse()
        || useAppStore.getState().currentSession?.id !== sessionId
      ) return false
      const { sessionUpdates, storeUpdates } = transformLoadResponse(reloadData)
      useAppStore.getState().updateSession(sessionUpdates)
      applyStoreUpdates(storeUpdates)
      return true
    })
    .catch(reloadErr => {
      if (ownsResponse()) console.error(errorLabel, reloadErr)
      return false
    })
}

export function useWebSocket() {
  const { currentSession, addMessage, updateSession, setWsConnected, setMessages, addAgentError, setStateVersion } = useAppStore()
  const wsRef = useRef<WebSocket | null>(null)
  const connectionAttemptRef = useRef(0)
  const connectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectCountRef = useRef<number>(0)  // Track reconnection attempts for history recovery
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const restoreCommitRefreshesRef = useRef<{
    keys: Set<string>
    order: string[]
  }>({ keys: new Set(), order: [] })
  const [connected, setConnected] = useState(false)

  const connect = useCallback(() => {
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
      clearCurrentConnectTimeout()
      setConnected(true)
      setWsConnected(true)
      devLog('WebSocket: Connected successfully')

      // Use fresh state to avoid stale closure
      const freshSession = useAppStore.getState().currentSession

      // On reconnect, restore message history from server
      if (isReconnect && freshSession) {
        const loadingSessionId = freshSession.id
        try {
          devLog('WebSocket: Reconnect detected, loading message history...')
          // Mark that we're loading history to prevent race conditions
          const ownsReconnectLoad = () => (
            isCurrentAttempt()
            && useAppStore.getState().currentSession?.id === loadingSessionId
          )
          const data = await apiFetch<any>(
            `/session/${freshSession.id}/load`,
            undefined,
            ownsReconnectLoad,
          )

          if (!isCurrentAttempt()) return

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
          console.error('WebSocket: Failed to restore message history:', e)
        }
      }

      // Reset reconnect count on successful connection
      if (isCurrentAttempt()) reconnectCountRef.current = 0
    }
    
    ws.onmessage = (event) => {
      if (!isCurrentAttempt()) return
      try {
        const data = JSON.parse(event.data)

        // Server heartbeat ping — ignore silently
        if (data.type === 'ping') return

        if (
          data.type === 'session_error' &&
          (data.code === 'SESSION_NOT_FOUND' || data.code === 'SESSION_GONE')
        ) {
          const current = useAppStore.getState().currentSession
          if (data.session_id === session.id && current?.id === session.id) {
            useAppStore.getState().setCurrentSession(null)
            useAppStore.getState().clearMessages()
          }
          ws.close()
          return
        }

        // Handle error messages that indicate session/config issues
        if (data.error) {
          console.error('WebSocket: Server error:', data.error)
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
              const ownsReload = () => (
                isCurrentAttempt()
                && useAppStore.getState().currentSession?.id === sessionId
              )
              void reloadSessionState(
                sessionId,
                'WebSocket: Failed to reload session state:',
                ownsReload,
              )
            }
          } else {
            // Update state version and session status. Producers that omit
            // `version` (the restore-job transaction sync) must not reset the
            // tracked version to undefined — that would disable the
            // conflict-reload branch above for every later message.
            if (typeof syncContent.version === 'number') {
              useAppStore.getState().setStateVersion(syncContent.version)
            }

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

                // A committed restore has already rewritten session.json, the
                // notebook, the workspace and the graph checkpoint on the
                // server, and this sync deliberately carries no `status` /
                // `current_round` / `version`, so the partial-update block below
                // writes nothing. The refresh used to be gated on
                // `info.reconnect`, a field no producer emits —
                // SnapshotTransactionRecord has no such key — so whenever the
                // Snapshot panel was unmounted at commit time (page reloaded
                // mid-restore: the panel only auto-reopens for a *retry*
                // transaction) the client kept rendering pre-restore state and
                // dispatched the user's next action against a server that had
                // moved. Refetch rather than reconnect: it is idempotent and
                // does not tear down a live socket. Guarded per transaction id
                // because `_resend_wait_input_state_sync` replays the stored
                // transaction on every reconnect.
                const activeSessionId = store.currentSession?.id
                const refreshKey = (
                  activeSessionId && info.transaction_id
                    ? `${activeSessionId}:${attempt}:${info.transaction_id}`
                    : null
                )
                if (
                  info.status === 'committed'
                  && activeSessionId
                  && info.session_id === activeSessionId
                  && refreshKey
                  && !restoreCommitRefreshesRef.current.keys.has(refreshKey)
                ) {
                  const refreshes = restoreCommitRefreshesRef.current
                  refreshes.keys.add(refreshKey)
                  refreshes.order.push(refreshKey)
                  while (refreshes.order.length > MAX_RESTORE_COMMIT_REFRESH_KEYS) {
                    const expired = refreshes.order.shift()
                    if (expired) refreshes.keys.delete(expired)
                  }
                  const ownsRestoreReload = () => (
                    isCurrentAttempt()
                    && useAppStore.getState().currentSession?.id === activeSessionId
                  )
                  void reloadSessionState(
                    activeSessionId,
                    'WebSocket: Failed to reload session state after restore commit:',
                    ownsRestoreReload,
                  ).then(applied => {
                    if (
                      !applied
                      && restoreCommitRefreshesRef.current.keys.has(refreshKey)
                    ) {
                      const retained = restoreCommitRefreshesRef.current
                      retained.keys.delete(refreshKey)
                      retained.order = retained.order.filter(key => key !== refreshKey)
                    }
                  })
                }
              }
            } else if (
              syncContent.change_type === 'snapshot_failure' ||
              syncContent.change_type === 'runtime_persistence_failure'
            ) {
              const failure = syncContent.snapshot_failure
              if (failure) {
                const store = useAppStore.getState()
                store.setSnapshotFailure({
                  failure,
                  allowed_actions: resolveSnapshotFailureActions(failure),
                })
              } else if (failure === null) {
                useAppStore.getState().clearSnapshotFailure()
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

            // Only write fields the message actually carries. `updateSession`
            // spreads this object over the session, so a key present with value
            // `undefined` erases the real value rather than being ignored — and
            // not every state_sync producer sends the full shape. The restore
            // job's `snapshot_transaction` sync in particular sends only
            // `change_type` + `snapshot_transaction`, so every progress event
            // during a restore used to blank `status` and `currentRound`,
            // leaving the Analysis Control header with no matching branch and
            // therefore no buttons at all until a manual page reload.
            const sessionUpdate: Partial<Session> = {}
            if (syncContent.status !== undefined) sessionUpdate.status = syncContent.status
            if (syncContent.current_round !== undefined) sessionUpdate.currentRound = syncContent.current_round
            if (syncContent.analysis_phase) sessionUpdate.analysisPhase = syncContent.analysis_phase
            if (Object.keys(sessionUpdate).length > 0) {
              useAppStore.getState().updateSession(sessionUpdate)
            }
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
            const ownsChatRefresh = () => (
              isCurrentAttempt()
              && useAppStore.getState().currentSession?.id === sessionId
            )
            apiFetch<{ messages: ChatMessage[] }>(
              `/session/${sessionId}/chat`,
              undefined,
              ownsChatRefresh,
            )
              .then((res) => {
                if (!ownsChatRefresh()) return
                useAppStore.getState().setChatMessages(res.messages || [])
              })
              .catch((e) => {
                if (ownsChatRefresh()) {
                  devLog('Chat refresh after delegation failed:', e)
                }
              })
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

            // Use the same DTO transformer as reload/reconnect.  Maintaining a
            // second hand-written mapping here caused evidence and lineage
            // fields to disappear until the next page refresh.
            const newRoundResult = transformRoundResults([roundData])[0]
            if (!newRoundResult) return
            const incomingKey = resultIdentityKey(newRoundResult)
            const existingIndex = existingRounds.findIndex((r: RoundResult) => (
              resultIdentityKey(r) === incomingKey
            ))

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
      if (!isCurrentAttempt()) return
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
              connect()
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
  }, [currentSession, addMessage, updateSession, setWsConnected, setMessages, addAgentError, setStateVersion])
  
  const disconnect = useCallback(() => {
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
  }, [setWsConnected])
  
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])
  
  return { connect, disconnect, connected }
}
