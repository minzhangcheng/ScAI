import { useAppStore, Message, Session, AgentType, RoundResult } from '../../../stores/appStore'
import { transformRoundResults } from '../../../utils/transformLoadResponse'
import { devLog } from '../../../utils/logger'
import { resultIdentityKey } from '../../../utils/roundKey'
import type { ChatMessage } from '../../../types'
import { resolveSnapshotFailureActions } from '../../../utils/snapshotV2'
import { reconcileChatHistory, upsertChatMessage } from '../../../utils/chatMessages'
import { apiFetch, generateId } from '../client'
import { mergeRealtimeMessages } from '../realtimeHistory'
import { reloadSessionState } from './sessionReload'
import type { MessageHandlerContext } from './types'

const SNAPSHOT_CLEANUP_THRESHOLD = 20
const SNAPSHOT_REMINDER_INTERVAL = 5
const MAX_RESTORE_COMMIT_REFRESH_KEYS = 64

function chatIdentityFields(value: any): Pick<ChatMessage, 'followup_id' | 'branch_id'> {
  return {
    ...(value?.followup_id ? { followup_id: String(value.followup_id) } : {}),
    ...(value?.branch_id ? { branch_id: String(value.branch_id) } : {}),
  }
}

function handleStateSync(data: any, context: MessageHandlerContext): void {
  const {
    attempt, isCurrentAttempt, restoreCommitRefreshesRef, stateSyncEpochRef,
  } = context
  // Normalise: some backend paths nest fields under `content`
  // while others place them at the top level of the message.
  const syncContent = data.content ?? data
  const restoreInfo = syncContent.change_type === 'snapshot_transaction'
    ? syncContent.snapshot_transaction
    : null
  const restoreSessionId = useAppStore.getState().currentSession?.id
  const restoreRefreshKey = (
    restoreSessionId && restoreInfo?.transaction_id
      ? `${restoreSessionId}:${attempt}:${restoreInfo.transaction_id}`
      : null
  )
  const repeatedRestoreCommit = Boolean(
    restoreInfo?.status === 'committed'
    && restoreInfo.session_id === restoreSessionId
    && restoreRefreshKey
    && restoreCommitRefreshesRef.current.keys.has(restoreRefreshKey)
  )
  // A replayed commit is the same causal sync that already owns the in-flight
  // reload. Advancing the epoch while deduplicating its replacement would
  // invalidate the only authoritative response.
  if (!repeatedRestoreCommit) stateSyncEpochRef.current += 1
  const currentVersion = useAppStore.getState().stateVersion

  const terminalChat = syncContent.chat_message
  if (terminalChat?.id) {
    const currentChat = useAppStore.getState().chatMessages
    useAppStore.getState().setChatMessages(upsertChatMessage(currentChat, {
      id: terminalChat.id,
      role: terminalChat.role === 'user' ? 'user' : 'assistant',
      content: terminalChat.content || '',
      timestamp: terminalChat.timestamp || data.timestamp || new Date().toISOString(),
          pending_figures: terminalChat.pending_figures,
          reply_to: terminalChat.reply_to,
          ...chatIdentityFields(terminalChat),
        }))
  }

  // Check for version conflict - reload full session state from server
  if (syncContent.version < currentVersion) {
    console.warn('WebSocket: Received outdated state sync, current:', currentVersion, 'received:', syncContent.version, '. Reloading session state.')
    const sessionId = useAppStore.getState().currentSession?.id
    if (sessionId) {
      const reloadEpoch = stateSyncEpochRef.current
      const ownsReload = () => (
        isCurrentAttempt()
        && useAppStore.getState().currentSession?.id === sessionId
        && stateSyncEpochRef.current === reloadEpoch
      )
      void reloadSessionState(sessionId, ownsReload).catch(reloadErr => {
        if (ownsReload()) {
          console.error('WebSocket: Failed to reload session state:', reloadErr)
        }
      })
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
          const restoreReloadEpoch = stateSyncEpochRef.current
          const ownsRestoreReload = () => (
            isCurrentAttempt()
            && useAppStore.getState().currentSession?.id === activeSessionId
            && stateSyncEpochRef.current === restoreReloadEpoch
          )
          const releaseRefresh = () => {
            if (restoreCommitRefreshesRef.current.keys.has(refreshKey)) {
              const retained = restoreCommitRefreshesRef.current
              retained.keys.delete(refreshKey)
              retained.order = retained.order.filter(key => key !== refreshKey)
            }
          }
          void reloadSessionState(activeSessionId, ownsRestoreReload).then(applied => {
            if (!applied) releaseRefresh()
          }).catch(reloadErr => {
            if (ownsRestoreReload()) {
              console.error(
                'WebSocket: Failed to reload session state after restore commit:',
                reloadErr,
              )
            }
            releaseRefresh()
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
    if (
      syncContent.runtime_persistence
      && typeof syncContent.runtime_persistence === 'object'
    ) {
      sessionUpdate.runtimePersistence = syncContent.runtime_persistence
    }
    if (Object.keys(sessionUpdate).length > 0) {
      useAppStore.getState().updateSession(sessionUpdate)
    }
  }
  return  // Don't add state_sync to message stream
}

export function handleWebSocketMessage(
  event: MessageEvent,
  context: MessageHandlerContext,
): void {
  const {
    session, ws, isCurrentAttempt, ownsSocketSession,
  } = context
  // A store selection can move from A to B before B creates its socket.
  // Socket-attempt ownership alone still regards A as current in that
  // window, so every global-store mutation must also remain session-owned.
  if (!ownsSocketSession()) return
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
      handleStateSync(data, context)
      return
    }

    // Handle message_history - batch restore messages (e.g. reconnect to completed session)
    if (data.type === 'message_history') {
      if (data.content?.messages && Array.isArray(data.content.messages)) {
        mergeRealtimeMessages(data.content.messages)
        devLog(`WebSocket: Merged ${data.content.messages.length} legacy history messages`)
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
      const terminal = data.content || {}
      if (terminal.id) {
        const current = useAppStore.getState().chatMessages
        useAppStore.getState().setChatMessages(upsertChatMessage(current, {
          id: terminal.id,
          role: 'assistant',
          content: terminal.content || '',
          timestamp: terminal.timestamp || data.timestamp || new Date().toISOString(),
          figures: terminal.figures,
          pending_figures: terminal.pending_figures,
          reply_to: terminal.reply_to,
          ...chatIdentityFields(terminal),
        }))
      }
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
            const current = useAppStore.getState().chatMessages
            useAppStore.getState().setChatMessages(
              reconcileChatHistory(res.messages || [], current),
            )
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
      const terminal = data.content || {}
      if (terminal.id) {
        const current = useAppStore.getState().chatMessages
        useAppStore.getState().setChatMessages(upsertChatMessage(current, {
          id: terminal.id,
          role: 'assistant',
          content: terminal.content || terminal.message || 'Cancelled',
          timestamp: data.timestamp || new Date().toISOString(),
          pending_figures: false,
          reply_to: terminal.reply_to,
          ...chatIdentityFields(terminal),
        }))
      }
      useAppStore.getState().setChatLoading(false)
      useAppStore.getState().setChatDelegationActive(false)
      useAppStore.getState().setChatDelegationPlan(null)
      useAppStore.getState().setChatAbortController(null)
      return  // Don't add to MessageStream
    }

    // Handle chat_error - delegation failed
    if (data.type === 'chat_error') {
      const terminal = data.content || {}
      if (terminal.id) {
        const current = useAppStore.getState().chatMessages
        useAppStore.getState().setChatMessages(upsertChatMessage(current, {
          id: terminal.id,
          role: 'assistant',
          content: terminal.content || `Error: ${terminal.error || 'Unknown error during analysis'}`,
          timestamp: data.timestamp || new Date().toISOString(),
          pending_figures: false,
          reply_to: terminal.reply_to,
          ...chatIdentityFields(terminal),
        }))
      } else {
          useAppStore.getState().addChatMessage({
            role: 'assistant',
            content: `Error: ${terminal.error || 'Unknown error during analysis'}`,
            timestamp: new Date().toISOString(),
            ...chatIdentityFields(terminal),
          })
      }
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
          status: roundData.wait_policy === 'no_wait'
            ? 'running'
            : roundData.status === 'waiting_input'
              ? 'waiting_input'
              : latestSession.status,
        })
      }
    }
  } catch (e) {
    console.error('WebSocket: Failed to parse message:', e)
  }
}
