import { useAppStore, Message, Session, AgentType, RoundResult } from '../../../stores/appStore'
import { chatOwnerKey } from '../../../utils/chatOwnership'
import { transformRoundResults } from '../../../utils/transformLoadResponse'
import { devLog } from '../../../utils/logger'
import { resultIdentityKey } from '../../../utils/roundKey'
import type { ChatMessage } from '../../../types'
import { reconcileChatHistory, upsertChatMessage } from '../../../utils/chatMessages'
import { apiFetch, generateId } from '../client'
import { mergeRealtimeMessages } from '../realtimeHistory'
import { notifySnapshotListingChanged } from '../snapshots'
import { reloadSessionState } from './sessionReload'
import type { MessageHandlerContext } from './types'

const contextRefreshes = new WeakMap<WebSocket, Map<AgentType, symbol>>()

function ownsHandlerSession(
  context: MessageHandlerContext,
  sessionId: string,
): boolean {
  const current = useAppStore.getState().currentSession
  return context.isCurrentAttempt()
    && current?.id === sessionId
    && (
      context.ownerKey === undefined
      || chatOwnerKey(current.id, current.restoreBranchId) === context.ownerKey
    )
}

function refreshContextBudget(agent: AgentType, context: MessageHandlerContext): void {
  const store = useAppStore.getState()
  const session = store.currentSession
  const config = store.agentServerConfigs[agent]
  const model = store.models[agent]
  const apiConfig = store.apiConfig
  const token = Symbol()
  const requests = contextRefreshes.get(context.ws) ?? new Map<AgentType, symbol>()
  contextRefreshes.set(context.ws, requests)
  requests.set(agent, token)
  const ownsResponse = () => {
    const current = useAppStore.getState()
    return context.isCurrentAttempt() && context.ownsSocketSession()
      && current.currentSession === session
      && current.agentServerConfigs[agent] === config
      && current.models[agent] === model
      && current.apiConfig === apiConfig
      && requests.get(agent) === token
  }
  void apiFetch<Record<string, { effective_context_window?: number }>>(
    `/session/${context.session.id}/agent-server-config`, undefined, ownsResponse,
  ).then(data => {
    const effective = data?.[agent]?.effective_context_window
    if (!ownsResponse() || typeof effective !== 'number') return
    const current = useAppStore.getState()
    current.setAgentServerConfig(agent, {
      llmBaseUrl: '', apiKey: '', ...config, effectiveContextWindow: effective,
    })
  }).catch(error => { if (ownsResponse()) devLog('Context budget refresh failed:', error) })
}

function chatIdentityFields(value: any): Pick<ChatMessage, 'followup_id' | 'branch_id'> {
  return {
    ...(value?.followup_id ? { followup_id: String(value.followup_id) } : {}),
    ...(value?.branch_id ? { branch_id: String(value.branch_id) } : {}),
  }
}

function handleStateSync(data: any, context: MessageHandlerContext): void {
  const { stateSyncEpochRef } = context
  // Normalise: some backend paths nest fields under `content`
  // while others place them at the top level of the message.
  const syncContent = data.content ?? data
  stateSyncEpochRef.current += 1
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
        ownsHandlerSession(context, sessionId)
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
        failureKind: typeof info.failure_kind === 'string' ? info.failure_kind : undefined,
        nodeName: info.node_name,
        generation: info.generation,
        waitSeconds: typeof info.wait_seconds === 'number' ? info.wait_seconds : undefined,
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
    } else if (
      syncContent.change_type === 'llm_retry' ||
      syncContent.change_type === 'llm_skip' ||
      syncContent.change_type === 'llm_terminate'
    ) {
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
    } else if (
      syncContent.change_type === 'species_input_received' ||
      syncContent.change_type === 'species_resolved' ||
      syncContent.change_type === 'species_auto_skipped'
    ) {
      useAppStore.getState().clearSpeciesInputState()
    }

    // Only write fields the message actually carries. `updateSession`
    // spreads this object over the session, so a key present with value
    // `undefined` erases the real value rather than being ignored — and
    // not every state_sync producer sends the full shape.
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

function handleExplorationReport(data: any): void {
  useAppStore.getState().updateSession({
    explorationReportUrl: data.content?.html_url ?? '',
  })
}

export function handleWebSocketMessage(
  event: MessageEvent,
  context: MessageHandlerContext,
): void {
  const {
    session, ws, ownsSocketSession,
  } = context
  // A store selection can move from A to B before B creates its socket.
  // Socket-attempt ownership alone still regards A as current in that
  // window, so every global-store mutation must also remain session-owned.
  if (!ownsSocketSession()) return
  try {
    const data = JSON.parse(event.data)

    // Server heartbeat ping — ignore silently
    if (data.type === 'ping') return

    if (data.type === 'agent_context_updated') {
      if (data.session_id === session.id && typeof data.agent === 'string') {
        refreshContextBudget(data.agent as AgentType, context)
      }
      return
    }

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

    // Snapshot listings recursively account for filesystem occupancy, so the
    // capture producer invalidates them once instead of every browser polling.
    if (data.type === 'snapshot_listing_changed') {
      notifySnapshotListingChanged(session.id)
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
          chat_review_ids: terminal.chat_review_ids,
          execution_incomplete: terminal.execution_incomplete,
          reply_to: terminal.reply_to,
          ...chatIdentityFields(terminal),
        }))
      }
      // Refresh full chat history from backend to ensure consistency
      const sessionId = useAppStore.getState().currentSession?.id
      if (sessionId) {
        const ownsChatRefresh = () => ownsHandlerSession(context, sessionId)
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

    if (data.type === 'exploration_report') {
      handleExplorationReport(data)
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
