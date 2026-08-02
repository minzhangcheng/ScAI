import { useEffect, useState, useMemo, useCallback, useRef } from 'react'
import { useAppStore, type TaskStatus } from '../stores/appStore'
import { useSession, useConfig, useUserConfig, generateId, apiFetch, getSnapshotTransaction } from '../hooks/useApi'
import { useTranslation } from '../i18n/translations'
import type { RoundSummary } from '../types'
import { RoundSummaries } from './analysis/RoundSummaries'
import type { SnapshotFailureAction, SnapshotTransactionRecord } from '../types/snapshotV2'
import { resultIdentityKey } from '../utils/roundKey'
import type { AnnotationData } from '../utils/annotationGate'
import ReactMarkdown from 'react-markdown'
import { RoundFeedbackDialog, UploadedFileWithAgent } from './RoundFeedbackDialog'
import { AnnotationConfirmationDialog } from './AnnotationConfirmationDialog'
import { ExplorationReportDialog, ExplorationReport } from './ExplorationReportDialog'
import { IntegrationGatePanel } from './IntegrationGatePanel'
import { PlanReviewDialog, AnalysisPlan } from './PlanReviewDialog'
import { ValidationDialog } from './ValidationDialog'
import { StopDialog } from './StopDialog'
import LlmTimeoutDialog from './LlmTimeoutDialog'
import LlmModifySubDialog from './LlmModifySubDialog'
import ExplorationFailedDialog from './ExplorationFailedDialog'
import { useCountdownTimer } from '../hooks/useCountdownTimer'
import { SnapshotPanel } from './SnapshotPanel'
import { SnapshotFailureDialog } from './snapshots/SnapshotFailureDialog'
import { RuntimeWarningBanner } from './snapshots/RuntimeWarningBanner'
import { TextInputDialog } from './TextInputDialog'
import { ConfirmDialog } from './ConfirmDialog'
import {
  Play,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Users,
  Microscope,
  Code2,
  FlaskConical,
  BookOpen,
  Palette,
  Eye,
  Wrench,
  Pause,
  Square,
  Camera,
  Trash2
} from 'lucide-react'

const API_BASE = '/api'
const MAX_EXPLORATION_ROUND = 1
const SNAPSHOT_RETRY_POLL_INTERVAL_MS = 2_000
const SNAPSHOT_RETRY_MAX_POLLS = 150
const SNAPSHOT_TERMINAL_STATUSES = new Set<SnapshotTransactionRecord['status']>([
  'committed',
  'failed',
  'cancelled',
  'timed_out',
])

// Content shape for plan_review HITL message from backend
interface PlanReviewData {
  plan?: AnalysisPlan | null
  dialogue_html?: string
  phase?: string
  round?: number
  stage?: string
  wait_seconds?: number
}

// Content shape for exploration_report HITL message from backend
interface ExplorationReportData extends Partial<ExplorationReport> {
  report_a?: ExplorationReport
  round_summary?: { summary?: string }
  html_url?: string
  detailed_report_url?: string
}

// Content shape from backend round_summary messages (snake_case)
interface RoundSummaryContent {
  round?: number
  max_rounds?: number
  summary?: string
  findings?: string[]
  data_requests?: { pathology?: string[]; biochemistry?: string[]; other?: string[] }
  status?: string
  pathology_findings?: string[]
  bioinformatics_findings?: string[]
  biochemistry_findings?: string[]
  literature_findings?: string[]
  phase?: string
  integration_candidates?: { direction: string; score?: number; notes?: string; [key: string]: unknown }[]
  integration_report_summary?: string
  lineage_id?: string
  direction?: string
  direction_key?: string
  result_identity_key?: string
}

export function buildDirectionsPayload(
  directions: Array<{ direction: string; notes: string; isCustom?: boolean }>,
) {
  return directions.map(d => ({
    direction: d.direction,
    notes: d.notes ?? '',
    is_custom: !!d.isCustom,
  }))
}

export function AnalysisPanel() {
  const { currentSession, messages, language, showConfigPanel, wsConnected, wsConnect, agentErrors, readMode, addMessage, authUser } = useAppStore()
  const { startAnalysis, resumeAnalysis, continueRound, pauseAnalysis, stopAnalysis } = useSession()
  const { checkStatus, validateApiKey, validateSavedConfig, saveConfig } = useConfig()
  const userConfig = useUserConfig()
  const { t } = useTranslation(language)

  // LLM failure state selectors
  const llmFailureInfo = useAppStore(s => s.llmFailureInfo)
  const showLlmTimeoutDialog = useAppStore(s => s.showLlmTimeoutDialog)
  const setShowLlmTimeoutDialog = useAppStore(s => s.setShowLlmTimeoutDialog)
  const llmModifySaved = useAppStore(s => s.llmModifySaved)
  const roundWaitSeconds = useAppStore(s => s.roundWaitSeconds)
  const maxRounds = useAppStore(s => s.maxRounds)

  // Authoritative annotation gate from /load; the fallback when the
  // annotation_confirmation message is not (or no longer) in the stream.
  const annotationGateData = useAppStore(s => s.annotationGateData)

  const speciesInputInfo = useAppStore(s => s.speciesInputInfo)
  const showSpeciesInputDialog = useAppStore(s => s.showSpeciesInputDialog)
  const clearSpeciesInputState = useAppStore(s => s.clearSpeciesInputState)

  // Exploration failed user intervention
  const explorationFailedInfo = useAppStore(s => s.explorationFailedInfo)
  const showExplorationFailedDialog = useAppStore(s => s.showExplorationFailedDialog)
  const clearExplorationFailedState = useAppStore(s => s.clearExplorationFailedState)

  // Round results state
  const [expandedRound, setExpandedRound] = useState<string | null>(null)

  // Round feedback dialog state
  const [showFeedbackDialog, setShowFeedbackDialog] = useState(false)
  const [feedbackRoundData, setFeedbackRoundData] = useState<RoundSummary | null>(null)
  const lastDialogRoundRef = useRef<number>(-1)

  // Annotation confirmation dialog state
  const [showAnnotationDialog, setShowAnnotationDialog] = useState(false)
  const [annotationData, setAnnotationData] = useState<AnnotationData | null>(null)
  const [showPlanReviewDialog, setShowPlanReviewDialog] = useState(false)
  const [planReviewData, setPlanReviewData] = useState<PlanReviewData | null>(null)
  const lastAnnotationMsgRef = useRef<string>('')
  const lastPlanReviewMsgRef = useRef<string>('')

  // Exploration report dialog state
  const [showExplorationDialog, setShowExplorationDialog] = useState(false)
  const [explorationReportData, setExplorationReportData] = useState<ExplorationReportData | null>(null)
  const lastExplorationMsgRef = useRef<string>('')

  // Integration gate state
  const [integrationGateData, setIntegrationGateData] = useState<any>(null)
  const [showIntegrationGateDialog, setShowIntegrationGateDialog] = useState(false)
  const lastIntegrationMsgRef = useRef<string>('')

  // Supplementary analysis state
  const [isSupplementaryRunning, setIsSupplementaryRunning] = useState(false)
  // The annotation and direction gates re-open on a fresh gate message once their
  // supplementary pass finishes. The round gate has no such message — the workflow
  // simply returns to the same wait — so latch the request and re-arm on status.
  const roundSupplementaryPendingRef = useRef(false)

  // Snapshot panel state
  const [showSnapshotPanel, setShowSnapshotPanel] = useState(false)
  const snapshotV2Enabled = useAppStore(state => state.snapshotV2Enabled)
  const snapshotRetryPollRef = useRef<{
    transactionId: string
    cancelled: boolean
    attempts: number
    timer: ReturnType<typeof setTimeout> | null
  } | null>(null)
  const snapshotAutoOpenedForRef = useRef<string | null>(null)

  // Snapshot cleanup reminder state
  const snapshotCount = useAppStore(state => state.snapshotCount)
  const showSnapshotCleanupReminder = useAppStore(state => state.showSnapshotCleanupReminder)
  const setShowSnapshotCleanupReminder = useAppStore(state => state.setShowSnapshotCleanupReminder)

  // Snapshot V2 failure / runtime restore safety
  const snapshotFailure = useAppStore(s => s.snapshotFailure)
  const clearSnapshotFailure = useAppStore(s => s.clearSnapshotFailure)
  const runtimeRestoreStatus = useAppStore(s => s.runtimeRestoreStatus)
  const runtimeRestoreWarning = useAppStore(s => s.runtimeRestoreWarning)
  const isForwardResumeBlocked = useAppStore(s => s.isForwardResumeBlocked)
  const snapshotTransaction = useAppStore((state) => state.snapshotTransaction)

  const stopSnapshotRetryPolling = useCallback(() => {
    const poll = snapshotRetryPollRef.current
    if (poll) {
      poll.cancelled = true
      if (poll.timer) clearTimeout(poll.timer)
    }
    snapshotRetryPollRef.current = null
  }, [])

  const monitorSnapshotRetry = useCallback((
    sessionId: string,
    seed: SnapshotTransactionRecord,
  ) => {
    const activePoll = snapshotRetryPollRef.current
    if (
      activePoll?.transactionId === seed.transaction_id &&
      !activePoll.cancelled
    ) {
      return
    }
    stopSnapshotRetryPolling()
    useAppStore.getState().setSnapshotTransaction(seed)
    const pollState = {
      transactionId: seed.transaction_id,
      cancelled: false,
      attempts: 0,
      timer: null as ReturnType<typeof setTimeout> | null,
    }
    snapshotRetryPollRef.current = pollState

    const poll = async () => {
      if (pollState.cancelled) return
      const current = useAppStore.getState().snapshotTransaction
      if (current?.transaction_id !== pollState.transactionId) {
        stopSnapshotRetryPolling()
        return
      }
      if (SNAPSHOT_TERMINAL_STATUSES.has(current.status)) {
        stopSnapshotRetryPolling()
        return
      }
      if (pollState.attempts >= SNAPSHOT_RETRY_MAX_POLLS) {
        // 150 × 2s is exactly the backend's 300s safe-point timeout, so a
        // restore that waits near it and then spends time applying outlives
        // this loop. Stopping silently froze the progress dialog on its last
        // status with a live Cancel button, and when the restore did commit
        // nothing ever noticed. Say so instead: the transaction is unchanged
        // on the server, and reopening the Snapshot panel resumes tracking.
        console.warn(
          'Lost contact with restore transaction', pollState.transactionId,
          '— stopped polling after', SNAPSHOT_RETRY_MAX_POLLS, 'attempts',
        )
        const stalled = useAppStore.getState().snapshotTransaction
        if (stalled && stalled.transaction_id === pollState.transactionId) {
          useAppStore.getState().setSnapshotTransaction({
            ...stalled,
            poll_timed_out: true,
          } as SnapshotTransactionRecord)
        }
        stopSnapshotRetryPolling()
        return
      }
      pollState.attempts += 1
      try {
        const transaction = await getSnapshotTransaction(sessionId, seed.transaction_id)
        if (pollState.cancelled) return
        const latest = useAppStore.getState().snapshotTransaction
        if (latest?.transaction_id !== pollState.transactionId) {
          stopSnapshotRetryPolling()
          return
        }
        useAppStore.getState().setSnapshotTransaction(transaction)
        if (SNAPSHOT_TERMINAL_STATUSES.has(transaction.status)) {
          stopSnapshotRetryPolling()
          return
        }
      } catch (error) {
        // The subsystem was turned off under a tab that was open before the
        // restart. Every further poll is a guaranteed 404, so stop and correct
        // the stale capability flag instead of retrying until the tab closes.
        // Matched on the error code rather than `instanceof ApiError` so the
        // check survives a mocked api module and a second copy of the class.
        if ((error as { code?: string } | null)?.code === 'SNAPSHOT_V2_DISABLED') {
          useAppStore.getState().setSnapshotV2Enabled(false)
          stopSnapshotRetryPolling()
          return
        }
        console.error('Failed to poll snapshot retry transaction:', error)
      }
      if (!pollState.cancelled) {
        pollState.timer = setTimeout(poll, SNAPSHOT_RETRY_POLL_INTERVAL_MS)
      }
    }

    void poll()
  }, [stopSnapshotRetryPolling])

  useEffect(() => {
    snapshotAutoOpenedForRef.current = null
    return stopSnapshotRetryPolling
  }, [stopSnapshotRetryPolling, currentSession?.id])

  // Both effects below are driven by a record they themselves cause to be
  // rewritten: the poller calls `setSnapshotTransaction` on every tick and
  // `setSnapshotTransaction` always allocates a new object, so a dependency on
  // `snapshotTransaction` fires roughly twice a second for the whole restore.
  // That turned each effect against its own purpose — the monitor tore down and
  // re-seeded the loop it was supposed to own (attempts back to 0, so
  // SNAPSHOT_RETRY_MAX_POLLS could never be reached and the give-up branch
  // instead became a 300s restart cycle), and the auto-open re-asserted itself
  // within a tick of every close, making the full-screen panel undismissable.
  // Depend on the scalars that actually decide whether monitoring or opening is
  // needed, and read the record itself from the store when it is.
  const snapshotTxnId = snapshotTransaction?.transaction_id ?? null
  const snapshotTxnSessionId = snapshotTransaction?.session_id ?? null
  const snapshotTxnIsRetry = Boolean(snapshotTransaction?.retry_of_transaction_id)
  const snapshotTxnTerminal = snapshotTransaction
    ? SNAPSHOT_TERMINAL_STATUSES.has(snapshotTransaction.status)
    : true
  const currentSessionId = currentSession?.id ?? null

  useEffect(() => {
    if (!snapshotV2Enabled) return
    if (!currentSessionId || !snapshotTxnId) return
    if (snapshotTxnSessionId !== currentSessionId) return
    if (snapshotTxnTerminal) {
      stopSnapshotRetryPolling()
      return
    }
    const seed = useAppStore.getState().snapshotTransaction
    if (!seed || seed.transaction_id !== snapshotTxnId) return
    monitorSnapshotRetry(currentSessionId, seed)
  }, [
    currentSessionId,
    snapshotV2Enabled,
    snapshotTxnId,
    snapshotTxnSessionId,
    snapshotTxnTerminal,
    monitorSnapshotRetry,
    stopSnapshotRetryPolling,
  ])

  useEffect(() => {
    if (!snapshotV2Enabled) return
    if (!currentSessionId || !snapshotTxnId) return
    if (snapshotTxnSessionId !== currentSessionId) return
    if (!snapshotTxnIsRetry) return
    // Terminal covers more than 'committed': a failed/cancelled/timed_out retry
    // must not pop the modal back up after the user has dismissed it either.
    if (snapshotTxnTerminal) return
    // Edge-triggered per transaction, so a close stays closed.
    if (snapshotAutoOpenedForRef.current === snapshotTxnId) return
    snapshotAutoOpenedForRef.current = snapshotTxnId
    setShowSnapshotPanel(true)
  }, [
    currentSessionId,
    snapshotTxnId,
    snapshotTxnSessionId,
    snapshotTxnIsRetry,
    snapshotTxnTerminal,
    snapshotV2Enabled,
  ])

  useEffect(() => {
    void checkStatus()
  }, [checkStatus])

  useEffect(() => {
    if (snapshotV2Enabled) return
    setShowSnapshotPanel(false)
    stopSnapshotRetryPolling()
    useAppStore.getState().setShowSnapshotCleanupReminder(false)
  }, [snapshotV2Enabled, stopSnapshotRetryPolling])
  const isRerunEnabled = useAppStore(s => s.isRerunEnabled)

  // Reset message refs when session changes so dialogs can re-trigger after page reload
  useEffect(() => {
    lastAnnotationMsgRef.current = ''
    lastExplorationMsgRef.current = ''
  }, [currentSession?.id])

  // Stop dialog state
  const [showStopDialog, setShowStopDialog] = useState(false)
  const [stopping, setStopping] = useState(false)

  // API validation state
  const [validating, setValidating] = useState(false)
  const [showValidationDialog, setShowValidationDialog] = useState(false)
  const [validationMessage, setValidationMessage] = useState('')
  const [validationUncertain, setValidationUncertain] = useState(false)

  const agents = [
    { id: 'pi', name: t('piAgent'), icon: Users, color: 'violet' },
    { id: 'pathology', name: t('pathologyAgent'), icon: Microscope, color: 'red' },
    { id: 'bioinformatics', name: t('bioinformaticsAgent'), icon: Code2, color: 'green' },
    { id: 'biochemistry', name: t('biochemistryAgent'), icon: FlaskConical, color: 'amber' },
    { id: 'literature', name: t('literatureAgent'), icon: BookOpen, color: 'blue' },
    { id: 'painter', name: t('painterAgent'), icon: Palette, color: 'pink' },
    { id: 'supervisor', name: t('supervisorAgent'), icon: Eye, color: 'cyan' },
    { id: 'technique', name: t('techniqueAgent'), icon: Wrench, color: 'teal' },
  ]

  // Memoize unresolved errors to avoid recalculation
  const unresolvedErrors = useMemo(
    () => agentErrors.filter(e => !e.resolved),
    [agentErrors]
  )

  // Memoize per-agent last message lookup to avoid O(n) filter per agent per render
  const agentLastMessages = useMemo(() => {
    const map: Record<string, typeof messages[0] | undefined> = {}
    for (const msg of messages) {
      if (msg.agent) {
        map[msg.agent] = msg
      }
    }
    return map
  }, [messages])

  // Get latest status for each agent - balances persistent errors with recovery
  const getAgentStatus = (agentId: string) => {
    const lastMessage = agentLastMessages[agentId]
    const agentErrors = unresolvedErrors.filter(e => e.agent === agentId)

    if (!lastMessage) {
      return agentErrors.length > 0 ? 'error' : 'idle'
    }

    // Session completed → all agents with messages show as done
    if (currentSession?.status === 'completed') {
      return 'done'
    }

    // Report generation phase → analysis agents stay done, only PI/Painter may be working
    if (currentSession?.status === 'generating_report') {
      const analysisAgents = ['pathology', 'bioinformatics', 'biochemistry', 'literature']
      if (analysisAgents.includes(agentId)) {
        return 'done'
      }
    }

    // These message types indicate the agent has completed its work
    const doneTypes = ['result', 'completed', 'integrated_report', 'detailed_report',
                       'graphic_abstract', 'round_summary', 'findings']
    if (doneTypes.includes(lastMessage.type)) {
      return 'done'
    }

    if (agentErrors.length > 0) {
      if (lastMessage.type === 'status') {
        return 'working'
      }
      return 'error'
    }

    if (lastMessage.type === 'status') return 'working'
    if (lastMessage.type === 'error') return 'error'
    return 'working'
  }

  // Extract round summaries from messages (memoized to avoid recalculation on every render)
  const roundSummaries = useMemo((): RoundSummary[] => {
    const summaries = new Map<string, RoundSummary>()

    // Look for round_summary messages from PI
    const piMessages = messages.filter(m => m.agent === 'pi')

    for (const msg of piMessages) {
      if (msg.type === 'round_summary' && typeof msg.content === 'object') {
        const content = msg.content as RoundSummaryContent
        const round = content.round || summaries.size + 1
        const phase = (content.phase as RoundSummary['phase']) || ''
        const lineageId = content.lineage_id || 'main'
        const directionKey = content.direction_key || content.direction || ''
        const identity = resultIdentityKey({
          result_identity_key: content.result_identity_key,
          lineageId,
          phase,
          round,
          directionKey,
        })
        summaries.set(identity, {
          round,
          maxRounds: content.max_rounds,
          summary: content.summary || '',
          findings: content.findings || [],
          dataRequests: {
            pathology: content.data_requests?.pathology || [],
            biochemistry: content.data_requests?.biochemistry || [],
            other: content.data_requests?.other || [],
          },
          status: (content.status as RoundSummary['status']) || 'completed',
          pathologyFindings: content.pathology_findings || [],
          bioinformaticsFindings: content.bioinformatics_findings || [],
          biochemistryFindings: content.biochemistry_findings || [],
          literatureFindings: content.literature_findings || [],
          phase,
          integrationCandidates: content.integration_candidates || [],
          integrationReportSummary: content.integration_report_summary || '',
          lineageId,
          directionKey,
          resultIdentityKey: identity,
        })
      }
    }

    // Also check roundResults from session
    if (currentSession?.roundResults) {
      for (const round of currentSession.roundResults) {
        const identity = resultIdentityKey(round)
        if (!summaries.has(identity)) {
          summaries.set(identity, {
            round: round.round,
            summary: round.summary || '',
            findings: [
              ...(round.pathologyFindings || []),
              ...(round.bioinformaticsFindings || []),
              ...(round.biochemistryFindings || [])
            ],
            dataRequests: { pathology: [], biochemistry: [], other: [] },
            status: 'completed',
            pathologyFindings: round.pathologyFindings || [],
            bioinformaticsFindings: round.bioinformaticsFindings || [],
            biochemistryFindings: round.biochemistryFindings || [],
            literatureFindings: round.literatureFindings || [],
            phase: round.phase || '',
            integrationCandidates: round.integrationCandidates || [],
            integrationReportSummary: round.integrationReportSummary || '',
            lineageId: round.lineageId,
            directionKey: round.directionKey,
            resultIdentityKey: identity,
          })
        }
      }
    }

    return Array.from(summaries.values()).sort(
      (a, b) => a.round - b.round
        || resultIdentityKey(a).localeCompare(resultIdentityKey(b)),
    )
  }, [messages, currentSession?.roundResults])

  const notifySyncFailures = (failures: string[]) => {
    if (failures.length === 0) return
    const label = language === 'zh' ? '以下设置同步失败，将使用默认值' : 'Settings sync failed, using defaults for'
    addMessage({
      id: generateId(),
      type: 'error',
      content: `${label}: ${failures.join(', ')}`,
      timestamp: new Date().toISOString(),
    })
  }

  // Annotation confirmation handler
  const handleConfirmAnnotations = useCallback(async (
    annotations: Array<{ cluster_id: string; user_override?: string; user_confirmed: boolean }>
  ) => {
    if (!currentSession) return
    try {
      await apiFetch(`/session/${currentSession.id}/confirm-annotations`, {
        method: 'POST',
        body: JSON.stringify({ annotations }),
      })
      setShowAnnotationDialog(false)
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Failed to confirm annotations:', e)
    }
  }, [currentSession, wsConnected, wsConnect])

  // Plan review handlers
  const handlePlanApprove = useCallback(async () => {
    if (!currentSession) return
    try {
      await apiFetch(`/session/${currentSession.id}/plan-review`, {
        method: 'POST',
        body: JSON.stringify({ action: 'approve' }),
      })
      setShowPlanReviewDialog(false)
      if (!wsConnected) { setTimeout(() => wsConnect?.(), 100) }
    } catch (e) {
      console.error('Failed to approve plan:', e)
    }
  }, [currentSession, wsConnected, wsConnect])

  const handlePlanSkip = useCallback(async () => {
    if (!currentSession) return
    try {
      await apiFetch(`/session/${currentSession.id}/plan-review`, {
        method: 'POST',
        body: JSON.stringify({ action: 'skip' }),
      })
      setShowPlanReviewDialog(false)
      if (!wsConnected) { setTimeout(() => wsConnect?.(), 100) }
    } catch (e) {
      console.error('Failed to skip plan review:', e)
    }
  }, [currentSession, wsConnected, wsConnect])

  const handlePlanModify = useCallback(async (userInput: string) => {
    if (!currentSession) return
    try {
      await apiFetch(`/session/${currentSession.id}/plan-review`, {
        method: 'POST',
        body: JSON.stringify({ action: 'modify', user_input: userInput }),
      })
      // Don't close dialog — it will re-show if PI sends plan back for modification
      if (!wsConnected) { setTimeout(() => wsConnect?.(), 100) }
    } catch (e) {
      console.error('Failed to submit plan modification:', e)
    }
  }, [currentSession, wsConnected, wsConnect])

  // Direction choice handler (multi-direction)
  const handleChooseDirections = useCallback(async (directions: Array<{ direction: string; notes: string; isCustom: boolean }>) => {
    if (!currentSession) return
    try {
      await apiFetch(`/session/${currentSession.id}/choose-direction`, {
        method: 'POST',
        body: JSON.stringify({
          directions: buildDirectionsPayload(directions)
        }),
      })
      setShowExplorationDialog(false)
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Failed to choose directions:', e)
    }
  }, [currentSession, wsConnected, wsConnect])

  // Let PI decide handler
  const handleLetPiDecide = useCallback(async () => {
    if (!currentSession) return
    try {
      await apiFetch(`/session/${currentSession.id}/choose-direction`, {
        method: 'POST',
        body: JSON.stringify({ direction: '', let_pi_decide: true }),
      })
      setShowExplorationDialog(false)
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Failed to let PI decide:', e)
    }
  }, [currentSession, wsConnected, wsConnect])

  const submitIntegrationDirections = async (selected: any[]) => {
    if (!currentSession) return
    await apiFetch(`/session/${currentSession.id}/choose-direction`, {
      method: 'POST',
      body: JSON.stringify({ directions: buildDirectionsPayload(
        selected.map((s: any) => ({ direction: s.direction, notes: s.notes ?? '', isCustom: !!s.isCustom }))
      ) }),
    })
    setShowIntegrationGateDialog(false)
    setIntegrationGateData(null)
  }
  const finalizeFromIntegration = async () => {
    if (!currentSession) return
    await apiFetch(`/session/${currentSession.id}/finalize-from-integration`, { method: 'POST' })
    setShowIntegrationGateDialog(false)
    setIntegrationGateData(null)
  }
  const handleLetPiDecideIntegration = async () => {
    if (!currentSession) return
    await apiFetch(`/session/${currentSession.id}/choose-direction`, {
      method: 'POST',
      body: JSON.stringify({ let_pi_decide: true }),
    })
    setShowIntegrationGateDialog(false)
    setIntegrationGateData(null)
  }

  const handleSupplementaryAnalysis = useCallback(async (requestText: string, target: 'annotation' | 'direction' | 'round') => {
    if (!currentSession) return
    setIsSupplementaryRunning(true)
    if (target === 'round') roundSupplementaryPendingRef.current = true
    try {
      await apiFetch(`/session/${currentSession.id}/supplementary-analysis`, {
        method: 'POST',
        body: JSON.stringify({ request_text: requestText, target }),
      })
      // isSupplementaryRunning will be reset when a new annotation_confirmation or
      // exploration_report message arrives, or — for the round gate — when the
      // workflow returns to waiting_input
    } catch (error) {
      console.error('Supplementary analysis error:', error)
      setIsSupplementaryRunning(false)
      roundSupplementaryPendingRef.current = false
    }
  }, [currentSession])

  const handleStart = async () => {
    // Block in read mode
    if (readMode) {
      alert(t('readModeRestriction'))
      return
    }

    try {
      const isAdmin = authUser?.role === 'admin'

      // Step 1: Validate API credentials (admin uses in-memory config, normal user uses saved config)
      setValidating(true)
      let validation: { valid: boolean; message?: string; uncertain?: boolean }

      if (isAdmin) {
        // Admin: ensure frontend config is synced to backend first
        const { apiConfig, apiConfigured } = useAppStore.getState()
        if (!apiConfig.apiKey && !apiConfigured) {
          setValidating(false)
          if (showConfigPanel) showConfigPanel()
          return
        }
        if (apiConfig.apiKey) {
          // Local key present: sync to backend, then validate it.
          const status = await checkStatus()
          if (!status.configured) {
            await saveConfig()
          }
          validation = await validateApiKey()
        } else {
          // No local key but server is configured: validate the saved global
          // key instead of sending an empty key.
          validation = await validateSavedConfig()
        }
      } else {
        // Non-admin: validate credentials saved on the server
        validation = await userConfig.validateSavedConfig()
      }
      setValidating(false)

      if (validation.valid) {
        const failures = await startAnalysis()
        if (failures) notifySyncFailures(failures)
        setTimeout(() => wsConnect?.(), 100)
      } else {
        setValidationMessage(validation.message || t('apiKeyInvalid'))
        setValidationUncertain(validation.uncertain || false)
        setShowValidationDialog(true)
      }
    } catch (e: any) {
      setValidating(false)
      console.error('Failed to start analysis:', e)
      addMessage({
        id: generateId(),
        type: 'error',
        content: `Failed to start analysis: ${e?.message || String(e)}`,
        timestamp: new Date().toISOString(),
      })
    }
  }

  const handleIgnoreValidationError = async () => {
    setShowValidationDialog(false)
    try {
      const failures = await startAnalysis()
      if (failures) notifySyncFailures(failures)
      setTimeout(() => wsConnect?.(), 100)
    } catch (e: any) {
      console.error('Failed to start analysis:', e)
      addMessage({
        id: generateId(),
        type: 'error',
        content: `Failed to start analysis: ${e?.message || String(e)}`,
        timestamp: new Date().toISOString(),
      })
    }
  }

  const handleReconfigureFromValidation = () => {
    setShowValidationDialog(false)
    if (showConfigPanel) {
      showConfigPanel()
    }
  }

  // Every path that advances the analysis goes through this: Resume, the
  // round-end dialog's Continue, and sending input from that dialog. Only
  // Resume used to check, so the other two sent a request the backend refused
  // and then marked the session running anyway — which hides the re-run button
  // behind `!isRunning` and leaves a page reload as the only way back.
  const blockForwardResume = useCallback((): boolean => {
    if (!isForwardResumeBlocked()) return false
    addMessage({
      id: generateId(),
      type: 'error',
      content:
        language === 'zh'
          ? '当前会话禁止继续分析（运行时不可用或快照失败）。请使用重新运行或快照失败处置操作。'
          : 'Forward resume is blocked (runtime unavailable or snapshot failure). Use re-run or snapshot failure actions.',
      timestamp: new Date().toISOString(),
    })
    return true
  }, [isForwardResumeBlocked, addMessage, language])

  // Dialog handler: Send user input with files (uploaded to agent-specific endpoints)
  const handleDialogSend = useCallback(async (message: string, files: UploadedFileWithAgent[]) => {
    if (!currentSession) return
    // Leave the dialog mounted: its message and file selection are local state
    // that is destroyed on unmount, and this path bails out before anything is
    // uploaded. Closing here would discard what the user typed with nothing
    // sent and nothing saved.
    if (blockForwardResume()) return

    // Upload files to their respective agent endpoints
    const uploadedData: Array<{ type: string; filename: string; id?: string }> = []

    for (const { file, targetAgent, groupId, sampleNumber, dataType } of files) {
      try {
        const formData = new FormData()
        formData.append('file', file)

        let endpoint: string
        const section = targetAgent === 'pathology' ? 'pathology'
                      : targetAgent === 'bioinformatics' ? 'sequencing'
                      : targetAgent === 'biochemistry' ? 'biochemistry'
                      : null

        if (section) {
          formData.append('section', section)
          formData.append('data_type', dataType || (
            section === 'pathology' ? 'user_upload'
            : section === 'sequencing' ? 'user_upload'
            : (file.type.startsWith('image/') ? 'image' : 'file')))
          if (groupId) formData.append('group_id', groupId)
          if (sampleNumber != null) formData.append('sample_number', String(sampleNumber))
          endpoint = `${API_BASE}/session/${currentSession.id}/upload/data`
        } else {
          endpoint = `${API_BASE}/session/${currentSession.id}/upload/file`
        }

        const response = await apiFetch<any>(endpoint.replace(API_BASE, ''), { method: 'POST', body: formData })
        if (response) {
          uploadedData.push({
            type: file.type.startsWith('image/') ? 'image' : 'file',
            filename: file.name,
            id: response.data?.file_id || response.data?.image_id,
          })
        }
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err)
      }
    }

    // Send human-input with message + file references
    try {
      await apiFetch(`/session/${currentSession.id}/human-input`, {
        method: 'POST',
        body: JSON.stringify({
          message,
          uploaded_files: uploadedData,
          target_agent: 'pi',
        }),
      })

      // Resume (signals backend workflow to continue)
      await resumeAnalysis()
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Failed to send input:', e)
    }

    setShowFeedbackDialog(false)
  }, [currentSession, resumeAnalysis, wsConnected, wsConnect, blockForwardResume])

  // Dialog handler: Skip/Continue (auto-proceed without input)
  const handleDialogSkip = useCallback(async () => {
    if (blockForwardResume()) {
      setShowFeedbackDialog(false)
      return
    }
    try {
      await continueRound()
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Failed to continue round:', e)
    }
    setShowFeedbackDialog(false)
  }, [continueRound, wsConnected, wsConnect, blockForwardResume])

  // Pause handler
  const handlePause = useCallback(async () => {
    try {
      await pauseAnalysis()
    } catch (e) {
      console.error('Failed to pause:', e)
    }
  }, [pauseAnalysis])

  // Resume handler — gated by backend runtime_restore_status / snapshot_failure
  const handleResume = useCallback(async () => {
    if (blockForwardResume()) return
    try {
      await resumeAnalysis()
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Failed to resume:', e)
    }
  }, [resumeAnalysis, wsConnected, wsConnect, blockForwardResume])

  const handleRerun = useCallback(async () => {
    if (!currentSession) return
    try {
      await apiFetch(`/session/${currentSession.id}/run`, { method: 'POST' })
      useAppStore.getState().setRuntimeRestoreStatus('complete')
      useAppStore.getState().setRuntimeRestoreWarning(null)
      useAppStore.getState().updateSession({ status: 'running' })
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Failed to re-run analysis:', e)
      addMessage({
        id: generateId(),
        type: 'error',
        content: `Failed to re-run analysis: ${e instanceof Error ? e.message : String(e)}`,
        timestamp: new Date().toISOString(),
      })
    }
  }, [currentSession, wsConnected, wsConnect, addMessage])

  // Retry and Proceed are the same capture dispatch. Proceed differs only in
  // that the backend arms a one-shot memory-preflight override before running
  // it, so both share this handler rather than keeping two copies in sync.
  const runSnapshotFailureCapture = useCallback(async (
    action: 'retry' | 'proceed',
  ) => {
    if (!currentSession) return
    try {
      // Capture-only retry — must not regenerate completed reports
      const resp = await apiFetch<{
        success?: boolean
        data?: {
          status?: TaskStatus
          still_blocked?: boolean
          snapshot_failure?: Record<string, unknown> | null
          snapshot_failure_allowed_actions?: SnapshotFailureAction[]
          transaction_id?: string
          snapshot_transaction?: SnapshotTransactionRecord
        }
      }>(`/session/${currentSession.id}/snapshot-failure/${action}`, {
        method: 'POST',
        body: JSON.stringify({ regenerate_reports: false }),
      })
      const ok = resp?.success === true
      if (!ok) {
        // HTTP 200 with success:false / still_blocked must keep the HITL gate
        const data = resp?.data
        const rawFailure = data?.snapshot_failure
        if (rawFailure && typeof rawFailure === 'object') {
          useAppStore.getState().hydrateRuntimeRestoreFromSession({
            snapshot_failure: rawFailure,
            snapshot_failure_allowed_actions: data?.snapshot_failure_allowed_actions,
          })
        }
        // else leave existing snapshotFailure intact
        return
      }
      const transaction = resp?.data?.snapshot_transaction
      if (transaction) {
        // Claim the auto-open edge here too: this explicit open and the effect
        // above are the same one-shot, so without it a user who closes the panel
        // gets it back the first time the effect observes the new transaction.
        snapshotAutoOpenedForRef.current = transaction.transaction_id
        setShowSnapshotPanel(true)
        monitorSnapshotRetry(currentSession.id, transaction)
      }
      if (resp?.data?.status) {
        useAppStore.getState().updateSession({ status: resp.data.status })
      }
      clearSnapshotFailure()
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error(`Snapshot failure ${action} failed:`, e)
    }
  }, [currentSession, clearSnapshotFailure, monitorSnapshotRetry, wsConnected, wsConnect])

  const handleSnapshotFailureRetry = useCallback(
    async () => {
      if (!currentSession) return
      if ((snapshotFailure?.failure.code || '').startsWith('runtime_persistence_')) {
        try {
          const details = snapshotFailure?.failure.details || {}
          await apiFetch(`/session/${currentSession.id}/runtime-persistence/retry`, {
            method: 'POST',
            body: JSON.stringify({
              gate_token: details.gate_token,
              gate_generation: details.gate_generation,
            }),
          })
        } catch (e) {
          console.error('Runtime persistence retry failed:', e)
        }
        return
      }
      await runSnapshotFailureCapture('retry')
    },
    [currentSession, snapshotFailure, runSnapshotFailureCapture],
  )

  const handleSnapshotFailureProceed = useCallback(
    () => runSnapshotFailureCapture('proceed'),
    [runSnapshotFailureCapture],
  )

  const handleSnapshotFailureSkip = useCallback(async () => {
    if (!currentSession) return
    try {
      const resp = await apiFetch<{
        success?: boolean
        data?: { status?: TaskStatus }
      }>(`/session/${currentSession.id}/snapshot-failure/skip`, {
        method: 'POST',
      })
      if (resp?.success !== true) return
      if (resp?.data?.status) {
        useAppStore.getState().updateSession({ status: resp.data.status })
      }
      clearSnapshotFailure()
      // Reconnect the WebSocket so the workflow resumes past the skipped boundary.
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Snapshot failure skip failed:', e)
    }
  }, [currentSession, clearSnapshotFailure, wsConnected, wsConnect])

  const handleSnapshotFailureManage = useCallback(async () => {
    if (!currentSession) return
    if ((snapshotFailure?.failure.code || '').startsWith('runtime_persistence_')) {
      try {
        const details = snapshotFailure?.failure.details || {}
        const resp = await apiFetch<{
          success?: boolean
          data?: { status?: TaskStatus; still_blocked?: boolean }
        }>(`/session/${currentSession.id}/runtime-persistence/recompute`, {
          method: 'POST',
          body: JSON.stringify({
            gate_token: details.gate_token,
            gate_generation: details.gate_generation,
          }),
        })
        if (resp?.data?.status) {
          useAppStore.getState().updateSession({ status: resp.data.status })
        }
        if (resp?.success && resp?.data?.still_blocked === false) {
          clearSnapshotFailure()
        }
      } catch (e) {
        console.error('Runtime persistence recompute failed:', e)
      }
      return
    }
    setShowSnapshotPanel(true)
  }, [currentSession, snapshotFailure, clearSnapshotFailure])

  const handleSnapshotFailureTerminate = useCallback(async () => {
    if (!currentSession) return
    try {
      const runtimeFailure = (snapshotFailure?.failure.code || '').startsWith('runtime_persistence_')
      const endpoint = runtimeFailure
        ? `/session/${currentSession.id}/runtime-persistence/terminate`
        : `/session/${currentSession.id}/snapshot-failure/terminate`
      const resp = await apiFetch<{
        success?: boolean
        message?: string
        data?: {
          status?: TaskStatus
          still_blocked?: boolean
          action_pending?: boolean
          cleanup_incomplete?: boolean
          termination_committed?: boolean
        }
      }>(endpoint, {
        method: 'POST',
        ...(runtimeFailure
          ? {
              body: JSON.stringify({
                gate_token: snapshotFailure?.failure.details?.gate_token,
                gate_generation: snapshotFailure?.failure.details?.gate_generation,
              }),
            }
          : {}),
      })
      if (resp?.data?.status) {
        useAppStore.getState().updateSession({ status: resp.data.status })
      }
      if (
        resp?.data?.termination_committed
        || (resp?.success && resp?.data?.still_blocked === false)
      ) {
        // A committed termination clears the recovery gate even when resource
        // cleanup needs a background retry. A live workflow's accepted action
        // remains pending and must keep the dialog reachable until its close
        // consumer publishes the committed STOPPED transition.
        clearSnapshotFailure()
      }
      if (!resp?.success) {
        addMessage({
          id: generateId(),
          type: 'error',
          content: resp?.message || (
            language === 'zh'
              ? '终止已提交，但后台资源清理尚未完成；系统将继续重试。'
              : 'Termination was committed, but background resource cleanup is incomplete and will be retried.'
          ),
          timestamp: new Date().toISOString(),
        })
      }
    } catch (e) {
      console.error('Snapshot failure terminate failed:', e)
    }
  }, [currentSession, snapshotFailure, clearSnapshotFailure, addMessage, language])

  // Stop handler (immediate or summarize)
  const handleStop = useCallback(async (action: 'immediate' | 'summarize') => {
    setStopping(true)
    try {
      await stopAnalysis(action)
      setShowStopDialog(false)
      setShowFeedbackDialog(false)
    } catch (e) {
      console.error('Failed to stop:', e)
    } finally {
      setStopping(false)
    }
  }, [stopAnalysis])

  // ── Exploration failed handlers ──────────────────────────────────
  // Exploration (Stage 1) did not complete — most often a data problem.
  // The workflow is paused at the round gate; these actions hand control
  // back to the user without auto-advancing rounds.
  const handleExplorationFailedGoToData = useCallback(() => {
    clearExplorationFailedState()
    useAppStore.getState().setActiveTab('data')
  }, [clearExplorationFailedState])

  const handleExplorationFailedContinue = useCallback(() => {
    // Close the dialog; the user will retry Stage 1 via the normal continue flow.
    clearExplorationFailedState()
  }, [clearExplorationFailedState])

  const handleExplorationFailedStop = useCallback(() => {
    clearExplorationFailedState()
    handleStop('immediate')
  }, [clearExplorationFailedState, handleStop])

  // ── LLM failure handlers ──────────────────────────────────────────

  const handleLlmSkip = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    if (!currentSession || !info) return
    try {
      await apiFetch(`/session/${currentSession.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({
          action: 'skip',
          node_name: info.nodeName,
          generation: info.generation,
        }),
      })
      setShowLlmTimeoutDialog(false)
    } catch (e) {
      console.error('Failed to signal LLM skip:', e)
    }
  }, [currentSession])

  const handleLlmRetry = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    if (!currentSession || !info) return
    try {
      await apiFetch(`/session/${currentSession.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({
          action: 'retry',
          node_name: info.nodeName,
          generation: info.generation,
        }),
      })
      setShowLlmTimeoutDialog(false)
    } catch (e) {
      console.error('Failed to signal LLM retry:', e)
    }
  }, [currentSession])

  const handleSpeciesInput = useCallback(async (input: string) => {
    if (!currentSession) return
    try {
      await apiFetch(`/session/${currentSession.id}/species-input`, {
        method: 'POST',
        body: JSON.stringify({ input }),
      })
      clearSpeciesInputState()
    } catch (error) {
      console.error('Failed to submit species input:', error)
    }
  }, [currentSession, clearSpeciesInputState])

  const handleLlmModifyAgent = useCallback(() => {
    setShowLlmTimeoutDialog(false)
    useAppStore.getState().setLlmModifyPending(true)
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setActiveTab('prompts')
  }, [])

  const handleLlmTerminate = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    if (!currentSession || !info) return
    try {
      await apiFetch(`/session/${currentSession.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({
          action: 'terminate',
          node_name: info.nodeName,
          generation: info.generation,
        }),
      })
      setShowLlmTimeoutDialog(false)
    } catch (e) {
      console.error('Failed to signal LLM terminate:', e)
    }
  }, [currentSession])

  // StopDialog cancel callback: close StopDialog, reopen LlmTimeoutDialog if LLM failure in progress
  const handleStopDialogCancel = useCallback(() => {
    setShowStopDialog(false)
    if (useAppStore.getState().llmFailureInfo) {
      setShowLlmTimeoutDialog(true)
    }
  }, [])

  const handleLlmResumeAfterModify = useCallback(() => {
    useAppStore.getState().setLlmModifyPending(false)
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setActiveTab('analysis')
    handleLlmRetry()
  }, [handleLlmRetry])

  const handleLlmReturnToSettings = useCallback(() => {
    useAppStore.getState().setLlmModifySaved(false)
  }, [])

  const handleLlmReturnToTimeout = useCallback(() => {
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setShowLlmTimeoutDialog(true)
  }, [])

  const handleReopenDialog = useCallback(() => {
    if (currentSession?.status === 'waiting_annotation') {
      // The dialog renders only with a payload, so restore it from the server's
      // annotation state when the gate message is not in the stream — otherwise
      // this button is a no-op after a browser restart.
      if (!annotationData && annotationGateData) setAnnotationData(annotationGateData)
      setShowAnnotationDialog(true)
    } else if (currentSession?.status === 'waiting_input') {
      if (integrationGateData) {
        // Integration gate has its own panel; don't reopen the generic feedback dialog.
        setShowIntegrationGateDialog(true)
      } else if (explorationReportData && (currentSession.currentRound ?? 0) <= MAX_EXPLORATION_ROUND) {
        setShowExplorationDialog(true)
      } else if (feedbackRoundData && feedbackRoundData.phase !== 'integration') {
        setShowFeedbackDialog(true)
      } else {
        // The initial /load request can race with the reconnect state_sync:
        // setCurrentSession clears transient dialog state after the socket has
        // restored it. Reconnect on demand so the backend re-emits the pending
        // HITL payload instead of leaving this badge as a no-op.
        wsConnect?.()
      }
    } else if (currentSession?.status === 'waiting_plan_review') {
      setShowPlanReviewDialog(true)
    } else if (currentSession?.status === 'waiting_species_input') {
      useAppStore.getState().setShowSpeciesInputDialog(true)
    }
  }, [currentSession?.status, currentSession?.currentRound, explorationReportData, feedbackRoundData, integrationGateData, annotationData, annotationGateData, wsConnect])

  // Countdown timer for LLM failure auto-skip
  const { timeRemaining } = useCountdownTimer({
    countdownSeconds: roundWaitSeconds > 0 ? roundWaitSeconds : 0,
    onExpire: handleLlmSkip,
    resetKey: llmFailureInfo?.generation != null
      ? String(llmFailureInfo.generation)
      : undefined,
  })

  // Auto-connect WebSocket when session is running and not connected
  // Uses wsConnect from the store (managed by App.tsx) to avoid duplicate WebSocket instances
  useEffect(() => {
    if (currentSession &&
        (currentSession.status === 'running' ||
         currentSession.status === 'waiting_input' ||
         currentSession.status === 'waiting_annotation' ||
         currentSession.status === 'waiting_plan_review' ||
         currentSession.status === 'waiting_species_input') &&
        !wsConnected && wsConnect) {
      wsConnect()
    }
  }, [currentSession?.id, currentSession?.status, wsConnected, wsConnect])

  // Auto-expand latest round + show feedback dialog when round completes with waiting_input
  useEffect(() => {
    if (roundSummaries.length > 0) {
      const latest = roundSummaries[roundSummaries.length - 1]
      setExpandedRound(resultIdentityKey(latest))

      // Show feedback dialog when a round completes and is waiting for input.
      // Use ref guard to prevent duplicate triggers for the same round.
      // Integration-phase rounds are handled by the dedicated IntegrationGatePanel
      // (direction selection / let-PI-decide / finalize); the generic feedback
      // dialog's continue would send /human-input and leave the integration gate
      // unresolved, so never open it for an integration round.
      if (latest.status === 'waiting_input' && currentSession?.status === 'waiting_input'
          && latest.phase !== 'integration'
          && latest.round !== lastDialogRoundRef.current) {
        lastDialogRoundRef.current = latest.round
        setFeedbackRoundData(latest)
        setShowFeedbackDialog(true)
      }
    }
  }, [roundSummaries, currentSession?.status])

  // Auto-dismiss dialog when analysis resumes
  useEffect(() => {
    if (currentSession?.status === 'running' && showFeedbackDialog) {
      setShowFeedbackDialog(false)
    }
  }, [currentSession?.status, showFeedbackDialog])

  // Re-arm the round dialog after a round-target supplementary pass: it returns to
  // the same gate without emitting a new round summary, so the per-round guard above
  // would never re-open it. Depending on status alone is what makes this safe — the
  // request is issued while already waiting_input, so it cannot consume its own latch.
  useEffect(() => {
    if (currentSession?.status !== 'waiting_input') return
    if (!roundSupplementaryPendingRef.current) return
    roundSupplementaryPendingRef.current = false
    setIsSupplementaryRunning(false)
    setShowFeedbackDialog(true)
  }, [currentSession?.status])

  // Detect annotation_confirmation messages and show dialog (supports data refresh for supplementary analysis)
  useEffect(() => {
    if (currentSession?.status !== 'waiting_annotation') {
      if (isSupplementaryRunning) setIsSupplementaryRunning(false)
      return
    }
    let annMsg: typeof messages[number] | undefined
    for (let i = messages.length - 1; i >= 0; i--) { if (messages[i].type === 'annotation_confirmation') { annMsg = messages[i]; break } }
    if (annMsg && annMsg.id !== lastAnnotationMsgRef.current) {
      lastAnnotationMsgRef.current = annMsg.id
      setAnnotationData(annMsg.content as AnnotationData)
      setIsSupplementaryRunning(false)
      if (!showAnnotationDialog) {
        setShowAnnotationDialog(true)
      }
    } else if (!annMsg && !annotationData && annotationGateData) {
      // No gate message in the stream — a restarted browser, a replay window
      // that dropped it, or a snapshot restore. Fall back to the server's
      // authoritative annotation state so the gate stays reachable. A live
      // message always wins above: it is fresher after supplementary analysis.
      setAnnotationData(annotationGateData)
      setShowAnnotationDialog(true)
    }
  }, [messages, currentSession?.status, annotationGateData, annotationData])

  // Detect plan_review messages and show dialog
  useEffect(() => {
    if (currentSession?.status !== 'waiting_plan_review') return
    let planMsg: typeof messages[number] | undefined
    for (let i = messages.length - 1; i >= 0; i--) { if (messages[i].type === 'plan_review') { planMsg = messages[i]; break } }
    if (planMsg && planMsg.id !== lastPlanReviewMsgRef.current) {
      lastPlanReviewMsgRef.current = planMsg.id
      setPlanReviewData(planMsg.content as PlanReviewData)
      if (!showPlanReviewDialog) {
        setShowPlanReviewDialog(true)
      }
    }
  }, [messages, currentSession?.status, showPlanReviewDialog])

  // Detect plan_approved messages and close dialog
  useEffect(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].type === 'plan_approved') {
        if (showPlanReviewDialog) {
          setShowPlanReviewDialog(false)
        }
        return
      }
    }
  }, [messages])

  // Auto-dismiss plan review dialog when status changes away from waiting_plan_review
  useEffect(() => {
    if (currentSession?.status !== 'waiting_plan_review' && currentSession?.status !== 'running') {
      // Don't dismiss if we're in running state (plan was just approved)
    } else if (currentSession?.status === 'running') {
      if (showPlanReviewDialog) {
        setShowPlanReviewDialog(false)
      }
    }
  }, [currentSession?.status, showPlanReviewDialog])

  // Detect exploration_report messages and show dialog (supports data refresh for supplementary analysis)
  useEffect(() => {
    let expMsg: typeof messages[number] | undefined
    for (let i = messages.length - 1; i >= 0; i--) { if (messages[i].type === 'exploration_report') { expMsg = messages[i]; break } }
    if (expMsg && expMsg.id !== lastExplorationMsgRef.current && currentSession?.status === 'waiting_input') {
      if ((currentSession.currentRound ?? 0) > 1) {
        lastExplorationMsgRef.current = expMsg.id
        return
      }
      lastExplorationMsgRef.current = expMsg.id
      setExplorationReportData(expMsg.content as ExplorationReportData)
      setIsSupplementaryRunning(false)
      if (!showExplorationDialog) {
        setShowExplorationDialog(true)
        setShowFeedbackDialog(false)
      }
    }
  }, [messages, currentSession?.status, currentSession?.currentRound])

  const isCurrentIntegrationGate = currentSession?.status === 'waiting_input' && currentSession?.analysisPhase === 'integration'

  useEffect(() => {
    if (!isCurrentIntegrationGate) {
      setShowIntegrationGateDialog(false)
      // Functional update returns the same ref when already null, so React skips
      // the re-render and the effect doesn't need integrationGateData as a dep.
      setIntegrationGateData((prev: any) => (prev ? null : prev))
      return
    }
    let msg: any = null
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].type === 'integration_gate') { msg = messages[i]; break }
    }
    // Dedup: only (re)open the panel for a message we haven't shown yet, so a
    // submit that clears integrationGateData isn't immediately re-populated from
    // the same stale message before the backend flips status away from waiting.
    if (msg && lastIntegrationMsgRef.current !== msg.id) {
      lastIntegrationMsgRef.current = msg.id
      setIntegrationGateData(msg.content)
      setShowIntegrationGateDialog(true)
    }
  }, [messages, isCurrentIntegrationGate])

  // Auto-dismiss annotation/exploration dialogs when analysis resumes
  useEffect(() => {
    if (currentSession?.status === 'running') {
      if (showAnnotationDialog) setShowAnnotationDialog(false)
      if (showExplorationDialog) setShowExplorationDialog(false)
    }
  }, [currentSession?.status, showAnnotationDialog, showExplorationDialog])

  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 rounded-full mb-6">
          <Users className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="font-display text-xl font-semibold text-slate-900 mb-2">
          {t('startNewAnalysis')}
        </h2>
        <p className="text-slate-500 max-w-md">
          {t('welcomeDesc')}
        </p>
      </div>
    )
  }

  const isPaused = currentSession.status === 'paused'
  const isWaitingInput = currentSession.status === 'waiting_input' || currentSession.status === 'waiting_annotation' || currentSession.status === 'waiting_plan_review' || currentSession.status === 'waiting_species_input'
  const isRunning = currentSession.status === 'running'
  const isCompleted = currentSession.status === 'completed'
  const isFailed = currentSession.status === 'failed'

  return (
    <div className="flex flex-col h-full">
      {/* Validating spinner overlay */}
      {validating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-lg p-6 flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-accent-600" />
            <span className="text-slate-700 font-medium">
              {t('validatingApiKey')}
            </span>
          </div>
        </div>
      )}

      {/* Validation failure dialog */}
      {showValidationDialog && (
        <ValidationDialog
          message={validationMessage}
          uncertain={validationUncertain}
          onIgnore={handleIgnoreValidationError}
          onReconfigure={handleReconfigureFromValidation}
          language={language}
        />
      )}

      {/* Runtime restore safety banner (persists across reload).
          Not gated on snapshotV2Enabled: an ordinary session *import* sets
          runtime_restore_status='unavailable' with Snapshot V2 switched off, and
          hiding the banner there left the Resume button disabled with the
          explanation suppressed. */}
      {(runtimeRestoreStatus === 'unavailable' || runtimeRestoreStatus === 'intentionally_kept') && (
        <div className="mb-2">
          <RuntimeWarningBanner
            runtimeRestoreStatus={runtimeRestoreStatus}
            warning={runtimeRestoreWarning}
            isZh={language === 'zh'}
          />
        </div>
      )}

      {/* Header - Compact */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-200">
        <div>
          <h2 className="font-display text-lg font-semibold text-slate-900">{t('analysisControl')}</h2>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {currentSession.status === 'pending' && (
            <button
              onClick={handleStart}
              disabled={readMode || isForwardResumeBlocked()}
              className={`flex items-center gap-2 px-4 py-1.5 text-white rounded-lg font-medium transition-colors ${
                readMode || isForwardResumeBlocked()
                  ? 'bg-slate-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
              title={
                isForwardResumeBlocked()
                  ? (language === 'zh' ? '运行时不可用或快照失败，无法启动' : 'Blocked by runtime/snapshot failure')
                  : readMode
                    ? t('readModeRestriction')
                    : undefined
              }
            >
              <Play className="w-4 h-4" />
              {t('startAnalysis')}
            </button>
          )}

          {isRerunEnabled() && !isRunning && (
            <button
              onClick={handleRerun}
              disabled={readMode}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                readMode
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-indigo-100 hover:bg-indigo-200 text-indigo-800'
              }`}
              title={language === 'zh' ? '从原始数据重新运行' : 'Re-run analysis from raw inputs'}
            >
              <Play className="w-3.5 h-3.5" />
              {language === 'zh' ? '重新运行' : 'Re-run'}
            </button>
          )}
          
          {isRunning && (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2 px-3 py-1 bg-blue-100 text-blue-700 rounded-lg">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="font-medium">Round {currentSession.currentRound}</span>
              </div>
              <button
                onClick={handlePause}
                className="flex items-center gap-1.5 px-3 py-1 bg-amber-100 hover:bg-amber-200 text-amber-700 rounded-lg text-sm font-medium transition-colors"
              >
                <Pause className="w-4 h-4" />
                {language === 'zh' ? '暂停' : 'Pause'}
              </button>
              <button
                onClick={() => setShowStopDialog(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg text-sm font-medium transition-colors"
              >
                <Square className="w-3.5 h-3.5" />
                {language === 'zh' ? '停止' : 'Stop'}
              </button>
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
            </div>
          )}

          {isPaused && (
            <div className="flex items-center gap-2">
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
              <div className="flex items-center gap-2 px-3 py-1 bg-slate-100 text-slate-600 rounded-lg">
                <Pause className="w-4 h-4" />
                <span className="font-medium">
                  {language === 'zh' ? '已暂停' : 'Paused'}
                </span>
              </div>
              <button
                onClick={handleResume}
                disabled={isForwardResumeBlocked()}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                  isForwardResumeBlocked()
                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                    : 'bg-green-100 hover:bg-green-200 text-green-700'
                }`}
                title={
                  isForwardResumeBlocked()
                    ? (language === 'zh' ? '运行时不可用，无法继续' : 'Resume blocked — runtime unavailable')
                    : undefined
                }
              >
                <Play className="w-3.5 h-3.5" />
                {language === 'zh' ? '继续' : 'Resume'}
              </button>
            </div>
          )}

          {isWaitingInput && (
            <div className="flex items-center gap-2">
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
              {llmFailureInfo ? (
                <button
                  onClick={() => setShowLlmTimeoutDialog(true)}
                  className="flex items-center gap-2 px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors"
                  aria-label={t('llmCallFailedBadge')}
                >
                  <AlertCircle className="w-4 h-4" aria-hidden="true" />
                  <span className="font-medium">
                    {t('llmCallFailedBadge')}
                  </span>
                </button>
              ) : (
                <button
                  onClick={handleReopenDialog}
                  className={`flex items-center gap-2 px-3 py-1 rounded-lg transition-colors cursor-pointer ${
                    currentSession.status === 'waiting_annotation'
                      ? 'bg-accent-100 text-accent-700 hover:bg-accent-200'
                      : 'bg-amber-100 text-amber-700 hover:bg-amber-200'
                  }`}
                  title={language === 'zh' ? '点击重新打开交互窗口' : 'Click to reopen interaction dialog'}
                >
                  <AlertCircle className="w-4 h-4" />
                  <span className="font-medium">
                    {currentSession.status === 'waiting_annotation'
                      ? (language === 'zh' ? '等待注释确认' : 'Waiting for Annotation')
                      : currentSession.status === 'waiting_species_input'
                      ? t('waitingSpeciesInput')
                      : (language === 'zh' ? '等待输入' : 'Waiting for Input')}
                  </span>
                </button>
              )}
              <button
                onClick={() => setShowStopDialog(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg text-sm font-medium transition-colors"
              >
                <Square className="w-3.5 h-3.5" />
                {language === 'zh' ? '停止' : 'Stop'}
              </button>
            </div>
          )}

          {isCompleted && (
            <div className="flex items-center gap-2">
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
              <div className="flex items-center gap-2 px-3 py-1 bg-green-100 text-green-700 rounded-lg">
                <CheckCircle2 className="w-4 h-4" />
                <span className="font-medium">{t('analysisComplete')}</span>
              </div>
            </div>
          )}

          {isFailed && (
            <div className="flex items-center gap-2">
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
              <div className="flex items-center gap-2 px-3 py-1 bg-red-100 text-red-700 rounded-lg">
                <AlertCircle className="w-4 h-4" />
                <span className="font-medium">{language === 'zh' ? '分析失败' : 'Analysis Failed'}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Agent Status - Compact Row */}
      <div className="py-3 border-b border-slate-200">
        <div className="flex items-center gap-2 flex-wrap">
          {agents.map((agent) => {
            const status = getAgentStatus(agent.id)
            const Icon = agent.icon

            return (
              <div
                key={agent.id}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm whitespace-nowrap ${
                  status === 'working'
                    ? 'bg-blue-100 text-blue-700'
                    : status === 'done'
                    ? 'bg-green-100 text-green-700'
                    : status === 'error'
                    ? 'bg-red-100 text-red-700'
                    : 'bg-slate-100 text-slate-600'
                }`}
              >
                {status === 'working' ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Icon className="w-3.5 h-3.5" />
                )}
                <span>{agent.name}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Agent Errors Display */}
      {(() => {
        if (unresolvedErrors.length === 0) return null

        return (
          <div className="py-3 border-b border-slate-200">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="w-4 h-4 text-red-500" />
              <span className="text-sm font-medium text-red-700">
                {language === 'zh' ? '分析错误' : 'Analysis Errors'} ({unresolvedErrors.length})
              </span>
            </div>
            <div className="space-y-2">
              {unresolvedErrors.map((error) => {
                const agentInfo = agents.find(a => a.id === error.agent)
                return (
                  <div
                    key={error.id}
                    className="flex items-start gap-2 p-2 bg-red-50 border border-red-200 rounded-lg text-sm"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          error.agent === 'pathology' ? 'bg-rose-100 text-rose-700' :
                          error.agent === 'bioinformatics' ? 'bg-emerald-100 text-emerald-700' :
                          error.agent === 'biochemistry' ? 'bg-amber-100 text-amber-700' :
                          error.agent === 'literature' ? 'bg-blue-100 text-blue-700' :
                          'bg-slate-100 text-slate-700'
                        }`}>
                          {agentInfo?.name || error.agent}
                        </span>
                        <span className="text-xs text-slate-500">Round {error.round}</span>
                        <span className="text-xs text-slate-400">{error.errorType}</span>
                      </div>
                      <p className="text-red-700 mt-1">{error.message}</p>
                      {error.details && (
                        <details className="mt-1">
                          <summary className="text-xs text-red-500 cursor-pointer hover:text-red-700">
                            {language === 'zh' ? '查看详情' : 'View Details'}
                          </summary>
                          <pre className="text-xs text-red-600 mt-1 whitespace-pre-wrap bg-red-100 p-2 rounded max-h-32 overflow-y-auto">
                            {error.details}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )
      })()}

      {/* Main Content Area - Scrollable */}
      <div className="flex-1 overflow-y-auto py-4 space-y-4">
        <RoundSummaries
          roundSummaries={roundSummaries}
          currentSession={currentSession}
          expandedRound={expandedRound}
          setExpandedRound={setExpandedRound}
          isRunning={isRunning}
          language={language}
        />

        {/* Brief Analysis Result */}
        {currentSession.finalSummary && (
          <div className="bg-gradient-to-br from-accent-50 to-accent-100 border border-accent-200 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-accent-100 rounded-lg">
                <CheckCircle2 className="w-5 h-5 text-accent-600" />
              </div>
              <h3 className="font-display font-semibold text-accent-900">{t('finalReport')}</h3>
            </div>
            <div className="prose prose-slate prose-sm max-w-none">
              <ReactMarkdown>{currentSession.finalSummary}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* Annotation Confirmation Dialog — shown during exploration when annotations need review */}
      {showAnnotationDialog && annotationData && currentSession && (
        <AnnotationConfirmationDialog
          annotations={annotationData.annotations || []}
          tissue={annotationData.tissue || ''}
          organism={annotationData.organism || ''}
          tissueSummary={annotationData.tissue_summary || ''}
          reportHtmlUrl={annotationData.report_html_url || undefined}
          detailedReportUrl={annotationData.detailed_report_url || undefined}
          validationStatus={annotationData.validation_status}
          validationSummary={annotationData.validation_summary}
          validationTextSummary={annotationData.validation_text_summary}
          validationResult={annotationData.validation_result}
          reclusteringRecommendations={annotationData.reclustering_recommendations}
          proportionGroupLabels={annotationData.proportion_group_labels || []}
          proportionDimension={annotationData.proportion_dimension || 'group'}
          language={language}
          countdownSeconds={roundWaitSeconds}
          onConfirm={handleConfirmAnnotations}
          onRequestSupplementary={(text) => handleSupplementaryAnalysis(text, 'annotation')}
          isSupplementaryRunning={isSupplementaryRunning}
          onClose={() => setShowAnnotationDialog(false)}
        />
      )}

      {/* Plan Review Dialog — shown when analysis plan needs user approval */}
      {showPlanReviewDialog && planReviewData && currentSession && (
        <PlanReviewDialog
          open={showPlanReviewDialog}
          plan={planReviewData.plan || null}
          dialogueHtml={planReviewData.dialogue_html || ''}
          phase={planReviewData.phase || ''}
          round={planReviewData.round || currentSession.currentRound || 1}
          stage={planReviewData.stage || 'full'}
          countdownSeconds={planReviewData.wait_seconds != null ? planReviewData.wait_seconds : roundWaitSeconds}
          onApprove={handlePlanApprove}
          onSkip={handlePlanSkip}
          onModify={handlePlanModify}
        />
      )}

      {/* Exploration Report Dialog — shown after Round 1 exploration completes */}
      {showExplorationDialog && explorationReportData && currentSession && (
        <ExplorationReportDialog
          report={(explorationReportData.report_a || explorationReportData) as ExplorationReport}
          roundSummary={explorationReportData.round_summary?.summary || ''}
          explorationReportUrl={explorationReportData.html_url || undefined}
          detailedReportUrl={explorationReportData.detailed_report_url || undefined}
          language={language}
          countdownSeconds={roundWaitSeconds}
          onChooseDirections={handleChooseDirections}
          onLetPiDecide={handleLetPiDecide}
          onRequestSupplementary={(text) => handleSupplementaryAnalysis(text, 'direction')}
          isSupplementaryRunning={isSupplementaryRunning}
          onClose={() => setShowExplorationDialog(false)}
        />
      )}

      {isCurrentIntegrationGate && integrationGateData && (
        <IntegrationGatePanel
          cycle={integrationGateData.cycle}
          candidates={integrationGateData.candidates}
          onSubmit={submitIntegrationDirections}
          onFinalize={finalizeFromIntegration}
          onLetPiDecide={handleLetPiDecideIntegration}
          countdownSeconds={roundWaitSeconds}
          language={language}
          reportUrl={integrationGateData.html_url || undefined}
          onClose={() => setShowIntegrationGateDialog(false)}
          open={showIntegrationGateDialog}
        />
      )}

      {/* Round Feedback Dialog — shown after each round completes (not during exploration
          report, and never at the integration gate, which has its own IntegrationGatePanel).
          The integration exclusion is also enforced upstream where showFeedbackDialog is set;
          this render-level guard defends against stale state on reconnect/rehydrate. */}
      {showFeedbackDialog && !showExplorationDialog
        && !(isCurrentIntegrationGate && integrationGateData)
        && feedbackRoundData && feedbackRoundData.phase !== 'integration'
        && currentSession && !llmFailureInfo && (
        <RoundFeedbackDialog
          roundNumber={feedbackRoundData.round}
          maxRounds={maxRounds}
          roundSummary={feedbackRoundData}
          language={language}
          countdownSeconds={roundWaitSeconds}
          treatmentGroups={currentSession.treatmentGroups || []}
          onSend={handleDialogSend}
          onSkip={handleDialogSkip}
          onRequestSupplementary={(text) => handleSupplementaryAnalysis(text, 'round')}
          isSupplementaryRunning={isSupplementaryRunning}
          onStopImmediate={() => handleStop('immediate')}
          onStopSummarize={() => handleStop('summarize')}
        />
      )}

      {/* Stop Confirmation Dialog */}
      {showStopDialog && (
        <StopDialog
          onStopImmediate={() => handleStop('immediate')}
          onStopSummarize={() => handleStop('summarize')}
          onClose={handleStopDialogCancel}
          stopping={stopping}
          language={language}
        />
      )}

      {/* LLM Timeout Dialog */}
      {llmFailureInfo && (
        <LlmTimeoutDialog
          open={showLlmTimeoutDialog}
          llmFailureInfo={llmFailureInfo}
          timeRemaining={timeRemaining}
          hasCountdown={roundWaitSeconds > 0}
          onRetry={handleLlmRetry}
          onModifyAgent={handleLlmModifyAgent}
          onTerminate={handleLlmTerminate}
          onSkip={handleLlmSkip}
          onClose={() => setShowLlmTimeoutDialog(false)}
          language={language}
        />
      )}

      {/* LLM Modify Sub-Dialog */}
      {llmModifySaved && llmFailureInfo && !showLlmTimeoutDialog && (
        <LlmModifySubDialog
          onResume={handleLlmResumeAfterModify}
          onReturnToSettings={handleLlmReturnToSettings}
          onReturnToTimeout={handleLlmReturnToTimeout}
          language={language}
        />
      )}

      {/* Species Input Dialog */}
      {showSpeciesInputDialog && speciesInputInfo && currentSession && (
        <TextInputDialog
          title={speciesInputInfo.source === 'pi_inference'
            ? t('speciesInputTitle')
            : t('speciesGeneValidationTitle')}
          description={speciesInputInfo.context || ''}
          suggestions={speciesInputInfo.suggestions}
          placeholder={speciesInputInfo.source === 'pi_inference'
            ? t('speciesInputPlaceholder')
            : t('speciesGeneValidationPlaceholder')}
          language={language}
          countdownSeconds={roundWaitSeconds}
          resetKey={speciesInputInfo.source}
          onSubmit={handleSpeciesInput}
          onClose={() => clearSpeciesInputState()}
        />
      )}

      {/* Exploration Failed Dialog (likely a data problem) */}
      {showExplorationFailedDialog && explorationFailedInfo && currentSession && (
        <ExplorationFailedDialog
          open={showExplorationFailedDialog}
          info={explorationFailedInfo}
          language={language}
          onGoToData={handleExplorationFailedGoToData}
          onContinue={handleExplorationFailedContinue}
          onStop={handleExplorationFailedStop}
        />
      )}

      {/* Snapshot failure HITL — Retry re-enters capture only; Skip continues
          without capturing at that boundary (skippable producers only) */}
      {(snapshotV2Enabled || (snapshotFailure?.failure.code || '').startsWith('runtime_persistence_')) && snapshotFailure && !showSnapshotPanel && (
        <SnapshotFailureDialog
          failure={snapshotFailure}
          isZh={language === 'zh'}
          onRetry={() => {
            void handleSnapshotFailureRetry()
          }}
          onProceed={() => {
            void handleSnapshotFailureProceed()
          }}
          onSkip={() => {
            void handleSnapshotFailureSkip()
          }}
          onManage={handleSnapshotFailureManage}
          onTerminate={() => {
            void handleSnapshotFailureTerminate()
          }}
        />
      )}

      {/* Snapshot Panel */}
      {snapshotV2Enabled && showSnapshotPanel && currentSession && (
        <SnapshotPanel
          sessionId={currentSession.id}
          language={language}
          onClose={() => setShowSnapshotPanel(false)}
          onRestore={() => {
            setShowSnapshotPanel(false)
            // Reload page to get restored state
            window.location.reload()
          }}
        />
      )}

      {/* Snapshot Cleanup Reminder */}
      {snapshotV2Enabled && <ConfirmDialog
        open={showSnapshotCleanupReminder}
        title={language === 'zh' ? '快照数量提醒' : 'Snapshot Count Reminder'}
        message={language === 'zh'
          ? `当前会话已有 ${snapshotCount} 个快照，建议清理不需要的快照以释放空间。`
          : `This session has ${snapshotCount} snapshots. Consider cleaning up unused snapshots to free space.`}
        confirmLabel={language === 'zh' ? '去清理' : 'Cleanup'}
        titleIcon={<Camera className="w-5 h-5 text-accent-600" />}
        confirmIcon={<Trash2 className="w-4 h-4" />}
        onConfirm={() => {
          setShowSnapshotCleanupReminder(false)
          setShowSnapshotPanel(true)
        }}
        onCancel={() => setShowSnapshotCleanupReminder(false)}
      />}
    </div>
  )
}
