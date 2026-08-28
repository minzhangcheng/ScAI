import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { BookOpen, Code2, Eye, FlaskConical, Microscope, Palette, Users, Wrench } from 'lucide-react'

import { generateId, useConfig, useSession, useUserConfig } from '../../hooks/useApi'
import { useCountdownTimer } from '../../hooks/useCountdownTimer'
import { useTranslation } from '../../i18n/translations'
import { useAppStore, type Message, type Session } from '../../stores/appStore'
import type { RoundSummary } from '../../types'
import { resultIdentityKey } from '../../utils/roundKey'
import type { AnnotationData } from '../../utils/annotationGate'
import { useAnalysisPanelGateActions } from './actions/useAnalysisPanelGateActions'
import { useAnalysisPanelInterventionActions } from './actions/useAnalysisPanelInterventionActions'
import {
  useAnalysisPanelLifecycleActions,
  useAnalysisPanelStartActions,
} from './actions/useAnalysisPanelLifecycleActions'
import { useAnalysisPanelSnapshotFailureActions } from './actions/useAnalysisPanelSnapshotFailureActions'
import type { ExplorationReportData, PlanReviewData, RoundSummaryContent } from './types'
import { useAnalysisPanelEffects } from './useAnalysisPanelEffects'
import { useSnapshotRetryMonitor } from './useSnapshotRetryMonitor'

const MAX_EXPLORATION_ROUND = 1

function buildRoundSummaries(messages: Message[], currentSession: Session | null): RoundSummary[] {
  const summaries = new Map<string, RoundSummary>()
  for (const message of messages.filter(item => item.agent === 'pi')) {
    if (message.type !== 'round_summary' || typeof message.content !== 'object') continue
    const content = message.content as RoundSummaryContent
    const round = content.round || summaries.size + 1
    const phase = (content.phase as RoundSummary['phase']) || ''
    const lineageId = content.lineage_id || 'main'
    const directionKey = content.direction_key || content.direction || ''
    const identity = resultIdentityKey({
      result_identity_key: content.result_identity_key,
      lineageId, phase, round, directionKey,
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
      lineageId, directionKey, resultIdentityKey: identity,
    })
  }
  for (const result of currentSession?.roundResults || []) {
    const identity = resultIdentityKey(result)
    if (summaries.has(identity)) continue
    summaries.set(identity, {
      round: result.round,
      summary: result.summary || '',
      findings: [
        ...(result.pathologyFindings || []),
        ...(result.bioinformaticsFindings || []),
        ...(result.biochemistryFindings || []),
      ],
      dataRequests: { pathology: [], biochemistry: [], other: [] },
      status: 'completed',
      pathologyFindings: result.pathologyFindings || [],
      bioinformaticsFindings: result.bioinformaticsFindings || [],
      biochemistryFindings: result.biochemistryFindings || [],
      literatureFindings: result.literatureFindings || [],
      phase: result.phase || '',
      integrationCandidates: result.integrationCandidates || [],
      integrationReportSummary: result.integrationReportSummary || '',
      lineageId: result.lineageId,
      directionKey: result.directionKey,
      resultIdentityKey: identity,
    })
  }
  return Array.from(summaries.values()).sort(
    (left, right) => left.round - right.round
      || resultIdentityKey(left).localeCompare(resultIdentityKey(right)),
  )
}

function usePanelDerivedState(messages: Message[], currentSession: Session | null, agentErrors: ReturnType<typeof useAppStore.getState>['agentErrors']) {
  const unresolvedErrors = useMemo(() => agentErrors.filter(error => !error.resolved), [agentErrors])
  const agentLastMessages = useMemo(() => {
    const result: Record<string, Message | undefined> = {}
    for (const message of messages) if (message.agent) result[message.agent] = message
    return result
  }, [messages])
  const roundSummaries = useMemo(
    () => buildRoundSummaries(messages, currentSession),
    [messages, currentSession?.roundResults],
  )
  const getAgentStatus = (agentId: string) => {
    const lastMessage = agentLastMessages[agentId]
    const errors = unresolvedErrors.filter(error => error.agent === agentId)
    if (!lastMessage) return errors.length > 0 ? 'error' : 'idle'
    if (currentSession?.status === 'completed') return 'done'
    if (currentSession?.status === 'generating_report'
      && ['pathology', 'bioinformatics', 'biochemistry', 'literature'].includes(agentId)) return 'done'
    if ([
      'result', 'completed', 'integrated_report', 'detailed_report',
      'graphic_abstract', 'round_summary', 'findings',
    ].includes(lastMessage.type)) return 'done'
    if (errors.length > 0) return lastMessage.type === 'status' ? 'working' : 'error'
    if (lastMessage.type === 'status') return 'working'
    if (lastMessage.type === 'error') return 'error'
    return 'working'
  }
  return { getAgentStatus, roundSummaries, unresolvedErrors }
}

export function useAnalysisPanelController() {
  const store = useAppStore()
  const { startAnalysis, resumeAnalysis, continueRound, pauseAnalysis, stopAnalysis } = useSession()
  const config = useConfig()
  const userConfig = useUserConfig()
  const { t } = useTranslation(store.language)
  const llmFailureInfo = useAppStore(state => state.llmFailureInfo)
  const showLlmTimeoutDialog = useAppStore(state => state.showLlmTimeoutDialog)
  const setShowLlmTimeoutDialog = useAppStore(state => state.setShowLlmTimeoutDialog)
  const llmModifySaved = useAppStore(state => state.llmModifySaved)
  const roundWaitSeconds = useAppStore(state => state.roundWaitSeconds)
  const maxRounds = useAppStore(state => state.maxRounds)
  const annotationGateData = useAppStore(state => state.annotationGateData)
  const speciesInputInfo = useAppStore(state => state.speciesInputInfo)
  const showSpeciesInputDialog = useAppStore(state => state.showSpeciesInputDialog)
  const clearSpeciesInputState = useAppStore(state => state.clearSpeciesInputState)
  const explorationFailedInfo = useAppStore(state => state.explorationFailedInfo)
  const showExplorationFailedDialog = useAppStore(state => state.showExplorationFailedDialog)
  const clearExplorationFailedState = useAppStore(state => state.clearExplorationFailedState)

  const [expandedRound, setExpandedRound] = useState<string | null>(null)
  const [showFeedbackDialog, setShowFeedbackDialog] = useState(false)
  const [feedbackRoundData, setFeedbackRoundData] = useState<RoundSummary | null>(null)
  const lastDialogRoundRef = useRef(-1)
  const [showAnnotationDialog, setShowAnnotationDialog] = useState(false)
  const [annotationData, setAnnotationData] = useState<AnnotationData | null>(null)
  const [showPlanReviewDialog, setShowPlanReviewDialog] = useState(false)
  const [planReviewData, setPlanReviewData] = useState<PlanReviewData | null>(null)
  const lastAnnotationMsgRef = useRef('')
  const lastPlanReviewMsgRef = useRef('')
  const [showExplorationDialog, setShowExplorationDialog] = useState(false)
  const [explorationReportData, setExplorationReportData] = useState<ExplorationReportData | null>(null)
  const lastExplorationMsgRef = useRef('')
  const [integrationGateData, setIntegrationGateData] = useState<any>(null)
  const [showIntegrationGateDialog, setShowIntegrationGateDialog] = useState(false)
  const lastIntegrationMsgRef = useRef('')
  const [isSupplementaryRunning, setIsSupplementaryRunning] = useState(false)
  const roundSupplementaryPendingRef = useRef(false)
  const [showSnapshotPanel, setShowSnapshotPanel] = useState(false)
  const snapshotV2Enabled = useAppStore(state => state.snapshotV2Enabled)
  const snapshotCount = useAppStore(state => state.snapshotCount)
  const showSnapshotCleanupReminder = useAppStore(state => state.showSnapshotCleanupReminder)
  const setShowSnapshotCleanupReminder = useAppStore(state => state.setShowSnapshotCleanupReminder)
  const snapshotFailure = useAppStore(state => state.snapshotFailure)
  const clearSnapshotFailure = useAppStore(state => state.clearSnapshotFailure)
  const runtimeRestoreStatus = useAppStore(state => state.runtimeRestoreStatus)
  const runtimeRestoreWarning = useAppStore(state => state.runtimeRestoreWarning)
  const isForwardResumeBlocked = useAppStore(state => state.isForwardResumeBlocked)
  const snapshotTransaction = useAppStore(state => state.snapshotTransaction)

  const snapshotMonitor = useSnapshotRetryMonitor({
    currentSessionId: store.currentSession?.id ?? null,
    snapshotV2Enabled,
    snapshotTransaction,
    setShowSnapshotPanel,
  })
  useEffect(() => { void config.checkStatus() }, [config.checkStatus])
  useEffect(() => {
    if (snapshotV2Enabled) return
    setShowSnapshotPanel(false)
    snapshotMonitor.stopSnapshotRetryPolling()
    useAppStore.getState().setShowSnapshotCleanupReminder(false)
  }, [snapshotV2Enabled, snapshotMonitor.stopSnapshotRetryPolling])
  const isRerunEnabled = useAppStore(state => state.isRerunEnabled)
  useEffect(() => {
    lastAnnotationMsgRef.current = ''
    lastExplorationMsgRef.current = ''
  }, [store.currentSession?.id])

  const [showStopDialog, setShowStopDialog] = useState(false)
  const [stopping, setStopping] = useState(false)
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
  const derived = usePanelDerivedState(store.messages, store.currentSession, store.agentErrors)
  const notifySyncFailures = useCallback((failures: string[]) => {
    if (failures.length === 0) return
    const label = store.language === 'zh' ? '以下设置同步失败，将使用默认值' : 'Settings sync failed, using defaults for'
    store.addMessage({
      id: generateId(), type: 'error', content: `${label}: ${failures.join(', ')}`,
      timestamp: new Date().toISOString(),
    })
  }, [store.language, store.addMessage])

  const gateActions = useAnalysisPanelGateActions({
    currentSession: store.currentSession, wsConnected: store.wsConnected, wsConnect: store.wsConnect,
    setShowAnnotationDialog, setShowPlanReviewDialog, setShowExplorationDialog,
    setShowIntegrationGateDialog, setIntegrationGateData, setIsSupplementaryRunning,
    roundSupplementaryPendingRef,
  })
  const startActions = useAnalysisPanelStartActions({
    readMode: store.readMode, translatedReadModeRestriction: t('readModeRestriction'),
    translatedApiKeyInvalid: t('apiKeyInvalid'), authRole: store.authUser?.role,
    showConfigPanel: store.showConfigPanel, checkStatus: config.checkStatus,
    saveConfig: config.saveConfig, validateApiKey: config.validateApiKey,
    validateSavedConfig: config.validateSavedConfig,
    validateUserSavedConfig: userConfig.validateSavedConfig,
    startAnalysis, notifySyncFailures, wsConnect: store.wsConnect, addMessage: store.addMessage,
    setValidating, setShowValidationDialog, setValidationMessage, setValidationUncertain,
  })
  const lifecycleActions = useAnalysisPanelLifecycleActions({
    currentSession: store.currentSession, language: store.language,
    isForwardResumeBlocked, addMessage: store.addMessage, resumeAnalysis, continueRound,
    pauseAnalysis, stopAnalysis, wsConnected: store.wsConnected, wsConnect: store.wsConnect,
    setShowFeedbackDialog, setStopping, setShowStopDialog,
  })
  const snapshotActions = useAnalysisPanelSnapshotFailureActions({
    currentSession: store.currentSession, snapshotFailure, clearSnapshotFailure,
    monitorSnapshotRetry: snapshotMonitor.monitorSnapshotRetry,
    snapshotAutoOpenedForRef: snapshotMonitor.snapshotAutoOpenedForRef,
    setShowSnapshotPanel, wsConnected: store.wsConnected, wsConnect: store.wsConnect,
    addMessage: store.addMessage, language: store.language,
  })
  const interventionActions = useAnalysisPanelInterventionActions({
    currentSession: store.currentSession, clearExplorationFailedState, clearSpeciesInputState,
    setShowLlmTimeoutDialog, setShowStopDialog, handleStop: lifecycleActions.handleStop,
    annotationData, annotationGateData, setAnnotationData, setShowAnnotationDialog,
    integrationGateData, setShowIntegrationGateDialog, explorationReportData,
    setShowExplorationDialog, feedbackRoundData, setShowFeedbackDialog,
    setShowPlanReviewDialog, wsConnect: store.wsConnect, maxExplorationRound: MAX_EXPLORATION_ROUND,
  })
  const { timeRemaining } = useCountdownTimer({
    countdownSeconds: roundWaitSeconds > 0 ? roundWaitSeconds : 0,
    onExpire: interventionActions.handleLlmSkip,
    resetKey: llmFailureInfo?.generation != null ? String(llmFailureInfo.generation) : undefined,
  })
  const isCurrentIntegrationGate = store.currentSession?.status === 'waiting_input'
    && store.currentSession?.analysisPhase === 'integration'
  useAnalysisPanelEffects({
    currentSession: store.currentSession, messages: store.messages,
    wsConnected: store.wsConnected, wsConnect: store.wsConnect,
    roundSummaries: derived.roundSummaries, showFeedbackDialog, setShowFeedbackDialog,
    setExpandedRound, setFeedbackRoundData, lastDialogRoundRef,
    roundSupplementaryPendingRef, isSupplementaryRunning, setIsSupplementaryRunning,
    annotationGateData, annotationData, setAnnotationData, showAnnotationDialog,
    setShowAnnotationDialog, lastAnnotationMsgRef, showPlanReviewDialog,
    setShowPlanReviewDialog, setPlanReviewData, lastPlanReviewMsgRef,
    showExplorationDialog, setShowExplorationDialog, setExplorationReportData,
    lastExplorationMsgRef, isCurrentIntegrationGate, setShowIntegrationGateDialog,
    setIntegrationGateData, lastIntegrationMsgRef,
  })

  return {
    ...store, t, llmFailureInfo, showLlmTimeoutDialog, llmModifySaved, roundWaitSeconds, maxRounds,
    annotationGateData, speciesInputInfo, showSpeciesInputDialog, explorationFailedInfo,
    showExplorationFailedDialog, expandedRound, setExpandedRound, showFeedbackDialog,
    setShowFeedbackDialog, feedbackRoundData, showAnnotationDialog, setShowAnnotationDialog,
    annotationData, showPlanReviewDialog, setShowPlanReviewDialog, planReviewData,
    showExplorationDialog, setShowExplorationDialog, explorationReportData,
    integrationGateData, showIntegrationGateDialog, setShowIntegrationGateDialog,
    isSupplementaryRunning, showSnapshotPanel, setShowSnapshotPanel, snapshotV2Enabled,
    snapshotCount, showSnapshotCleanupReminder, setShowSnapshotCleanupReminder,
    snapshotFailure, runtimeRestoreStatus, runtimeRestoreWarning, isRerunEnabled,
    showStopDialog, setShowStopDialog, stopping, validating, showValidationDialog,
    setShowValidationDialog, validationMessage, validationUncertain, agents,
    roundSummaries: derived.roundSummaries, getAgentStatus: derived.getAgentStatus,
    unresolvedErrors: derived.unresolvedErrors, timeRemaining, isCurrentIntegrationGate,
    ...gateActions, ...startActions, ...lifecycleActions, ...snapshotActions, ...interventionActions,
  }
}

export type AnalysisPanelController = ReturnType<typeof useAnalysisPanelController>
