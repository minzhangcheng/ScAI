import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { BookOpen, Code2, Eye, FlaskConical, Microscope, Palette, Users, Wrench } from 'lucide-react'

import { generateId, useConfig, useSession, useUserConfig } from '../../hooks/useApi'
import { useCountdownTimer } from '../../hooks/useCountdownTimer'
import { useTranslation } from '../../i18n/translations'
import { useAppStore, type Message, type Session } from '../../stores/appStore'
import type { RoundSummary } from '../../types'
import { resultIdentityKey } from '../../utils/roundKey'
import { chatOwnerKey } from '../../utils/chatOwnership'
import type { AnnotationData } from '../../utils/annotationGate'
import { useAnalysisPanelGateActions } from './actions/useAnalysisPanelGateActions'
import { useAnalysisPanelInterventionActions } from './actions/useAnalysisPanelInterventionActions'
import {
  useAnalysisPanelLifecycleActions,
  useAnalysisPanelStartActions,
} from './actions/useAnalysisPanelLifecycleActions'
import type { ExplorationReportData, PlanReviewData, RoundSummaryContent } from './types'
import { useAnalysisPanelEffects } from './useAnalysisPanelEffects'

const MAX_EXPLORATION_ROUND = 1

/** Transient UI state owned by one exact session/branch pair. */
function useOwnedValue<T>(initialValue: T, ownerKey: string | null) {
  const initialRef = useRef(initialValue)
  const [entry, setEntry] = useState<{ ownerKey: string | null; value: T }>(
    () => ({ ownerKey, value: initialValue }),
  )
  // Replace the state owner after a session/branch transition. The render
  // boundary below hides the old value immediately; this effect makes the
  // next owner's setter usable once the transition has committed, while an
  // old callback retained by the previous owner remains rejected.
  useEffect(() => {
    setEntry(previous => (
      previous.ownerKey === ownerKey
        ? previous
        : { ownerKey, value: initialValue }
    ))
  }, [ownerKey, initialValue])
  const setValue = useCallback((update: T | ((previous: T) => T)) => {
    setEntry(previous => {
      // The setter itself is owner-bound. A callback retained by the source
      // dialog must become inert after a branch/session switch, even after
      // the replacement owner's state has been installed.
      if (previous.ownerKey !== ownerKey) return previous
      const base = previous.value
      const value = typeof update === 'function'
        ? (update as (previous: T) => T)(base)
        : update
      return { ownerKey, value }
    })
  }, [ownerKey])
  return [
    entry.ownerKey === ownerKey ? entry.value : initialRef.current,
    setValue,
  ] as const
}

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
  const annotationGateDataRaw = useAppStore(state => state.annotationGateData)
  const annotationGateOwnerKey = useAppStore(state => state.annotationGateOwnerKey)
  const speciesInputInfo = useAppStore(state => state.speciesInputInfo)
  const speciesInputOwnerKey = useAppStore(state => state.speciesInputOwnerKey)
  const showSpeciesInputDialog = useAppStore(state => state.showSpeciesInputDialog)
  const clearSpeciesInputState = useAppStore(state => state.clearSpeciesInputState)
  const explorationFailedInfo = useAppStore(state => state.explorationFailedInfo)
  const explorationFailedOwnerKey = useAppStore(state => state.explorationFailedOwnerKey)
  const showExplorationFailedDialog = useAppStore(state => state.showExplorationFailedDialog)
  const clearExplorationFailedState = useAppStore(state => state.clearExplorationFailedState)

  const gateOwnerKey = chatOwnerKey(
    store.currentSession?.id, store.currentSession?.restoreBranchId,
  )
  // Gate-derived values kept in the global store are still scoped to the
  // exact session/branch owner. The owner check happens during render, before
  // any cleanup effect can run, so a branch switch cannot briefly expose the
  // source dialog or submit its payload under the child owner.
  const annotationGateData = annotationGateOwnerKey === gateOwnerKey
    ? annotationGateDataRaw
    : null
  const ownedSpeciesInputInfo = speciesInputOwnerKey === gateOwnerKey
    ? speciesInputInfo
    : null
  const ownedShowSpeciesInputDialog = speciesInputOwnerKey === gateOwnerKey
    && showSpeciesInputDialog
  const ownedExplorationFailedInfo = explorationFailedOwnerKey === gateOwnerKey
    ? explorationFailedInfo
    : null
  const ownedShowExplorationFailedDialog = explorationFailedOwnerKey === gateOwnerKey
    && showExplorationFailedDialog

  const [expandedRound, setExpandedRound] = useOwnedValue<string | null>(
    null, gateOwnerKey,
  )
  const [showFeedbackDialog, setShowFeedbackDialog] = useOwnedValue(false, gateOwnerKey)
  const [feedbackRoundData, setFeedbackRoundData] = useOwnedValue<RoundSummary | null>(null, gateOwnerKey)
  const lastDialogRoundRef = useRef(-1)
  const [showAnnotationDialog, setShowAnnotationDialog] = useOwnedValue(false, gateOwnerKey)
  const [annotationData, setAnnotationData] = useOwnedValue<AnnotationData | null>(null, gateOwnerKey)
  const [showPlanReviewDialog, setShowPlanReviewDialog] = useOwnedValue(false, gateOwnerKey)
  const [planReviewData, setPlanReviewData] = useOwnedValue<PlanReviewData | null>(null, gateOwnerKey)
  const lastAnnotationMsgRef = useRef('')
  const lastPlanReviewMsgRef = useRef('')
  const [showExplorationDialog, setShowExplorationDialog] = useOwnedValue(false, gateOwnerKey)
  const [explorationReportData, setExplorationReportData] = useOwnedValue<ExplorationReportData | null>(null, gateOwnerKey)
  const lastExplorationMsgRef = useRef('')
  const [integrationGateData, setIntegrationGateData] = useOwnedValue<any>(null, gateOwnerKey)
  const [showIntegrationGateDialog, setShowIntegrationGateDialog] = useOwnedValue(false, gateOwnerKey)
  const lastIntegrationMsgRef = useRef('')
  const [isSupplementaryRunning, setIsSupplementaryRunning] = useOwnedValue(false, gateOwnerKey)
  const roundSupplementaryPendingRef = useRef(false)
  const runtimeRestoreStatus = useAppStore(state => state.runtimeRestoreStatus)
  const runtimeRestoreWarning = useAppStore(state => state.runtimeRestoreWarning)
  const isForwardResumeBlocked = useAppStore(state => state.isForwardResumeBlocked)

  useEffect(() => { void config.checkStatus() }, [config.checkStatus])
  const isRerunEnabled = useAppStore(state => state.isRerunEnabled)
  useEffect(() => {
    lastAnnotationMsgRef.current = ''
    lastPlanReviewMsgRef.current = ''
    lastExplorationMsgRef.current = ''
    lastIntegrationMsgRef.current = ''
    lastDialogRoundRef.current = -1
    roundSupplementaryPendingRef.current = false
  }, [gateOwnerKey])

  const [showStopDialog, setShowStopDialog] = useOwnedValue(false, gateOwnerKey)
  const [stopping, setStopping] = useOwnedValue(false, gateOwnerKey)
  const [validating, setValidating] = useOwnedValue(false, gateOwnerKey)
  const [showValidationDialog, setShowValidationDialog] = useOwnedValue(false, gateOwnerKey)
  const [validationMessage, setValidationMessage] = useOwnedValue('', gateOwnerKey)
  const [validationUncertain, setValidationUncertain] = useOwnedValue(false, gateOwnerKey)
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
    currentSession: store.currentSession,
    stateVersion: store.stateVersion,
    wsConnected: store.wsConnected, wsConnect: store.wsConnect,
    setShowAnnotationDialog, setShowPlanReviewDialog, setShowExplorationDialog,
    setShowIntegrationGateDialog, setIntegrationGateData, setIsSupplementaryRunning,
    roundSupplementaryPendingRef,
    onGateActionError: (error, action) => {
      const detail = error instanceof Error ? error.message : String(error)
      store.addMessage({
        id: generateId(),
        type: 'error',
        content: `${action}: ${detail}`,
        timestamp: new Date().toISOString(),
      })
    },
  })
  const startActions = useAnalysisPanelStartActions({
    ownerKey: gateOwnerKey,
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
    annotationGateData, speciesInputInfo: ownedSpeciesInputInfo,
    showSpeciesInputDialog: ownedShowSpeciesInputDialog,
    explorationFailedInfo: ownedExplorationFailedInfo,
    showExplorationFailedDialog: ownedShowExplorationFailedDialog,
    expandedRound, setExpandedRound, showFeedbackDialog,
    setShowFeedbackDialog, feedbackRoundData, showAnnotationDialog, setShowAnnotationDialog,
    annotationData, showPlanReviewDialog, setShowPlanReviewDialog, planReviewData,
    showExplorationDialog, setShowExplorationDialog, explorationReportData,
    integrationGateData, showIntegrationGateDialog, setShowIntegrationGateDialog,
    isSupplementaryRunning, runtimeRestoreStatus, runtimeRestoreWarning, isRerunEnabled,
    showStopDialog, setShowStopDialog, stopping, validating, showValidationDialog,
    setShowValidationDialog, validationMessage, validationUncertain, agents,
    roundSummaries: derived.roundSummaries, getAgentStatus: derived.getAgentStatus,
    unresolvedErrors: derived.unresolvedErrors, timeRemaining, isCurrentIntegrationGate,
    ...gateActions, ...startActions, ...lifecycleActions, ...interventionActions,
  }
}

export type AnalysisPanelController = ReturnType<typeof useAnalysisPanelController>
