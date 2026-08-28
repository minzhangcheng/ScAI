import { useEffect, type MutableRefObject } from 'react'

import type { RoundSummary } from '../../types'
import type { AnnotationData } from '../../utils/annotationGate'
import { resultIdentityKey } from '../../utils/roundKey'
import type { Message, Session } from '../../stores/appStore'
import type { ExplorationReportData, PlanReviewData } from './types'

interface AnalysisPanelEffectsOptions {
  currentSession: Session | null
  messages: Message[]
  wsConnected: boolean
  wsConnect: (() => void) | null
  roundSummaries: RoundSummary[]
  showFeedbackDialog: boolean
  setShowFeedbackDialog: (show: boolean) => void
  setExpandedRound: (round: string | null) => void
  setFeedbackRoundData: (data: RoundSummary | null) => void
  lastDialogRoundRef: MutableRefObject<number>
  roundSupplementaryPendingRef: MutableRefObject<boolean>
  isSupplementaryRunning: boolean
  setIsSupplementaryRunning: (running: boolean) => void
  annotationGateData: AnnotationData | null
  annotationData: AnnotationData | null
  setAnnotationData: (data: AnnotationData | null) => void
  showAnnotationDialog: boolean
  setShowAnnotationDialog: (show: boolean) => void
  lastAnnotationMsgRef: MutableRefObject<string>
  showPlanReviewDialog: boolean
  setShowPlanReviewDialog: (show: boolean) => void
  setPlanReviewData: (data: PlanReviewData | null) => void
  lastPlanReviewMsgRef: MutableRefObject<string>
  showExplorationDialog: boolean
  setShowExplorationDialog: (show: boolean) => void
  setExplorationReportData: (data: ExplorationReportData | null) => void
  lastExplorationMsgRef: MutableRefObject<string>
  isCurrentIntegrationGate: boolean
  setShowIntegrationGateDialog: (show: boolean) => void
  setIntegrationGateData: (update: any) => void
  lastIntegrationMsgRef: MutableRefObject<string>
}

export function useAnalysisPanelEffects(options: AnalysisPanelEffectsOptions) {
  useEffect(() => {
    if (options.currentSession && (
      options.currentSession.status === 'running'
      || options.currentSession.status === 'waiting_input'
      || options.currentSession.status === 'waiting_annotation'
      || options.currentSession.status === 'waiting_plan_review'
      || options.currentSession.status === 'waiting_species_input'
    ) && !options.wsConnected && options.wsConnect) options.wsConnect()
  }, [options.currentSession?.id, options.currentSession?.status, options.wsConnected, options.wsConnect])

  useEffect(() => {
    if (options.roundSummaries.length > 0) {
      const latest = options.roundSummaries[options.roundSummaries.length - 1]
      options.setExpandedRound(resultIdentityKey(latest))
      if (latest.status === 'waiting_input'
        && options.currentSession?.status === 'waiting_input'
        && latest.phase !== 'integration'
        && latest.round !== options.lastDialogRoundRef.current) {
        options.lastDialogRoundRef.current = latest.round
        options.setFeedbackRoundData(latest)
        options.setShowFeedbackDialog(true)
      }
    }
  }, [options.roundSummaries, options.currentSession?.status])

  useEffect(() => {
    if (options.currentSession?.status === 'running' && options.showFeedbackDialog) {
      options.setShowFeedbackDialog(false)
    }
  }, [options.currentSession?.status, options.showFeedbackDialog])

  useEffect(() => {
    if (options.currentSession?.status !== 'waiting_input') return
    if (!options.roundSupplementaryPendingRef.current) return
    options.roundSupplementaryPendingRef.current = false
    options.setIsSupplementaryRunning(false)
    options.setShowFeedbackDialog(true)
  }, [options.currentSession?.status])

  useEffect(() => {
    if (options.currentSession?.status !== 'waiting_annotation') {
      if (options.isSupplementaryRunning) options.setIsSupplementaryRunning(false)
      return
    }
    let annotationMessage: Message | undefined
    for (let index = options.messages.length - 1; index >= 0; index -= 1) {
      if (options.messages[index].type === 'annotation_confirmation') {
        annotationMessage = options.messages[index]
        break
      }
    }
    if (annotationMessage && annotationMessage.id !== options.lastAnnotationMsgRef.current) {
      options.lastAnnotationMsgRef.current = annotationMessage.id
      options.setAnnotationData(annotationMessage.content as AnnotationData)
      options.setIsSupplementaryRunning(false)
      if (!options.showAnnotationDialog) options.setShowAnnotationDialog(true)
    } else if (!annotationMessage && !options.annotationData && options.annotationGateData) {
      options.setAnnotationData(options.annotationGateData)
      options.setShowAnnotationDialog(true)
    }
  }, [options.messages, options.currentSession?.status, options.annotationGateData, options.annotationData])

  useEffect(() => {
    if (options.currentSession?.status !== 'waiting_plan_review') return
    let planMessage: Message | undefined
    for (let index = options.messages.length - 1; index >= 0; index -= 1) {
      if (options.messages[index].type === 'plan_review') {
        planMessage = options.messages[index]
        break
      }
    }
    if (planMessage && planMessage.id !== options.lastPlanReviewMsgRef.current) {
      options.lastPlanReviewMsgRef.current = planMessage.id
      options.setPlanReviewData(planMessage.content as PlanReviewData)
      if (!options.showPlanReviewDialog) options.setShowPlanReviewDialog(true)
    }
  }, [options.messages, options.currentSession?.status, options.showPlanReviewDialog])

  useEffect(() => {
    for (let index = options.messages.length - 1; index >= 0; index -= 1) {
      if (options.messages[index].type === 'plan_approved') {
        if (options.showPlanReviewDialog) options.setShowPlanReviewDialog(false)
        return
      }
    }
  }, [options.messages])

  useEffect(() => {
    if (options.currentSession?.status !== 'waiting_plan_review'
      && options.currentSession?.status !== 'running') {
      return
    }
    if (options.currentSession?.status === 'running' && options.showPlanReviewDialog) {
      options.setShowPlanReviewDialog(false)
    }
  }, [options.currentSession?.status, options.showPlanReviewDialog])

  useEffect(() => {
    let explorationMessage: Message | undefined
    for (let index = options.messages.length - 1; index >= 0; index -= 1) {
      if (options.messages[index].type === 'exploration_report') {
        explorationMessage = options.messages[index]
        break
      }
    }
    if (explorationMessage
      && explorationMessage.id !== options.lastExplorationMsgRef.current
      && options.currentSession?.status === 'waiting_input') {
      if ((options.currentSession.currentRound ?? 0) > 1) {
        options.lastExplorationMsgRef.current = explorationMessage.id
        return
      }
      options.lastExplorationMsgRef.current = explorationMessage.id
      options.setExplorationReportData(explorationMessage.content as ExplorationReportData)
      options.setIsSupplementaryRunning(false)
      if (!options.showExplorationDialog) {
        options.setShowExplorationDialog(true)
        options.setShowFeedbackDialog(false)
      }
    }
  }, [options.messages, options.currentSession?.status, options.currentSession?.currentRound])

  useEffect(() => {
    if (!options.isCurrentIntegrationGate) {
      options.setShowIntegrationGateDialog(false)
      options.setIntegrationGateData((previous: any) => previous ? null : previous)
      return
    }
    let message: Message | null = null
    for (let index = options.messages.length - 1; index >= 0; index -= 1) {
      if (options.messages[index].type === 'integration_gate') {
        message = options.messages[index]
        break
      }
    }
    if (message && options.lastIntegrationMsgRef.current !== message.id) {
      options.lastIntegrationMsgRef.current = message.id
      options.setIntegrationGateData(message.content)
      options.setShowIntegrationGateDialog(true)
    }
  }, [options.messages, options.isCurrentIntegrationGate])

  useEffect(() => {
    if (options.currentSession?.status === 'running') {
      if (options.showAnnotationDialog) options.setShowAnnotationDialog(false)
      if (options.showExplorationDialog) options.setShowExplorationDialog(false)
    }
  }, [options.currentSession?.status, options.showAnnotationDialog, options.showExplorationDialog])
}
