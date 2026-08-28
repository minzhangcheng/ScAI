import { useCallback, useEffect, useRef, useState } from 'react'

import { useCountdownTimer } from '../../hooks/useCountdownTimer'
import { useTranslation } from '../../i18n/translations'
import { useAppStore } from '../../stores/appStore'

interface FigureRequirement {
  figure_type: string
  description: string
  minimum_count: number
  platform_baseline?: '' | 'annotation' | 'batch' | 'cluster' | 'markers'
}

interface PlanStep {
  id: number
  title: string
  method: string
  input: string
  expected_output: string
  rationale: string
  figure_requirements?: FigureRequirement[]
  allow_text_only?: boolean
  text_only_reason?: string
  depends_on: number[]
  status: string
}

export interface AnalysisPlan {
  id: string
  phase: string
  stage: string
  overview: string
  steps: PlanStep[]
  key_decision_points: string[]
  expected_runtime: string
  status: string
}

export interface PlanReviewDialogProps {
  open: boolean
  plan: AnalysisPlan | null
  dialogueHtml: string
  phase: string
  round: number
  stage: string
  countdownSeconds: number
  onApprove: () => Promise<void>
  onSkip: () => Promise<void>
  onModify: (userInput: string) => Promise<void>
}

export function usePlanReviewController({
  open,
  plan,
  dialogueHtml,
  phase,
  round,
  stage,
  countdownSeconds,
  onApprove,
  onSkip,
  onModify,
}: PlanReviewDialogProps) {
  const [userInput, setUserInput] = useState('')
  const language = useAppStore(s => s.language)
  const { t } = useTranslation(language)
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())
  const [submitting, setSubmitting] = useState(false)
  const [showModifyInput, setShowModifyInput] = useState(false)
  const expiredRef = useRef(false)
  useEffect(() => {
    if (open) {
      expiredRef.current = false
    }
  }, [open, countdownSeconds])
  const handleAutoSkip = useCallback(() => {
    if (!submitting && !expiredRef.current) {
      expiredRef.current = true
      onSkip()
    }
  }, [onSkip, submitting])
  const planFingerprint = plan
    ? `${plan.id}|${plan.status}|${plan.stage}|${plan.steps.map(s => `${s.id}:${s.status}`).join(',')}`
    : 'no-plan'
  const { timeRemaining: countdown, isPaused, pause, resume, hasCountdown } = useCountdownTimer({
    countdownSeconds,
    onExpire: handleAutoSkip,
    resetKey: `${open}-${countdownSeconds}-${planFingerprint}`,
  })
  const minutes = Math.floor(countdown / 60)
  const seconds = countdown % 60
  const timerDisplay = `${minutes}:${seconds.toString().padStart(2, '0')}`
  const timerUrgent = countdown > 0 && countdown <= 30
  const toggleTimerPause = useCallback(() => {
    if (isPaused) resume()
    else pause()
  }, [isPaused, pause, resume])
  const phaseLabel = phase === 'exploration' ? t('planReviewStageExplore') : t('planReviewStageInnovate')
  const stageLabel = stage === 'stage1' ? t('planReviewStage1')
                     : stage === 'stage2' ? t('planReviewStage2') : t('planReviewStageFull')
  const toggleStep = (stepId: number) => {
    setExpandedSteps(prev => {
      const next = new Set(prev)
      if (next.has(stepId)) {
        next.delete(stepId)
      } else {
        next.add(stepId)
      }
      return next
    })
  }
  const handleDownloadDialogue = () => {
    if (!dialogueHtml) return
    const blob = new Blob([dialogueHtml], { type: 'text/html' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `plan-dialogue-round${round}-${stage}.html`
    a.click()
    URL.revokeObjectURL(url)
  }
  const handleApprove = async () => {
    if (expiredRef.current || submitting) return
    expiredRef.current = true
    setSubmitting(true)
    try {
      await onApprove()
    } catch {
      expiredRef.current = false
    } finally {
      setSubmitting(false)
    }
  }
  const handleSkip = async () => {
    if (expiredRef.current || submitting) return
    expiredRef.current = true
    setSubmitting(true)
    try {
      await onSkip()
    } catch {
      expiredRef.current = false
    } finally {
      setSubmitting(false)
    }
  }
  const handleSubmitModify = async () => {
    if (!userInput.trim()) return
    setSubmitting(true)
    try {
      await onModify(userInput.trim())
    } finally {
      setSubmitting(false)
      setShowModifyInput(false)
    }
  }
  return {
    dialogueHtml, expandedSteps, handleApprove, handleDownloadDialogue, handleSkip,
    handleSubmitModify, hasCountdown, isPaused, phaseLabel, plan, round,
    setShowModifyInput, setUserInput, showModifyInput, stageLabel, submitting, t,
    timerDisplay, timerUrgent, toggleStep, toggleTimerPause, userInput,
  }
}

export type PlanReviewController = ReturnType<typeof usePlanReviewController>
