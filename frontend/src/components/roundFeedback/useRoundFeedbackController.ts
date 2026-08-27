import { useCallback, useEffect, useRef, useState } from 'react'

import { useClickOutside } from '../../hooks/useClickOutside'
import { useCountdownTimer } from '../../hooks/useCountdownTimer'
import { useTranslation, type Language } from '../../i18n/translations'
import type { TreatmentGroup } from '../../stores/appStore'
import type { RoundSummary } from '../../types'
import {
  CUSTOM_TYPE_SENTINEL, getSection, getTypeOptions, hasCustomOption,
} from '../../utils/dataTypes'

export interface UploadedFileWithAgent {
  file: File
  targetAgent: string
  groupId?: string
  sampleNumber?: number
  dataType?: string
}

export interface RoundFeedbackDialogProps {
  roundNumber: number
  maxRounds: number
  roundSummary: RoundSummary
  language: Language
  countdownSeconds: number  // -1=manual skip only, 0=no wait (auto-skip), positive=countdown
  treatmentGroups: TreatmentGroup[]
  onSend: (message: string, files: UploadedFileWithAgent[]) => Promise<void>
  onSkip: () => Promise<void>
  onRequestSupplementary: (requestText: string) => Promise<void>
  onStopImmediate?: () => void
  onStopSummarize?: () => void
  isSupplementaryRunning?: boolean
}

export function useRoundFeedbackController({
  roundNumber,
  maxRounds,
  roundSummary,
  language,
  countdownSeconds,
  treatmentGroups,
  onSend,
  onSkip,
  onRequestSupplementary,
  onStopImmediate,
  onStopSummarize,
  isSupplementaryRunning = false,
}: RoundFeedbackDialogProps) {
  const { t } = useTranslation(language)
  const isManualMode = countdownSeconds === -1
  const isNoWait = countdownSeconds === 0
  const [userInput, setUserInput] = useState('')
  const [files, setFiles] = useState<UploadedFileWithAgent[]>([])
  const [sending, setSending] = useState(false)
  const [customTypeError, setCustomTypeError] = useState('')
  const [showSupplementaryInput, setShowSupplementaryInput] = useState(false)
  const [timerPaused, setTimerPaused] = useState(false)
  const hasTypedRef = useRef(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { timeRemaining, isPaused, pause, resume, hasCountdown } = useCountdownTimer({
    countdownSeconds,
  })
  const [openDropdownIndex, setOpenDropdownIndex] = useState<number | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const isLastRound = roundNumber >= (roundSummary.maxRounds ?? maxRounds)
  const onSkipRef = useRef(onSkip)
  onSkipRef.current = onSkip
  useEffect(() => {
    if (isNoWait && !sending) {
      onSkipRef.current()
    }
  }, [isNoWait, sending])
  useEffect(() => {
    setTimerPaused(isPaused)
  }, [isPaused])
  useEffect(() => {
    if (!hasCountdown) return
    if (timerPaused) {
      pause()
    } else {
      resume()
    }
  }, [timerPaused, hasCountdown, pause, resume])
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setUserInput(e.target.value)
    if (!hasTypedRef.current && e.target.value.length > 0) {
      hasTypedRef.current = true
      setTimerPaused(true)
    }
  }, [])
  const toggleTimerPause = useCallback(() => {
    setTimerPaused(prev => !prev)
  }, [])
  const handleSkip = useCallback(async () => {
    await onSkip()
  }, [onSkip])
  const openSupplementaryInput = useCallback(() => {
    setTimerPaused(true)
    setShowSupplementaryInput(true)
  }, [])
  const handleSend = useCallback(async () => {
    if (sending) return
    if (!userInput.trim() && files.length === 0) return
    const invalidFile = files.find(f =>
      hasCustomOption(getTypeOptions(f.targetAgent)) &&
      (f.dataType === CUSTOM_TYPE_SENTINEL || !f.dataType)
    )
    if (invalidFile) {
      setCustomTypeError(t('customTypeRequired'))
      return
    }
    setSending(true)
    try {
      await onSend(userInput.trim(), files)
    } catch (error) {
      console.error('Failed to send input:', error)
    } finally {
      setSending(false)
    }
  }, [userInput, files, sending, onSend])
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files
    if (!selectedFiles) return
    const newFiles: UploadedFileWithAgent[] = Array.from(selectedFiles).map(file => ({
      file,
      targetAgent: 'pi',
    }))
    setFiles(prev => [...prev, ...newFiles])
    if (!hasTypedRef.current) {
      hasTypedRef.current = true
      setTimerPaused(true)
    }
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])
  const removeFile = useCallback((index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }, [])
  const updateFileAgent = useCallback((index: number, agent: string) => {
    const types = getTypeOptions(agent)
    setFiles(prev => prev.map((f, i) => i === index
      ? { ...f, targetAgent: agent, groupId: undefined, sampleNumber: undefined, dataType: types[0] || undefined }
      : f))
    setOpenDropdownIndex(null)
  }, [])
  const updateFileGroup = useCallback((index: number, groupId: string | undefined) => {
    setFiles(prev => prev.map((f, i) => i === index
      ? { ...f, groupId, sampleNumber: undefined }
      : f))
  }, [])
  const updateFileSample = useCallback((index: number, sampleNumber: number | undefined) => {
    setFiles(prev => prev.map((f, i) => i === index
      ? { ...f, sampleNumber }
      : f))
  }, [])
  const updateFileType = useCallback((index: number, dataType: string) => {
    setFiles(prev => prev.map((f, i) => i === index
      ? { ...f, dataType }
      : f))
  }, [])
  useClickOutside(
    [dropdownRef],
    () => setOpenDropdownIndex(null),
    { enabled: openDropdownIndex !== null, eventType: 'click' },
  )
  const minutes = Math.floor(timeRemaining / 60)
  const seconds = timeRemaining % 60
  const timerDisplay = `${minutes}:${seconds.toString().padStart(2, '0')}`
  const timerUrgent = timeRemaining <= 30
  const pathologyFindings = roundSummary.pathologyFindings || []
  const bioinformaticsFindings = roundSummary.bioinformaticsFindings || []
  const biochemistryFindings = roundSummary.biochemistryFindings || []
  const literatureFindings = roundSummary.literatureFindings || []
  const hasDataRequests = roundSummary.dataRequests && (
    roundSummary.dataRequests.pathology?.length > 0 ||
    roundSummary.dataRequests.biochemistry?.length > 0 ||
    roundSummary.dataRequests.other?.length > 0
  )
  return {
    bioinformaticsFindings, biochemistryFindings, customTypeError, dropdownRef, fileInputRef,
    files, handleFileSelect, handleInputChange, handleSend, handleSkip, hasCountdown,
    hasDataRequests, isLastRound, isManualMode, isSupplementaryRunning, language,
    literatureFindings, onRequestSupplementary, onStopImmediate, onStopSummarize,
    openDropdownIndex, openSupplementaryInput, pathologyFindings, removeFile, roundNumber,
    roundSummary, sending, setCustomTypeError, setOpenDropdownIndex, setShowSupplementaryInput,
    showSupplementaryInput, t, timerDisplay, timerPaused, timerUrgent, toggleTimerPause,
    treatmentGroups, updateFileAgent, updateFileGroup, updateFileSample, updateFileType, userInput,
  }
}

export type RoundFeedbackController = ReturnType<typeof useRoundFeedbackController>
