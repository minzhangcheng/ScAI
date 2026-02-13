import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Pause,
  Play,
  SkipForward,
  Send,
  Upload,
  X,
  Loader2,
  Microscope,
  Code2,
  FlaskConical,
  BookOpen,
  AlertTriangle,
  Clock,
  Users,
  ChevronDown
} from 'lucide-react'
import { useTranslation, Language } from '../i18n/translations'

interface RoundSummary {
  round: number
  summary: string
  findings: string[]
  dataRequests: {
    pathology: string[]
    biochemistry: string[]
    other: string[]
  }
  status: 'analyzing' | 'waiting_input' | 'completed'
  // Per-agent findings (from round_summary WebSocket message)
  pathologyFindings?: string[]
  bioinformaticsFindings?: string[]
  biochemistryFindings?: string[]
  literatureFindings?: string[]
}

export interface UploadedFileWithAgent {
  file: File
  targetAgent: string
}

interface RoundFeedbackDialogProps {
  roundNumber: number
  maxRounds: number
  roundSummary: RoundSummary
  language: Language
  countdownSeconds: number  // -1=manual skip only, 0=no wait (auto-skip), positive=countdown
  onSend: (message: string, files: UploadedFileWithAgent[]) => Promise<void>
  onSkip: () => Promise<void>
}

const AGENT_OPTIONS = [
  { value: 'pi', label: 'PI Agent', labelZh: 'PI Agent' },
  { value: 'pathology', label: 'Pathology', labelZh: '病理' },
  { value: 'bioinformatics', label: 'Bioinformatics', labelZh: '生信' },
  { value: 'biochemistry', label: 'Biochemistry', labelZh: '生化' },
]

export function RoundFeedbackDialog({
  roundNumber,
  maxRounds,
  roundSummary,
  language,
  countdownSeconds,
  onSend,
  onSkip,
}: RoundFeedbackDialogProps) {
  const { t } = useTranslation(language)

  // Derived mode flags
  const isManualMode = countdownSeconds === -1
  const isNoWait = countdownSeconds === 0
  const hasCountdown = countdownSeconds > 0

  // Input state
  const [userInput, setUserInput] = useState('')
  const [files, setFiles] = useState<UploadedFileWithAgent[]>([])
  const [sending, setSending] = useState(false)

  // Timer state — uses wall-clock time so background tabs don't freeze the countdown
  const [timeRemaining, setTimeRemaining] = useState(hasCountdown ? countdownSeconds : 0)
  const [timerPaused, setTimerPaused] = useState(false)
  const hasTypedRef = useRef(false)
  const deadlineRef = useRef(hasCountdown ? Date.now() + countdownSeconds * 1000 : 0)
  const pausedAtRef = useRef<number | null>(null) // timestamp when paused, null if running
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Agent dropdown state (for file targeting)
  const [openDropdownIndex, setOpenDropdownIndex] = useState<number | null>(null)

  const isLastRound = roundNumber >= maxRounds

  // No-wait mode: auto-skip immediately on mount
  useEffect(() => {
    if (isNoWait && !sending) {
      onSkip()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Handle pause/resume: shift deadline when pausing/resuming
  useEffect(() => {
    if (!hasCountdown) return
    if (timerPaused) {
      // Record when we paused
      pausedAtRef.current = Date.now()
    } else if (pausedAtRef.current !== null) {
      // Resuming: shift deadline forward by the time spent paused
      const pausedDuration = Date.now() - pausedAtRef.current
      deadlineRef.current += pausedDuration
      pausedAtRef.current = null
    }
  }, [timerPaused, hasCountdown])

  // Countdown timer — uses wall-clock diff so background tabs catch up correctly
  useEffect(() => {
    if (!hasCountdown) return
    intervalRef.current = setInterval(() => {
      if (pausedAtRef.current !== null) return // paused, don't update
      const remaining = Math.max(0, Math.ceil((deadlineRef.current - Date.now()) / 1000))
      setTimeRemaining(remaining)
      if (remaining <= 0) {
        if (intervalRef.current) clearInterval(intervalRef.current)
      }
    }, 500) // 500ms for snappier updates when returning from background

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [hasCountdown])

  // Auto-proceed when timer reaches 0 (only in countdown mode)
  useEffect(() => {
    if (hasCountdown && timeRemaining === 0 && !sending) {
      onSkip()
    }
  }, [timeRemaining, sending, onSkip, hasCountdown])

  // Auto-pause when user starts typing
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
    if (intervalRef.current) clearInterval(intervalRef.current)
    await onSkip()
  }, [onSkip])

  const handleSend = useCallback(async () => {
    if (sending) return
    if (!userInput.trim() && files.length === 0) return

    setSending(true)
    // Pause timer during send
    if (intervalRef.current) clearInterval(intervalRef.current)

    try {
      await onSend(userInput.trim(), files)
    } catch (error) {
      console.error('Failed to send input:', error)
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

    // Auto-pause timer when files are added
    if (!hasTypedRef.current) {
      hasTypedRef.current = true
      setTimerPaused(true)
    }

    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  const removeFile = useCallback((index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }, [])

  const updateFileAgent = useCallback((index: number, agent: string) => {
    setFiles(prev => prev.map((f, i) => i === index ? { ...f, targetAgent: agent } : f))
    setOpenDropdownIndex(null)
  }, [])

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = () => setOpenDropdownIndex(null)
    if (openDropdownIndex !== null) {
      document.addEventListener('click', handleClickOutside)
      return () => document.removeEventListener('click', handleClickOutside)
    }
  }, [openDropdownIndex])

  // Format timer display
  const minutes = Math.floor(timeRemaining / 60)
  const seconds = timeRemaining % 60
  const timerDisplay = `${minutes}:${seconds.toString().padStart(2, '0')}`
  const timerUrgent = timeRemaining <= 30

  // Gather per-agent findings
  const pathologyFindings = roundSummary.pathologyFindings || []
  const bioinformaticsFindings = roundSummary.bioinformaticsFindings || []
  const biochemistryFindings = roundSummary.biochemistryFindings || []
  const literatureFindings = roundSummary.literatureFindings || []
  const hasDataRequests = roundSummary.dataRequests && (
    roundSummary.dataRequests.pathology?.length > 0 ||
    roundSummary.dataRequests.biochemistry?.length > 0 ||
    roundSummary.dataRequests.other?.length > 0
  )

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl mx-4 max-h-[85vh] flex flex-col overflow-hidden">

        {/* Header with timer */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-blue-50 to-indigo-50">
          <h2 className="text-lg font-semibold text-slate-800">
            {t('roundComplete').replace('{round}', String(roundNumber))}
          </h2>
          <div className="flex items-center gap-3">
            {/* Timer — only shown in countdown mode */}
            {hasCountdown && (
              <>
                <div className="flex items-center gap-2">
                  <Clock className={`w-4 h-4 ${timerUrgent ? 'text-red-500' : 'text-slate-500'}`} />
                  <span className={`font-mono text-lg tabular-nums ${
                    timerUrgent ? 'text-red-600 animate-pulse' :
                    timerPaused ? 'text-amber-600' : 'text-slate-700'
                  }`}>
                    {timerDisplay}
                  </span>
                  {timerPaused && (
                    <span className="text-xs text-amber-600 font-medium">{t('timerPaused')}</span>
                  )}
                </div>
                {/* Pause/Resume timer */}
                <button
                  onClick={toggleTimerPause}
                  className="p-1.5 rounded-lg hover:bg-white/60 text-slate-600 hover:text-slate-800 transition-colors"
                  title={timerPaused ? t('resumeTimer') : t('pauseTimer')}
                >
                  {timerPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                </button>
              </>
            )}
            {/* Manual mode indicator */}
            {isManualMode && (
              <span className="text-xs text-slate-500 font-medium">
                {language === 'zh' ? '手动跳过模式' : 'Manual skip mode'}
              </span>
            )}
            {/* Skip button */}
            <button
              onClick={handleSkip}
              disabled={sending}
              className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors disabled:opacity-50"
            >
              <SkipForward className="w-3.5 h-3.5" />
              {t('skipRound')}
            </button>
          </div>
        </div>

        {/* Content area — top 2/3 scrollable */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4" style={{ minHeight: 0 }}>

          {/* PI Summary */}
          {roundSummary.summary && (
            <div>
              <h3 className="text-sm font-semibold text-slate-600 mb-2">{t('roundSummaryTitle')}</h3>
              <div className="bg-slate-50 rounded-lg p-4 text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
                {roundSummary.summary}
              </div>
            </div>
          )}

          {/* Per-agent findings */}
          <div>
            <h3 className="text-sm font-semibold text-slate-600 mb-2">{t('keyFindings')}</h3>
            <div className="space-y-2">
              {pathologyFindings.length > 0 && (
                <FindingsSection
                  icon={<Microscope className="w-4 h-4 text-purple-600" />}
                  label={t('pathologyFindings')}
                  findings={pathologyFindings}
                />
              )}
              {bioinformaticsFindings.length > 0 && (
                <FindingsSection
                  icon={<Code2 className="w-4 h-4 text-blue-600" />}
                  label={t('bioinformaticsFindings')}
                  findings={bioinformaticsFindings}
                />
              )}
              {biochemistryFindings.length > 0 && (
                <FindingsSection
                  icon={<FlaskConical className="w-4 h-4 text-green-600" />}
                  label={t('biochemistryFindings')}
                  findings={biochemistryFindings}
                />
              )}
              {literatureFindings.length > 0 && (
                <FindingsSection
                  icon={<BookOpen className="w-4 h-4 text-orange-600" />}
                  label={t('literatureFindings')}
                  findings={literatureFindings}
                />
              )}
              {pathologyFindings.length === 0 && bioinformaticsFindings.length === 0 &&
               biochemistryFindings.length === 0 && literatureFindings.length === 0 &&
               roundSummary.findings?.length > 0 && (
                <FindingsSection
                  icon={<Users className="w-4 h-4 text-slate-600" />}
                  label={t('keyFindings')}
                  findings={roundSummary.findings}
                />
              )}
            </div>
          </div>

          {/* Data requests */}
          {hasDataRequests && (
            <div>
              <h3 className="text-sm font-semibold text-slate-600 mb-2">{t('additionalDataSuggested')}</h3>
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm space-y-1">
                {roundSummary.dataRequests.pathology?.map((req, i) => (
                  <div key={`p-${i}`} className="flex items-start gap-2">
                    <Microscope className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
                    <span className="text-amber-800">{req}</span>
                  </div>
                ))}
                {roundSummary.dataRequests.biochemistry?.map((req, i) => (
                  <div key={`b-${i}`} className="flex items-start gap-2">
                    <FlaskConical className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
                    <span className="text-amber-800">{req}</span>
                  </div>
                ))}
                {roundSummary.dataRequests.other?.map((req, i) => (
                  <div key={`o-${i}`} className="flex items-start gap-2">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
                    <span className="text-amber-800">{req}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Bottom input area — bottom 1/3 */}
        <div className="border-t border-slate-200 px-6 py-4 space-y-3 bg-slate-50">

          {/* Last round notice */}
          {isLastRound && (
            <div className="flex items-center gap-2 px-3 py-2 bg-amber-100 border border-amber-300 rounded-lg text-sm text-amber-800">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              {t('lastRoundNotice')}
            </div>
          )}

          {/* File list */}
          {files.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {files.map((f, index) => (
                <div key={index} className="flex items-center gap-1.5 bg-white border border-slate-200 rounded-lg px-2 py-1 text-xs">
                  <span className="text-slate-700 max-w-[120px] truncate" title={f.file.name}>
                    {f.file.name}
                  </span>
                  <span className="text-slate-400">→</span>
                  {/* Agent selector dropdown */}
                  <div className="relative">
                    <button
                      onClick={(e) => { e.stopPropagation(); setOpenDropdownIndex(openDropdownIndex === index ? null : index) }}
                      className="flex items-center gap-0.5 text-blue-600 hover:text-blue-800 font-medium"
                    >
                      {AGENT_OPTIONS.find(a => a.value === f.targetAgent)?.[language === 'zh' ? 'labelZh' : 'label'] || f.targetAgent}
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    {openDropdownIndex === index && (
                      <div className="absolute top-full left-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-lg z-10 py-1 min-w-[120px]">
                        {AGENT_OPTIONS.map(agent => (
                          <button
                            key={agent.value}
                            onClick={(e) => { e.stopPropagation(); updateFileAgent(index, agent.value) }}
                            className={`w-full text-left px-3 py-1.5 text-xs hover:bg-slate-50 ${
                              f.targetAgent === agent.value ? 'text-blue-600 font-medium' : 'text-slate-700'
                            }`}
                          >
                            {language === 'zh' ? agent.labelZh : agent.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="text-slate-400 hover:text-red-500 ml-1"
                    title={t('removeFile')}
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Input row */}
          <div className="flex items-end gap-2">
            <textarea
              value={userInput}
              onChange={handleInputChange}
              placeholder={t('roundInputPlaceholder')}
              className="flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
              rows={3}
              disabled={sending}
            />
            <div className="flex flex-col gap-2">
              {/* Upload button */}
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={sending}
                className="p-2 rounded-lg border border-slate-300 hover:bg-slate-100 text-slate-600 transition-colors disabled:opacity-50"
                title={t('uploadFilesDialog')}
              >
                <Upload className="w-4 h-4" />
              </button>
              {/* Send button */}
              <button
                onClick={handleSend}
                disabled={sending || (!userInput.trim() && files.length === 0)}
                className="p-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:bg-slate-300"
                title={t('sendInput')}
              >
                {sending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Sending status */}
          {sending && (
            <div className="flex items-center gap-2 text-sm text-blue-600">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              {t('sendingInput')}
            </div>
          )}

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.csv,.xlsx,.xls,.txt,.json,.tsv"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      </div>
    </div>
  )
}

/* Sub-component: Findings section for a single agent */
function FindingsSection({
  icon,
  label,
  findings,
}: {
  icon: React.ReactNode
  label: string
  findings: string[]
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1.5">
        {icon}
        <span className="text-sm font-medium text-slate-700">{label}</span>
      </div>
      <ul className="space-y-1 pl-6">
        {findings.map((finding, i) => (
          <li key={i} className="text-sm text-slate-600 list-disc">{finding}</li>
        ))}
      </ul>
    </div>
  )
}
