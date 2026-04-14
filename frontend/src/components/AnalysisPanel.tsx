import { useEffect, useState, useMemo, useCallback, useRef } from 'react'
import { useAppStore } from '../stores/appStore'
import { useSession, useConfig, generateId } from '../hooks/useApi'
import { useTranslation } from '../i18n/translations'
import ReactMarkdown from 'react-markdown'
import { RoundFeedbackDialog, UploadedFileWithAgent } from './RoundFeedbackDialog'
import { AnnotationConfirmationDialog } from './AnnotationConfirmationDialog'
import { ExplorationReportDialog } from './ExplorationReportDialog'
import { SnapshotPanel } from './SnapshotPanel'
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
  ChevronDown,
  ChevronUp,
  Palette,
  Eye,
  Wrench,
  Pause,
  Square,
  Camera
} from 'lucide-react'

const API_BASE = '/api'

// Content shape from backend round_summary messages (snake_case)
interface RoundSummaryContent {
  round?: number
  summary?: string
  findings?: string[]
  data_requests?: { pathology?: string[]; biochemistry?: string[]; other?: string[] }
  status?: string
  pathology_findings?: string[]
  bioinformatics_findings?: string[]
  biochemistry_findings?: string[]
  literature_findings?: string[]
}

export interface RoundSummary {
  round: number
  summary: string
  findings: string[]
  dataRequests: {
    pathology: string[]
    biochemistry: string[]
    other: string[]
  }
  status: 'analyzing' | 'waiting_input' | 'completed'
  pathologyFindings?: string[]
  bioinformaticsFindings?: string[]
  biochemistryFindings?: string[]
  literatureFindings?: string[]
}

export function AnalysisPanel() {
  const { currentSession, messages, language, showConfigPanel, wsConnected, wsConnect, agentErrors, readMode, addMessage } = useAppStore()
  const { startAnalysis, resumeAnalysis, continueRound, pauseAnalysis, stopAnalysis } = useSession()
  const { checkStatus, validateApiKey, saveConfig } = useConfig()
  const { t } = useTranslation(language)

  // Round results state
  const [expandedRound, setExpandedRound] = useState<number | null>(null)

  // Round feedback dialog state
  const [showFeedbackDialog, setShowFeedbackDialog] = useState(false)
  const [feedbackRoundData, setFeedbackRoundData] = useState<RoundSummary | null>(null)
  const lastDialogRoundRef = useRef<number>(-1)

  // Annotation confirmation dialog state
  const [showAnnotationDialog, setShowAnnotationDialog] = useState(false)
  const [annotationData, setAnnotationData] = useState<any>(null)
  const lastAnnotationMsgRef = useRef<string>('')

  // Exploration report dialog state
  const [showExplorationDialog, setShowExplorationDialog] = useState(false)
  const [explorationReportData, setExplorationReportData] = useState<any>(null)
  const lastExplorationMsgRef = useRef<string>('')

  // Supplementary analysis state
  const [isSupplementaryRunning, setIsSupplementaryRunning] = useState(false)

  // Snapshot panel state
  const [showSnapshotPanel, setShowSnapshotPanel] = useState(false)

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
    const summaries: RoundSummary[] = []

    // Look for round_summary messages from PI
    const piMessages = messages.filter(m => m.agent === 'pi')

    for (const msg of piMessages) {
      if (msg.type === 'round_summary' && typeof msg.content === 'object') {
        const content = msg.content as RoundSummaryContent
        summaries.push({
          round: content.round || summaries.length + 1,
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
        })
      }
    }

    // Also check roundResults from session
    if (currentSession?.roundResults) {
      for (const round of currentSession.roundResults) {
        const existing = summaries.find(s => s.round === round.round)
        if (!existing) {
          summaries.push({
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
          })
        }
      }
    }

    return summaries.sort((a, b) => a.round - b.round)
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
      await fetch(`${API_BASE}/session/${currentSession.id}/confirm-annotations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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

  // Direction choice handler (multi-direction)
  const handleChooseDirections = useCallback(async (directions: Array<{ direction: string; notes: string; isCustom: boolean }>) => {
    if (!currentSession) return
    try {
      await fetch(`${API_BASE}/session/${currentSession.id}/choose-direction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          directions: directions.map(d => ({
            direction: d.direction,
            notes: d.notes
          }))
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
      await fetch(`${API_BASE}/session/${currentSession.id}/choose-direction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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

  const handleSupplementaryAnalysis = useCallback(async (requestText: string, target: 'annotation' | 'direction') => {
    if (!currentSession) return
    setIsSupplementaryRunning(true)
    try {
      const response = await fetch(`${API_BASE}/session/${currentSession.id}/supplementary-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_text: requestText, target }),
      })
      if (!response.ok) {
        const err = await response.json()
        console.error('Supplementary analysis error:', err.detail || err)
      }
      // isSupplementaryRunning will be reset when new annotation_confirmation or exploration_report message arrives
    } catch (error) {
      console.error('Supplementary analysis error:', error)
      setIsSupplementaryRunning(false)
    }
  }, [currentSession])

  const handleStart = async () => {
    // Block in read mode
    if (readMode) {
      alert(t('readModeRestriction'))
      return
    }

    try {
      // Step 1: Check if API key exists at all
      const { apiConfig } = useAppStore.getState()

      // If frontend has no API key, show config panel
      if (!apiConfig.apiKey) {
        if (showConfigPanel) showConfigPanel()
        return
      }

      // Check backend config status; if not configured, auto-sync from frontend
      let status = await checkStatus()
      if (!status.configured) {
        try {
          await saveConfig()
          status = await checkStatus()
        } catch (e) {
          console.warn('[ScAI] Auto-sync config failed:', e)
        }
      }

      // Step 2: Validate the API key via actual API call
      setValidating(true)
      const validation = await validateApiKey()
      setValidating(false)

      if (validation.valid) {
        // Key is valid - proceed
        const failures = await startAnalysis()
        if (failures) notifySyncFailures(failures)
        setTimeout(() => wsConnect?.(), 100)
      } else {
        // Key is invalid or uncertain - show dialog
        setValidationMessage(validation.message)
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

  // Dialog handler: Send user input with files (uploaded to agent-specific endpoints)
  const handleDialogSend = useCallback(async (message: string, files: UploadedFileWithAgent[]) => {
    if (!currentSession) return

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

        const response = await fetch(endpoint, { method: 'POST', body: formData })
        if (response.ok) {
          const data = await response.json()
          uploadedData.push({
            type: file.type.startsWith('image/') ? 'image' : 'file',
            filename: file.name,
            id: data.data?.file_id || data.data?.image_id,
          })
        }
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err)
      }
    }

    // Send human-input with message + file references
    try {
      await fetch(`${API_BASE}/session/${currentSession.id}/human-input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
  }, [currentSession, resumeAnalysis, wsConnected, wsConnect])

  // Dialog handler: Skip/Continue (auto-proceed without input)
  const handleDialogSkip = useCallback(async () => {
    try {
      await continueRound()
      if (!wsConnected) {
        setTimeout(() => wsConnect?.(), 100)
      }
    } catch (e) {
      console.error('Failed to continue round:', e)
    }
    setShowFeedbackDialog(false)
  }, [continueRound, wsConnected, wsConnect])

  // Pause handler
  const handlePause = useCallback(async () => {
    try {
      await pauseAnalysis()
    } catch (e) {
      console.error('Failed to pause:', e)
    }
  }, [pauseAnalysis])

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

  // Auto-connect WebSocket when session is running and not connected
  // Uses wsConnect from the store (managed by App.tsx) to avoid duplicate WebSocket instances
  useEffect(() => {
    if (currentSession &&
        (currentSession.status === 'running' || currentSession.status === 'waiting_input' || currentSession.status === 'waiting_annotation') &&
        !wsConnected && wsConnect) {
      wsConnect()
    }
  }, [currentSession?.id, currentSession?.status, wsConnected, wsConnect])

  // Auto-expand latest round + show feedback dialog when round completes with waiting_input
  useEffect(() => {
    if (roundSummaries.length > 0) {
      const latest = roundSummaries[roundSummaries.length - 1]
      setExpandedRound(latest.round)

      // Show feedback dialog when a round completes and is waiting for input
      // Use ref guard to prevent duplicate triggers for the same round
      if (latest.status === 'waiting_input' && currentSession?.status === 'waiting_input'
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
      setAnnotationData(annMsg.content)
      setIsSupplementaryRunning(false)
      if (!showAnnotationDialog) {
        setShowAnnotationDialog(true)
      }
    }
  }, [messages, currentSession?.status])

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
      setExplorationReportData(expMsg.content)
      setIsSupplementaryRunning(false)
      if (!showExplorationDialog) {
        setShowExplorationDialog(true)
        setShowFeedbackDialog(false)
      }
    }
  }, [messages, currentSession?.status, currentSession?.currentRound])

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

  const isWaitingInput = currentSession.status === 'paused' || currentSession.status === 'waiting_input' || currentSession.status === 'waiting_annotation'
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-red-100 rounded-full">
                  <AlertCircle className="w-6 h-6 text-red-600" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900">
                  {validationUncertain ? t('apiKeyValidationUncertain') : t('apiKeyInvalid')}
                </h3>
              </div>
              <p className="text-sm text-slate-600 mb-2">
                {validationMessage}
              </p>
              <p className="text-xs text-slate-400">
                {validationUncertain
                  ? t('apiKeyValidationUncertainDesc')
                  : t('apiKeyInvalidDesc')}
              </p>
            </div>
            <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex justify-end gap-3">
              <button
                onClick={handleIgnoreValidationError}
                className="px-4 py-2 text-slate-600 hover:bg-slate-200 rounded-lg font-medium transition-colors text-sm"
              >
                {t('ignoreError')}
              </button>
              <button
                onClick={handleReconfigureFromValidation}
                className="px-4 py-2 bg-accent-600 hover:bg-accent-700 text-white rounded-lg font-medium transition-colors text-sm"
              >
                {t('reconfigureApiKey')}
              </button>
            </div>
          </div>
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
              disabled={readMode}
              className={`flex items-center gap-2 px-4 py-1.5 text-white rounded-lg font-medium transition-colors ${
                readMode
                  ? 'bg-slate-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
              title={readMode ? t('readModeRestriction') : undefined}
            >
              <Play className="w-4 h-4" />
              {t('startAnalysis')}
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
              <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>
            </div>
          )}

          {isWaitingInput && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>
              <div className={`flex items-center gap-2 px-3 py-1 rounded-lg ${
                currentSession.status === 'waiting_annotation'
                  ? 'bg-accent-100 text-accent-700'
                  : 'bg-amber-100 text-amber-700'
              }`}>
                <AlertCircle className="w-4 h-4" />
                <span className="font-medium">
                  {currentSession.status === 'waiting_annotation'
                    ? (language === 'zh' ? '等待注释确认' : 'Waiting for Annotation')
                    : (language === 'zh' ? '等待输入' : 'Waiting for Input')}
                </span>
              </div>
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
              <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>
              <div className="flex items-center gap-2 px-3 py-1 bg-green-100 text-green-700 rounded-lg">
                <CheckCircle2 className="w-4 h-4" />
                <span className="font-medium">{t('analysisComplete')}</span>
              </div>
            </div>
          )}

          {isFailed && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>
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
        {/* Round Summaries */}
        <div>
          <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wide mb-3">
            {language === 'zh' ? '分析进度' : 'Analysis Progress'}
          </h3>
          
          {roundSummaries.length === 0 && isRunning && (
            <div className="p-6 bg-blue-50 border border-blue-200 rounded-xl text-center">
              <Loader2 className="w-8 h-8 text-blue-500 mx-auto mb-3 animate-spin" />
              <p className="font-medium text-blue-900">
                {language === 'zh' ? 'Round 1 分析进行中...' : 'Round 1 analysis in progress...'}
              </p>
              <p className="text-sm text-blue-600 mt-1">
                {language === 'zh' 
                  ? 'PI Agent 正在协调各专家进行分析，请稍候' 
                  : 'PI Agent is coordinating analysis, please wait'}
              </p>
            </div>
          )}
          
          {roundSummaries.length === 0 && !isRunning && currentSession.status === 'pending' && (
            <div className="p-6 bg-slate-50 border border-slate-200 rounded-xl text-center">
              <p className="text-slate-600">
                {language === 'zh' 
                  ? '点击"开始分析"启动多智能体分析流程' 
                  : 'Click "Start Analysis" to begin the multi-agent analysis workflow'}
              </p>
            </div>
          )}
          
          <div className="space-y-3">
            {roundSummaries.map((round) => (
              <div
                key={round.round}
                className={`border rounded-xl overflow-hidden transition-all ${
                  round.status === 'waiting_input'
                    ? 'border-amber-300 bg-amber-50'
                    : 'border-slate-200 bg-white'
                }`}
              >
                {/* Round Header */}
                <button
                  onClick={() => setExpandedRound(expandedRound === round.round ? null : round.round)}
                  className="w-full flex items-center justify-between p-4 text-left hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className={`px-2.5 py-1 rounded-full text-sm font-medium ${
                      round.status === 'waiting_input'
                        ? 'bg-amber-200 text-amber-800'
                        : round.status === 'analyzing'
                        ? 'bg-blue-200 text-blue-800'
                        : 'bg-green-200 text-green-800'
                    }`}>
                      Round {round.round}
                    </span>
                    <span className="text-sm text-slate-600 line-clamp-1">
                      {(round.summary || '').substring(0, 80)}...
                    </span>
                  </div>
                  {expandedRound === round.round ? (
                    <ChevronUp className="w-5 h-5 text-slate-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-slate-400" />
                  )}
                </button>

                {/* Round Details */}
                {expandedRound === round.round && (
                  <div className="px-4 pb-4 border-t border-slate-200">
                    {/* Summary */}
                    <div className="mt-3">
                      <h4 className="text-sm font-medium text-slate-700 mb-2">
                        {language === 'zh' ? '本轮总结' : 'Round Summary'}
                      </h4>
                      <p className="text-sm text-slate-600 bg-slate-50 p-3 rounded-lg">
                        {round.summary || ''}
                      </p>
                    </div>

                    {/* Findings */}
                    {round.findings.length > 0 && (
                      <div className="mt-3">
                        <h4 className="text-sm font-medium text-slate-700 mb-2">
                          {language === 'zh' ? '主要发现' : 'Key Findings'}
                        </h4>
                        <ul className="text-sm text-slate-600 space-y-1">
                          {round.findings.slice(0, 5).map((finding, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-green-500 mt-0.5">•</span>
                              <span>{finding}</span>
                            </li>
                          ))}
                          {round.findings.length > 5 && (
                            <li className="text-slate-400 italic">
                              +{round.findings.length - 5} {language === 'zh' ? '更多发现...' : 'more findings...'}
                            </li>
                          )}
                        </ul>
                      </div>
                    )}

                    {/* Data Requests */}
                    {(round.dataRequests.pathology.length > 0 ||
                      round.dataRequests.biochemistry.length > 0 ||
                      round.dataRequests.other.length > 0) && (
                      <div className="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                        <h4 className="text-sm font-medium text-amber-800 mb-2 flex items-center gap-2">
                          <AlertCircle className="w-4 h-4" />
                          {language === 'zh' ? '需要补充的数据' : 'Additional Data Requested'}
                        </h4>

                        {round.dataRequests.pathology.length > 0 && (
                          <div className="mb-2">
                            <span className="text-xs font-medium text-red-700 flex items-center gap-1">
                              <Microscope className="w-3 h-3" />
                              {language === 'zh' ? '病理染色:' : 'Pathology:'}
                            </span>
                            <ul className="text-sm text-slate-700 ml-4">
                              {round.dataRequests.pathology.map((item, idx) => (
                                <li key={idx}>• {item}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {round.dataRequests.biochemistry.length > 0 && (
                          <div className="mb-2">
                            <span className="text-xs font-medium text-amber-700 flex items-center gap-1">
                              <FlaskConical className="w-3 h-3" />
                              {language === 'zh' ? '生化指标:' : 'Biochemistry:'}
                            </span>
                            <ul className="text-sm text-slate-700 ml-4">
                              {round.dataRequests.biochemistry.map((item, idx) => (
                                <li key={idx}>• {item}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {round.dataRequests.other.length > 0 && (
                          <div>
                            <span className="text-xs font-medium text-slate-700">
                              {language === 'zh' ? '其他:' : 'Other:'}
                            </span>
                            <ul className="text-sm text-slate-700 ml-4">
                              {round.dataRequests.other.map((item, idx) => (
                                <li key={idx}>• {item}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

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
          language={language}
          countdownSeconds={useAppStore.getState().roundWaitSeconds}
          onConfirm={handleConfirmAnnotations}
          onRequestSupplementary={(text) => handleSupplementaryAnalysis(text, 'annotation')}
          isSupplementaryRunning={isSupplementaryRunning}
          onClose={() => setShowAnnotationDialog(false)}
        />
      )}

      {/* Exploration Report Dialog — shown after Round 1 exploration completes */}
      {showExplorationDialog && explorationReportData && currentSession && (
        <ExplorationReportDialog
          report={explorationReportData.report_a || explorationReportData}
          roundSummary={explorationReportData.round_summary?.summary || ''}
          explorationReportUrl={explorationReportData.html_url || undefined}
          detailedReportUrl={explorationReportData.detailed_report_url || undefined}
          language={language}
          countdownSeconds={useAppStore.getState().roundWaitSeconds}
          onChooseDirections={handleChooseDirections}
          onLetPiDecide={handleLetPiDecide}
          onRequestSupplementary={(text) => handleSupplementaryAnalysis(text, 'direction')}
          isSupplementaryRunning={isSupplementaryRunning}
          onClose={() => setShowExplorationDialog(false)}
        />
      )}

      {/* Round Feedback Dialog — shown after each round completes (not during exploration report) */}
      {showFeedbackDialog && !showExplorationDialog && feedbackRoundData && currentSession && (
        <RoundFeedbackDialog
          roundNumber={feedbackRoundData.round}
          maxRounds={useAppStore.getState().maxRounds}
          roundSummary={feedbackRoundData}
          language={language}
          countdownSeconds={useAppStore.getState().roundWaitSeconds}
          treatmentGroups={currentSession.treatmentGroups || []}
          onSend={handleDialogSend}
          onSkip={handleDialogSkip}
          onStopImmediate={() => handleStop('immediate')}
          onStopSummarize={() => handleStop('summarize')}
        />
      )}

      {/* Stop Confirmation Dialog */}
      {showStopDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4">
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-red-100 rounded-full">
                  <Square className="w-5 h-5 text-red-600" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900">
                  {language === 'zh' ? '停止分析' : 'Stop Analysis'}
                </h3>
              </div>
              <p className="text-sm text-slate-600 mb-4">
                {language === 'zh' ? '请选择停止方式：' : 'Choose how to stop:'}
              </p>
            </div>
            <div className="px-6 pb-6 space-y-2">
              <button
                onClick={() => handleStop('immediate')}
                disabled={stopping}
                className="w-full flex items-center gap-3 px-4 py-3 bg-red-50 hover:bg-red-100 border border-red-200 rounded-xl text-left transition-colors disabled:opacity-50"
              >
                <Square className="w-5 h-5 text-red-600 flex-shrink-0" />
                <div>
                  <div className="font-medium text-red-800">
                    {language === 'zh' ? '立即终止' : 'Terminate Immediately'}
                  </div>
                  <div className="text-xs text-red-600">
                    {language === 'zh'
                      ? '完全终止分析，保存状态，稍后可恢复继续'
                      : 'Stop analysis, save state, resume later'}
                  </div>
                </div>
              </button>
              <button
                onClick={() => handleStop('summarize')}
                disabled={stopping}
                className="w-full flex items-center gap-3 px-4 py-3 bg-amber-50 hover:bg-amber-100 border border-amber-200 rounded-xl text-left transition-colors disabled:opacity-50"
              >
                <CheckCircle2 className="w-5 h-5 text-amber-600 flex-shrink-0" />
                <div>
                  <div className="font-medium text-amber-800">
                    {language === 'zh' ? '终止分析并总结结果' : 'Terminate and Summarize'}
                  </div>
                  <div className="text-xs text-amber-600">
                    {language === 'zh'
                      ? '停止分析轮次，根据已有结果生成报告'
                      : 'Stop rounds, generate report from current results'}
                  </div>
                </div>
              </button>
              <button
                onClick={() => setShowStopDialog(false)}
                disabled={stopping}
                className="w-full px-4 py-3 text-slate-600 hover:bg-slate-100 rounded-xl font-medium transition-colors text-center disabled:opacity-50"
              >
                {language === 'zh' ? '按错了，继续' : 'Cancel, Continue'}
              </button>
              {stopping && (
                <div className="flex items-center gap-2 text-sm text-blue-600 pt-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {language === 'zh' ? '正在停止...' : 'Stopping...'}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Snapshot Panel */}
      {showSnapshotPanel && currentSession && (
        <SnapshotPanel
          sessionId={currentSession.id}
          apiBase={API_BASE}
          language={language}
          onClose={() => setShowSnapshotPanel(false)}
          onRestore={() => {
            setShowSnapshotPanel(false)
            // Reload page to get restored state
            window.location.reload()
          }}
        />
      )}
    </div>
  )
}
