import { useEffect, useState, useMemo, useCallback } from 'react'
import { useAppStore } from '../stores/appStore'
import { useSession, useConfig } from '../hooks/useApi'
import { useTranslation } from '../i18n/translations'
import ReactMarkdown from 'react-markdown'
import { RoundFeedbackDialog, UploadedFileWithAgent } from './RoundFeedbackDialog'
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
  Palette
} from 'lucide-react'

const API_BASE = '/api'

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
  const { currentSession, messages, language, showConfigPanel, wsConnected, wsConnect, agentErrors, readMode } = useAppStore()
  const { startAnalysis, resumeAnalysis, continueRound } = useSession()
  const { checkStatus, validateApiKey } = useConfig()
  const { t } = useTranslation(language)

  // Round results state
  const [expandedRound, setExpandedRound] = useState<number | null>(null)

  // Round feedback dialog state
  const [showFeedbackDialog, setShowFeedbackDialog] = useState(false)
  const [feedbackRoundData, setFeedbackRoundData] = useState<RoundSummary | null>(null)

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

    if (lastMessage.type === 'result') {
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
        const content = msg.content as any
        summaries.push({
          round: content.round || summaries.length + 1,
          summary: content.summary || '',
          findings: content.findings || [],
          dataRequests: content.data_requests || { pathology: [], biochemistry: [], other: [] },
          status: content.status || 'completed',
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

  const handleStart = async () => {
    // Block in read mode
    if (readMode) {
      alert(t('readModeRestriction'))
      return
    }

    try {
      // Step 1: Check if API key exists at all
      const { apiConfig } = useAppStore.getState()
      const status = await checkStatus()
      if (!status.configured || !apiConfig.apiKey) {
        if (showConfigPanel) {
          showConfigPanel()
        }
        return
      }

      // Step 2: Validate the API key via actual API call
      setValidating(true)
      const validation = await validateApiKey()
      setValidating(false)

      if (validation.valid) {
        // Key is valid - proceed
        await startAnalysis()
        setTimeout(() => wsConnect?.(), 100)
      } else {
        // Key is invalid or uncertain - show dialog
        setValidationMessage(validation.message)
        setValidationUncertain(validation.uncertain || false)
        setShowValidationDialog(true)
      }
    } catch (e) {
      setValidating(false)
      console.error('Failed to start analysis:', e)
    }
  }

  const handleIgnoreValidationError = async () => {
    setShowValidationDialog(false)
    try {
      await startAnalysis()
      setTimeout(() => wsConnect?.(), 100)
    } catch (e) {
      console.error('Failed to start analysis:', e)
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

    for (const { file, targetAgent } of files) {
      try {
        const formData = new FormData()
        formData.append('file', file)

        let endpoint: string
        if (targetAgent === 'pathology') {
          formData.append('image_type', 'user_upload')
          endpoint = `${API_BASE}/session/${currentSession.id}/upload/pathology`
        } else if (targetAgent === 'biochemistry') {
          endpoint = `${API_BASE}/session/${currentSession.id}/upload/biochemistry`
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

  // Dialog handler: Skip (auto-proceed without input)
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

  // Auto-connect WebSocket when session is running and not connected
  // Uses wsConnect from the store (managed by App.tsx) to avoid duplicate WebSocket instances
  useEffect(() => {
    if (currentSession &&
        (currentSession.status === 'running' || currentSession.status === 'waiting_input') &&
        !wsConnected && wsConnect) {
      console.log('AnalysisPanel: Auto-connecting WebSocket for running session')
      wsConnect()
    }
  }, [currentSession?.id, currentSession?.status, wsConnected, wsConnect])

  // Auto-expand latest round + show feedback dialog when round completes with waiting_input
  useEffect(() => {
    if (roundSummaries.length > 0) {
      const latest = roundSummaries[roundSummaries.length - 1]
      setExpandedRound(latest.round)

      // Show feedback dialog when a round completes and is waiting for input
      if (latest.status === 'waiting_input' && currentSession?.status === 'waiting_input') {
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

  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 rounded-full mb-6">
          <Users className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="text-xl font-semibold text-slate-900 mb-2">
          {t('startNewAnalysis')}
        </h2>
        <p className="text-slate-500 max-w-md">
          {t('welcomeDesc')}
        </p>
      </div>
    )
  }

  const isWaitingInput = currentSession.status === 'paused' || currentSession.status === 'waiting_input'
  const isRunning = currentSession.status === 'running'
  const isCompleted = currentSession.status === 'completed'

  return (
    <div className="flex flex-col h-full">
      {/* Validating spinner overlay */}
      {validating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-lg p-6 flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-violet-600" />
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
                className="px-4 py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-lg font-medium transition-colors text-sm"
              >
                {t('reconfigureApiKey')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header - Compact */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-200">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">{t('analysisControl')}</h2>
          <p className="text-sm text-slate-500 truncate max-w-md">
            {currentSession.instruction}
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          {currentSession.status === 'pending' && (
            <button
              onClick={handleStart}
              disabled={readMode}
              className={`flex items-center gap-2 px-5 py-2.5 text-white rounded-xl font-medium transition-colors ${
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
            <div className="flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-xl">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="font-medium">Round {currentSession.currentRound}</span>
            </div>
          )}
          
          {isWaitingInput && (
            <div className="flex items-center gap-2 px-4 py-2 bg-amber-100 text-amber-700 rounded-xl">
              <AlertCircle className="w-4 h-4" />
              <span className="font-medium">
                {language === 'zh' ? '等待输入' : 'Waiting for Input'}
              </span>
            </div>
          )}
          
          {isCompleted && (
            <div className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-700 rounded-xl">
              <CheckCircle2 className="w-4 h-4" />
              <span className="font-medium">{t('analysisComplete')}</span>
            </div>
          )}
        </div>
      </div>

      {/* Agent Status - Compact Row */}
      <div className="py-3 border-b border-slate-200">
        <div className="flex items-center gap-2 overflow-x-auto">
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
              <Loader2 className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-3" />
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
                        {round.summary}
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
          <div className="bg-gradient-to-br from-violet-50 to-indigo-50 border border-violet-200 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-violet-100 rounded-lg">
                <CheckCircle2 className="w-5 h-5 text-violet-600" />
              </div>
              <h3 className="font-semibold text-violet-900">{t('finalReport')}</h3>
            </div>
            <div className="prose prose-slate prose-sm max-w-none">
              <ReactMarkdown>{currentSession.finalSummary}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* Round Feedback Dialog — shown after each round completes */}
      {showFeedbackDialog && feedbackRoundData && currentSession && (
        <RoundFeedbackDialog
          roundNumber={feedbackRoundData.round}
          maxRounds={useAppStore.getState().maxRounds}
          roundSummary={feedbackRoundData}
          language={language}
          countdownSeconds={useAppStore.getState().roundWaitSeconds}
          onSend={handleDialogSend}
          onSkip={handleDialogSkip}
        />
      )}
    </div>
  )
}
