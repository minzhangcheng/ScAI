import { useEffect, useState, useRef } from 'react'
import { useAppStore } from '../stores/appStore'
import { useSession, useWebSocket } from '../hooks/useApi'
import { useTranslation } from '../i18n/translations'
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
  Send,
  Upload,
  X,
  Image as ImageIcon,
  FileText,
  ChevronDown,
  ChevronUp
} from 'lucide-react'

const API_BASE = '/api'

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
}

export function AnalysisPanel() {
  const { currentSession, messages, language, wsConnected } = useAppStore()
  const { startAnalysis, resumeAnalysis } = useSession()
  const { connect, disconnect, connected } = useWebSocket()
  const { t } = useTranslation(language)
  
  // User input state
  const [userInput, setUserInput] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [sending, setSending] = useState(false)
  const [expandedRound, setExpandedRound] = useState<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const agents = [
    { id: 'pi', name: t('piAgent'), icon: Users, color: 'violet' },
    { id: 'pathology', name: t('pathologyAgent'), icon: Microscope, color: 'red' },
    { id: 'bioinformatics', name: t('bioinformaticsAgent'), icon: Code2, color: 'green' },
    { id: 'biochemistry', name: t('biochemistryAgent'), icon: FlaskConical, color: 'amber' },
    { id: 'literature', name: t('literatureAgent'), icon: BookOpen, color: 'blue' },
  ]

  // Get latest status for each agent
  const getAgentStatus = (agentId: string) => {
    const agentMessages = messages.filter(m => m.agent === agentId)
    if (agentMessages.length === 0) return 'idle'
    
    const lastMessage = agentMessages[agentMessages.length - 1]
    if (lastMessage.type === 'status') return 'working'
    if (lastMessage.type === 'result') return 'done'
    if (lastMessage.type === 'error') return 'error'
    return 'working'
  }

  // Extract round summaries from messages
  const getRoundSummaries = (): RoundSummary[] => {
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
          status: content.status || 'completed'
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
            summary: round.summary,
            findings: [
              ...round.pathologyFindings,
              ...round.bioinformaticsFindings,
              ...round.biochemistryFindings
            ],
            dataRequests: { pathology: [], biochemistry: [], other: [] },
            status: 'completed'
          })
        }
      }
    }
    
    return summaries.sort((a, b) => a.round - b.round)
  }

  const handleStart = async () => {
    try {
      await startAnalysis()
      setTimeout(() => connect(), 100)
    } catch (e) {
      console.error('Failed to start analysis:', e)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setUploadedFiles(prev => [...prev, ...files])
  }

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  // Continue analysis without additional input (just add one more round)
  const continueAnalysis = async () => {
    if (!currentSession) return

    setSending(true)
    try {
      // Send a simple continue message
      const response = await fetch(
        `${API_BASE}/session/${currentSession.id}/human-input`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: '[CONTINUE]',  // Special marker for continue without new instructions
            uploaded_files: [],
            target_agent: 'pi'
          })
        }
      )

      if (response.ok) {
        // Resume analysis
        await resumeAnalysis()
        if (!connected) {
          setTimeout(() => connect(), 100)
        }
      }
    } catch (e) {
      console.error('Failed to continue analysis:', e)
    } finally {
      setSending(false)
    }
  }

  // Send user input with optional files
  const sendUserInput = async () => {
    if (!currentSession || (!userInput.trim() && uploadedFiles.length === 0)) return

    setSending(true)
    try {
      // First upload any files
      const uploadedData: { type: string, filename: string, id?: string }[] = []
      
      for (const file of uploadedFiles) {
        const formData = new FormData()
        formData.append('file', file)
        
        // Determine file type and endpoint
        const isImage = file.type.startsWith('image/')
        
        if (isImage) {
          // Upload as pathology image (PI will classify it)
          formData.append('image_type', 'user_upload')
          const response = await fetch(
            `${API_BASE}/session/${currentSession.id}/upload/pathology`,
            { method: 'POST', body: formData }
          )
          if (response.ok) {
            const data = await response.json()
            uploadedData.push({ 
              type: 'image', 
              filename: file.name,
              id: data.data?.image_id 
            })
          }
        } else {
          // Upload as generic file
          const response = await fetch(
            `${API_BASE}/session/${currentSession.id}/upload/file`,
            { method: 'POST', body: formData }
          )
          if (response.ok) {
            uploadedData.push({ type: 'file', filename: file.name })
          }
        }
      }
      
      // Then send the message with file references
      const message = {
        message: userInput,
        uploaded_files: uploadedData,
        target_agent: 'pi'
      }
      
      const response = await fetch(
        `${API_BASE}/session/${currentSession.id}/human-input`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(message)
        }
      )
      
      if (response.ok) {
        setUserInput('')
        setUploadedFiles([])

        // Resume analysis if paused or completed (for re-analysis)
        if (currentSession.status === 'paused' || currentSession.status === 'waiting_input' || currentSession.status === 'completed') {
          await resumeAnalysis()
          if (!connected) {
            setTimeout(() => connect(), 100)
          }
        }
      }
    } catch (e) {
      console.error('Failed to send input:', e)
    } finally {
      setSending(false)
    }
  }

  // Clean up on unmount
  useEffect(() => {
    return () => disconnect()
  }, [disconnect])

  // Auto-connect WebSocket when session is running and not connected
  useEffect(() => {
    if (currentSession && 
        (currentSession.status === 'running' || currentSession.status === 'waiting_input') && 
        !connected) {
      console.log('Auto-connecting WebSocket for running session')
      connect()
    }
  }, [currentSession, connected, connect])

  // Auto-expand latest round
  useEffect(() => {
    const summaries = getRoundSummaries()
    if (summaries.length > 0) {
      setExpandedRound(summaries[summaries.length - 1].round)
    }
  }, [messages])

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

  const roundSummaries = getRoundSummaries()
  const isWaitingInput = currentSession.status === 'paused' || currentSession.status === 'waiting_input'
  const isRunning = currentSession.status === 'running'
  const isCompleted = currentSession.status === 'completed'

  return (
    <div className="flex flex-col h-full">
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
              className="flex items-center gap-2 px-5 py-2.5 bg-green-600 hover:bg-green-700 text-white rounded-xl font-medium transition-colors"
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
                      {round.summary.substring(0, 80)}...
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
      </div>

      {/* User Input Area - Fixed at Bottom */}
      {currentSession.status !== 'pending' && (
        <div className="border-t border-slate-200 pt-4 bg-white">
          <div className={`p-4 rounded-xl ${isCompleted ? 'bg-green-50' : 'bg-slate-50'}`}>
            <div className="flex items-start gap-3 mb-3">
              <Users className={`w-5 h-5 mt-0.5 ${isCompleted ? 'text-green-600' : 'text-violet-600'}`} />
              <div>
                <h4 className="font-medium text-slate-900">
                  {language === 'zh' ? '与 PI Agent 交流' : 'Communicate with PI Agent'}
                </h4>
                <p className="text-sm text-slate-500">
                  {isCompleted
                    ? (language === 'zh'
                        ? '分析已完成。如不满意，可以补充数据或要求重新分析'
                        : 'Analysis completed. You can add data or request re-analysis if needed')
                    : isWaitingInput
                    ? (language === 'zh'
                        ? '请上传补充数据或提供新的指导意见'
                        : 'Please upload additional data or provide guidance')
                    : (language === 'zh'
                        ? '您可以随时上传数据或调整分析方向'
                        : 'You can upload data or adjust analysis direction anytime')}
                </p>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="flex flex-wrap gap-2 mb-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 rounded-full transition-colors"
              >
                <Upload className="w-3.5 h-3.5" />
                {language === 'zh' ? '上传图片/文件' : 'Upload Files'}
              </button>
              {isCompleted ? (
                <>
                  <button
                    onClick={continueAnalysis}
                    disabled={sending}
                    className="px-3 py-1.5 text-xs bg-green-50 hover:bg-green-100 disabled:opacity-50 text-green-700 rounded-full transition-colors"
                  >
                    ▶️ {language === 'zh' ? '继续分析' : 'Continue'}
                  </button>
                  <button
                    onClick={() => setUserInput(language === 'zh' ? '我对分析结果不满意，请根据以下建议重新分析：' : 'I am not satisfied with the results. Please re-analyze with the following guidance:')}
                    className="px-3 py-1.5 text-xs bg-violet-50 hover:bg-violet-100 text-violet-700 rounded-full transition-colors"
                  >
                    🔄 {language === 'zh' ? '重新分析' : 'Re-analyze'}
                  </button>
                  <button
                    onClick={() => setUserInput(language === 'zh' ? '请补充以下数据后继续分析：' : 'Please continue analysis with additional data:')}
                    className="px-3 py-1.5 text-xs bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-full transition-colors"
                  >
                    ➕ {language === 'zh' ? '补充数据继续' : 'Add Data & Continue'}
                  </button>
                  <button
                    onClick={() => setUserInput(language === 'zh' ? '请针对以下方面进行更深入的分析：' : 'Please provide deeper analysis on the following aspects:')}
                    className="px-3 py-1.5 text-xs bg-amber-50 hover:bg-amber-100 text-amber-700 rounded-full transition-colors"
                  >
                    🔍 {language === 'zh' ? '深入分析' : 'Deeper Analysis'}
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => setUserInput(language === 'zh' ? '补充染色结果：' : 'Additional staining results:')}
                    className="px-3 py-1.5 text-xs bg-red-50 hover:bg-red-100 text-red-700 rounded-full transition-colors"
                  >
                    🔬 {language === 'zh' ? '补充染色' : 'Add Staining'}
                  </button>
                  <button
                    onClick={() => setUserInput(language === 'zh' ? '补充生化指标：' : 'Additional biochemistry data:')}
                    className="px-3 py-1.5 text-xs bg-amber-50 hover:bg-amber-100 text-amber-700 rounded-full transition-colors"
                  >
                    🧪 {language === 'zh' ? '补充生化' : 'Add Biochem'}
                  </button>
                  <button
                    onClick={() => setUserInput(language === 'zh' ? '请继续下一轮分析' : 'Please continue to next round')}
                    className="px-3 py-1.5 text-xs bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-full transition-colors"
                  >
                    ▶️ {language === 'zh' ? '继续分析' : 'Continue'}
                  </button>
                  <button
                    onClick={() => setUserInput(language === 'zh' ? '请结束分析并生成最终报告' : 'Please conclude and generate final report')}
                    className="px-3 py-1.5 text-xs bg-green-50 hover:bg-green-100 text-green-700 rounded-full transition-colors"
                  >
                    🏁 {language === 'zh' ? '结束分析' : 'Conclude'}
                  </button>
                </>
              )}
            </div>
            
            {/* Uploaded Files Preview */}
            {uploadedFiles.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-3">
                {uploadedFiles.map((file, idx) => (
                  <div key={idx} className="flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-sm">
                    {file.type.startsWith('image/') ? (
                      <ImageIcon className="w-4 h-4 text-blue-500" />
                    ) : (
                      <FileText className="w-4 h-4 text-slate-500" />
                    )}
                    <span className="max-w-[150px] truncate">{file.name}</span>
                    <button
                      onClick={() => removeFile(idx)}
                      className="text-slate-400 hover:text-red-500"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            
            {/* Input Area */}
            <div className="flex gap-2">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*,.txt,.csv,.json"
                className="hidden"
                onChange={handleFileSelect}
              />
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder={language === 'zh' 
                  ? '输入您的指令或描述补充数据...\n例如：\n• 已上传 Sirius Red 染色结果\n• ALT 值为 120 U/L，AST 值为 95 U/L\n• 请重点分析纤维化程度' 
                  : 'Enter your instructions or describe additional data...\ne.g.:\n• Uploaded Sirius Red staining results\n• ALT: 120 U/L, AST: 95 U/L\n• Please focus on fibrosis analysis'}
                className="flex-1 px-4 py-3 bg-white border border-slate-200 rounded-lg text-sm resize-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                rows={3}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && e.metaKey) {
                    sendUserInput()
                  }
                }}
              />
              <button
                onClick={sendUserInput}
                disabled={sending || (!userInput.trim() && uploadedFiles.length === 0)}
                className="px-4 py-2 bg-violet-600 hover:bg-violet-700 disabled:bg-violet-300 text-white rounded-lg font-medium transition-colors self-end"
              >
                {sending ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
            
            <p className="text-xs text-slate-400 mt-2">
              {language === 'zh' ? '按 ⌘+Enter 发送' : 'Press ⌘+Enter to send'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
