import { useState, useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import { apiFetch } from '../hooks/useApi'
import ReactMarkdown from 'react-markdown'
import { Send, Loader2, User, Bot, AlertCircle, Paperclip, X, FileText, Image as ImageIcon, Download, Square } from 'lucide-react'

const AGENT_OPTIONS = [
  { value: 'pi', labelKey: 'chatTargetPi' as const },
  { value: 'pathology', labelKey: 'chatTargetPathology' as const },
  { value: 'bioinformatics', labelKey: 'chatTargetBioinformatics' as const },
  { value: 'biochemistry', labelKey: 'chatTargetBiochemistry' as const },
]

// Section-specific data type options
const PATHOLOGY_TYPES = ['H&E', 'Other']
const SEQUENCING_TYPES = ['scRNA-seq', 'Loom']
const BIOCHEMISTRY_TYPES = ['image', 'file']

function getSection(agent: string): string | null {
  if (agent === 'pathology') return 'pathology'
  if (agent === 'bioinformatics') return 'sequencing'
  if (agent === 'biochemistry') return 'biochemistry'
  return null
}

function getTypeOptions(agent: string): string[] {
  if (agent === 'pathology') return PATHOLOGY_TYPES
  if (agent === 'bioinformatics') return SEQUENCING_TYPES
  if (agent === 'biochemistry') return BIOCHEMISTRY_TYPES
  return []
}

interface SelectedFileWithAgent {
  file: File
  targetAgent: string
  groupId?: string
  sampleNumber?: number
  dataType?: string
}

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'])

function isImageFile(filename: string): boolean {
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'))
  return IMAGE_EXTS.has(ext)
}

export function ChatPanel() {
  const {
    language, currentSession,
    chatMessages, addChatMessage, setChatMessages,
    chatLoading, setChatLoading,
    chatDelegationActive, chatDelegationPlan,
    setChatDelegationActive, setChatDelegationPlan,
    setChatAbortController,
  } = useAppStore()
  const { t } = useTranslation(language)
  const [input, setInput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [selectedFiles, setSelectedFiles] = useState<SelectedFileWithAgent[]>([])
  const [defaultAgent, setDefaultAgent] = useState('pi')
  const [filePreviews, setFilePreviews] = useState<Record<string, string>>({})
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const loadedSessionRef = useRef<string | null>(null)
  const [textareaHeight, setTextareaHeight] = useState(72)
  const resizingRef = useRef(false)
  const resizeStartRef = useRef({ y: 0, height: 0 })

  // Load chat history when session changes
  useEffect(() => {
    if (!currentSession?.id) {
      setChatMessages([])
      loadedSessionRef.current = null
      return
    }
    if (loadedSessionRef.current === currentSession.id) return
    loadedSessionRef.current = currentSession.id

    apiFetch<{ messages: typeof chatMessages }>(`/session/${currentSession.id}/chat`)
      .then((data) => {
        setChatMessages(data.messages || [])
      })
      .catch(() => {
        setChatMessages([])
      })
  }, [currentSession?.id, setChatMessages])

  // Poll backend during loading as fallback (covers both delegation and slow figure generation)
  useEffect(() => {
    if (!chatLoading || !currentSession?.id) return
    const interval = setInterval(async () => {
      try {
        const data = await apiFetch<{ messages: typeof chatMessages }>(
          `/session/${currentSession.id}/chat`
        )
        const msgs = data.messages || []
        // Check if backend has more messages than frontend (new response arrived)
        // Read latest messages from store to avoid stale closure
        const currentChatMessages = useAppStore.getState().chatMessages
        if (msgs.length > currentChatMessages.length) {
          const lastMsg = msgs[msgs.length - 1] as any
          if (lastMsg && !lastMsg?.delegation_active && lastMsg.role === 'assistant') {
            // Merge: preserve any local-only (optimistic) messages not yet on the server
            const serverMsgIds = new Set(msgs.map((m: any) => m.id || `${m.timestamp}|${m.role}`))
            const localOnly = currentChatMessages.filter(
              (m) => !serverMsgIds.has(m.id || `${m.timestamp}|${m.role}`)
            )
            setChatMessages([...msgs, ...localOnly])
            setChatLoading(false)
            setChatDelegationActive(false)
            setChatDelegationPlan(null)
          }
        }
      } catch {}
    }, 5000)
    return () => clearInterval(interval)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatLoading, currentSession?.id, setChatMessages, setChatLoading, setChatDelegationActive, setChatDelegationPlan])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  // Generate image previews for selected files
  useEffect(() => {
    const previews: Record<string, string> = {}
    const urls: string[] = []
    selectedFiles.forEach(({ file }) => {
      if (file.type.startsWith('image/')) {
        const url = URL.createObjectURL(file)
        previews[file.name + file.size] = url
        urls.push(url)
      }
    })
    setFilePreviews(previews)
    return () => urls.forEach(URL.revokeObjectURL)
  }, [selectedFiles])

  // Custom top-edge resize: drag upward to grow
  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    resizingRef.current = true
    resizeStartRef.current = { y: e.clientY, height: textareaHeight }
    document.body.style.cursor = 'ns-resize'
    document.body.style.userSelect = 'none'
  }, [textareaHeight])

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!resizingRef.current) return
      const delta = resizeStartRef.current.y - e.clientY // upward = positive
      setTextareaHeight(Math.max(72, resizeStartRef.current.height + delta))
    }
    const onMouseUp = () => {
      if (resizingRef.current) {
        resizingRef.current = false
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }
    }
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
    return () => {
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
    }
  }, [])

  const handleAddFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newFiles = Array.from(e.target.files || [])
    if (newFiles.length > 0) {
      setSelectedFiles((prev) => [
        ...prev,
        ...newFiles.map((file) => ({ file, targetAgent: defaultAgent })),
      ])
    }
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleFileAgentChange = (index: number, agent: string) => {
    const types = getTypeOptions(agent)
    setSelectedFiles((prev) =>
      prev.map((item, i) => (i === index
        ? { ...item, targetAgent: agent, groupId: undefined, sampleNumber: undefined, dataType: types[0] || undefined }
        : item))
    )
  }

  const handleFileGroupChange = (index: number, groupId: string | undefined) => {
    setSelectedFiles((prev) =>
      prev.map((item, i) => (i === index ? { ...item, groupId, sampleNumber: undefined } : item))
    )
  }

  const handleFileSampleChange = (index: number, sampleNumber: number | undefined) => {
    setSelectedFiles((prev) =>
      prev.map((item, i) => (i === index ? { ...item, sampleNumber } : item))
    )
  }

  const handleFileTypeChange = (index: number, dataType: string) => {
    setSelectedFiles((prev) =>
      prev.map((item, i) => (i === index ? { ...item, dataType } : item))
    )
  }

  const handleSend = async () => {
    if ((!input.trim() && selectedFiles.length === 0) || !currentSession?.id || chatLoading) return

    // Capture session ID at send time to detect cross-session pollution
    const sendSessionId = currentSession.id

    const message = input.trim() || (selectedFiles.length > 0 ? `[${selectedFiles.map(sf => sf.file.name).join(', ')}]` : '')
    setInput('')
    setError(null)
    setTextareaHeight(72)

    const filesToSend = [...selectedFiles]
    setSelectedFiles([])

    // Create AbortController for this request
    const abortController = new AbortController()
    setChatAbortController(abortController)

    // Optimistic: add user message immediately
    const userMsg = {
      role: 'user' as const,
      content: input.trim(),
      timestamp: new Date().toISOString(),
      files: filesToSend.map((sf) => ({
        filename: sf.file.name,
        type: sf.file.type.startsWith('image/') ? 'image' : 'file',
        target_agent: sf.targetAgent,
      })),
    }
    addChatMessage(userMsg)
    setChatLoading(true)

    try {
      const formData = new FormData()
      formData.append('message', message)
      // Send per-file metadata as JSON arrays
      formData.append('target_agents', JSON.stringify(filesToSend.map((sf) => sf.targetAgent)))
      formData.append('file_groups', JSON.stringify(filesToSend.map((sf) => sf.groupId || null)))
      formData.append('file_samples', JSON.stringify(filesToSend.map((sf) => sf.sampleNumber || null)))
      formData.append('file_types', JSON.stringify(filesToSend.map((sf) => sf.dataType || null)))
      filesToSend.forEach((sf) => formData.append('files', sf.file))

      // Use raw fetch (not apiFetch) because FormData needs no Content-Type header
      const response = await fetch(`/api/session/${currentSession.id}/chat`, {
        method: 'POST',
        body: formData,
        signal: abortController.signal,
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(err.detail || 'Chat request failed')
      }

      const data = await response.json()

      // Guard: discard response if user switched session during the request
      if (useAppStore.getState().currentSession?.id !== sendSessionId) {
        console.warn('ChatPanel: Session changed during request, discarding response')
        return
      }

      if (data.status === 'processing') {
        // Delegation mode: response will arrive via WebSocket (chat_result)
        // Ensure WebSocket is connected to receive progress messages
        const { wsConnect: connectWs, wsConnected: isWsConnected } = useAppStore.getState()
        if (!isWsConnected && connectWs) {
          connectWs()
        }
        // Mark delegation active BEFORE return so finally block won't reset chatLoading
        setChatDelegationActive(true)
        return
      }

      addChatMessage({
        role: 'assistant',
        content: data.content,
        timestamp: new Date().toISOString(),
        figures: data.figures,
      })
    } catch (e: any) {
      if (e.name === 'AbortError') {
        addChatMessage({
          role: 'assistant',
          content: t('chatCancelled'),
          timestamp: new Date().toISOString(),
        })
      } else {
        setError(t('chatError'))
      }
    } finally {
      setChatAbortController(null)
      if (!useAppStore.getState().chatDelegationActive) {
        setChatLoading(false)
      }
    }
  }

  const handleStop = async () => {
    if (!currentSession?.id) return

    const { chatAbortController: controller, chatDelegationActive: isDelegation } = useAppStore.getState()

    if (isDelegation) {
      // Cancel delegation via backend endpoint
      try {
        await fetch(`/api/session/${currentSession.id}/chat/cancel`, { method: 'POST' })
      } catch (e) {
        console.error('Failed to cancel chat delegation:', e)
      }
    } else if (controller) {
      // Cancel simple fetch request
      controller.abort()
    }

    // Immediate UI feedback
    setChatAbortController(null)
    setChatLoading(false)
    setChatDelegationActive(false)
    setChatDelegationPlan(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // No session state
  if (!currentSession) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center text-slate-400">
          <Bot className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p className="text-sm">{t('chatNoSession')}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {chatMessages.length === 0 ? (
          <div className="flex-1 flex items-center justify-center h-full min-h-[300px]">
            <div className="text-center text-slate-400">
              <Bot className="w-16 h-16 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium mb-1">{t('chatEmpty')}</p>
              <p className="text-sm">{t('chatEmptyDesc')}</p>
            </div>
          </div>
        ) : (
          chatMessages.map((msg, i) => (
            <div
              key={i}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-violet-600" />
                </div>
              )}
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-violet-600 text-white'
                    : 'bg-slate-100 text-slate-800'
                }`}
              >
                {/* File attachments */}
                {msg.files && msg.files.length > 0 && (
                  <div className="mb-2 space-y-1">
                    {msg.files.map((f: any, j: number) => (
                      <div
                        key={j}
                        className={`flex items-center gap-1.5 text-xs rounded-lg px-2 py-1 ${
                          msg.role === 'user' ? 'bg-violet-500/30' : 'bg-slate-200'
                        }`}
                      >
                        {isImageFile(f.filename) ? (
                          <ImageIcon className="w-3 h-3 shrink-0" />
                        ) : (
                          <FileText className="w-3 h-3 shrink-0" />
                        )}
                        <span className="truncate">{f.filename}</span>
                        {f.target_agent && f.target_agent !== 'pi' && (
                          <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            msg.role === 'user' ? 'bg-violet-400/40' : 'bg-violet-100 text-violet-700'
                          }`}>
                            {f.target_agent}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {/* Message content */}
                {msg.content && (
                  msg.role === 'assistant' ? (
                    <div className="prose prose-sm prose-slate max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  )
                )}
                {/* Referenced / generated figures */}
                {msg.figures && msg.figures.length > 0 && (
                  <div className="mt-3 space-y-3">
                    {msg.figures.map((fig, idx) => (
                      <div key={idx} className="border border-slate-200 rounded-lg overflow-hidden bg-white">
                        <img
                          src={fig.base64
                            ? `data:image/${fig.base64.startsWith('iVBOR') ? 'png' : 'jpeg'};base64,${fig.base64}`
                            : `/api/file?path=${encodeURIComponent(fig.path || '')}`}
                          alt={fig.name}
                          className="w-full h-auto"
                          loading="lazy"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                        />
                        <div className="px-3 py-2 border-t border-slate-100 flex items-center justify-between">
                          <p className="text-xs font-medium text-slate-700">{fig.name}</p>
                          <div className="flex gap-1.5">
                            {fig.pdfPath && (
                              <a
                                href={`/api/file?path=${encodeURIComponent(fig.pdfPath)}`}
                                download={`${fig.name}.pdf`}
                                className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-violet-700 bg-violet-50 hover:bg-violet-100 rounded-md transition-colors"
                              >
                                <Download className="w-3 h-3" />
                                PDF
                              </a>
                            )}
                            {fig.path && (
                              <a
                                href={`/api/file?path=${encodeURIComponent(fig.path)}`}
                                download={`${fig.name}.png`}
                                className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-slate-600 bg-slate-50 hover:bg-slate-100 rounded-md transition-colors"
                              >
                                <Download className="w-3 h-3" />
                                PNG
                              </a>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {msg.role === 'user' && (
                <div className="shrink-0 w-8 h-8 rounded-full bg-violet-600 flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
              )}
            </div>
          ))
        )}
        {(chatLoading || chatDelegationActive) && (
          <div className="flex gap-3 justify-start">
            <div className="shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center">
              <Bot className="w-4 h-4 text-violet-600" />
            </div>
            <div className="bg-slate-100 rounded-2xl px-4 py-3">
              {chatDelegationActive ? (
                <div className="flex items-center gap-2 text-sm text-slate-600">
                  <Loader2 className="w-4 h-4 animate-spin text-violet-500 shrink-0" />
                  <div>
                    <div className="font-medium text-violet-700">{chatDelegationPlan || 'Coordinating agents...'}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{language === 'zh' ? '分析进度请查看 Analysis 标签页' : 'Progress in Analysis tab'}</div>
                  </div>
                </div>
              ) : (
                <Loader2 className="w-4 h-4 animate-spin text-violet-500" />
              )}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 flex items-center gap-2 p-2 bg-red-50 border border-red-200 rounded-lg text-red-600 text-xs">
          <AlertCircle className="w-3 h-3 shrink-0" />
          {error}
        </div>
      )}

      {/* Selected files preview with per-file agent/group/sample/type selectors */}
      {selectedFiles.length > 0 && (
        <div className="mx-4 mb-2 space-y-1.5">
          {selectedFiles.map((sf, i) => {
            const preview = filePreviews[sf.file.name + sf.file.size]
            const section = getSection(sf.targetAgent)
            const typeOptions = getTypeOptions(sf.targetAgent)
            const groups = currentSession?.treatmentGroups || []
            const selectedGroupObj = groups.find(g => g.id === sf.groupId)
            const maxSamples = selectedGroupObj?.sampleCount || 0

            return (
              <div
                key={i}
                className="bg-slate-100 border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 space-y-1"
              >
                {/* Row 1: preview + filename + agent + delete */}
                <div className="flex items-center gap-2">
                  {preview ? (
                    <img src={preview} alt="" className="w-6 h-6 rounded object-cover shrink-0" />
                  ) : (
                    <FileText className="w-4 h-4 text-slate-400 shrink-0" />
                  )}
                  <span className="truncate max-w-[140px]">{sf.file.name}</span>
                  <select
                    value={sf.targetAgent}
                    onChange={(e) => handleFileAgentChange(i, e.target.value)}
                    className="ml-auto shrink-0 px-1.5 py-0.5 border border-slate-300 rounded text-[11px] text-slate-600 bg-white focus:ring-1 focus:ring-violet-500"
                  >
                    {AGENT_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {t(opt.labelKey)}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={() => handleRemoveFile(i)}
                    className="p-0.5 hover:bg-slate-200 rounded shrink-0"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>

                {/* Row 2: group + sample + type — only for non-PI agents */}
                {section && (
                  <div className="flex items-center gap-1.5 pl-1">
                    {/* Treatment group */}
                    <select
                      value={sf.groupId || '_none'}
                      onChange={(e) => handleFileGroupChange(i, e.target.value === '_none' ? undefined : e.target.value)}
                      className="px-1 py-0.5 border border-slate-200 rounded text-[11px] bg-white max-w-[100px]"
                    >
                      <option value="_none">{language === 'zh' ? '无分组' : 'No Group'}</option>
                      {groups.map(g => (
                        <option key={g.id} value={g.id}>{g.name}</option>
                      ))}
                    </select>

                    {/* Sample number */}
                    <select
                      value={sf.sampleNumber != null ? String(sf.sampleNumber) : '_none'}
                      disabled={!sf.groupId}
                      onChange={(e) => handleFileSampleChange(i, e.target.value === '_none' ? undefined : parseInt(e.target.value))}
                      className="px-1 py-0.5 border border-slate-200 rounded text-[11px] bg-white disabled:opacity-40 max-w-[60px]"
                    >
                      <option value="_none">-</option>
                      {Array.from({ length: maxSamples }, (_, j) => (
                        <option key={j + 1} value={String(j + 1)}>S{j + 1}</option>
                      ))}
                    </select>

                    {/* Data type */}
                    <select
                      value={sf.dataType || typeOptions[0] || ''}
                      onChange={(e) => handleFileTypeChange(i, e.target.value)}
                      className="px-1 py-0.5 border border-slate-200 rounded text-[11px] bg-white max-w-[90px]"
                    >
                      {typeOptions.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-slate-200 p-4">
        <div className="flex items-end gap-2">
          {/* Attach button */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.csv,.xlsx,.xls,.txt,.json,.tsv"
            onChange={handleAddFiles}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={chatLoading}
            className="shrink-0 p-3 text-slate-500 hover:text-violet-600 hover:bg-slate-100 rounded-xl transition-colors disabled:opacity-50"
            title={t('chatAttach')}
          >
            <Paperclip className="w-5 h-5" />
          </button>

          {/* Default agent selector for new files */}
          <select
            value={defaultAgent}
            onChange={(e) => setDefaultAgent(e.target.value)}
            disabled={chatLoading}
            className="shrink-0 px-2 py-3 border border-slate-300 rounded-xl text-xs text-slate-600 bg-white focus:ring-2 focus:ring-violet-500 focus:border-violet-500 disabled:opacity-50"
            title={language === 'zh' ? '新文件默认Agent' : 'Default agent for new files'}
          >
            {AGENT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {t(opt.labelKey)}
              </option>
            ))}
          </select>

          {/* Textarea with top-edge resize handle */}
          <div className="flex-1 flex flex-col">
            {/* Drag handle at top */}
            <div
              onMouseDown={handleResizeMouseDown}
              className="flex justify-center py-1 cursor-ns-resize group"
            >
              <div className="w-8 h-1 rounded-full bg-slate-300 group-hover:bg-violet-400 transition-colors" />
            </div>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t('chatPlaceholder')}
              className="w-full resize-none px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-violet-500 focus:border-violet-500 text-sm"
              style={{ height: textareaHeight + 'px' }}
              disabled={chatLoading}
            />
          </div>

          {/* Send / Stop button */}
          {chatLoading ? (
            <button
              onClick={handleStop}
              className="shrink-0 p-3 bg-red-500 hover:bg-red-600 text-white rounded-xl transition-colors"
              title={t('chatStop')}
            >
              <Square className="w-5 h-5 fill-current" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim() && selectedFiles.length === 0}
              className="shrink-0 p-3 bg-violet-600 hover:bg-violet-700 text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
