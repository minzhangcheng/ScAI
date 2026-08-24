import { useState, useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import { apiFetch } from '../hooks/useApi'
import ReactMarkdown from 'react-markdown'
import { Send, Loader2, User, Bot, AlertCircle, Paperclip, X, FileText, Image as ImageIcon, Download, Square } from 'lucide-react'
import { base64ToDataUri } from '../utils/image'
import { AGENT_OPTIONS } from '../constants/agents'
import type { ChatMessage } from '../types'
import { reconcileChatHistory, upsertChatMessage } from '../utils/chatMessages'

// Section-specific data type options
import { getSection, getTypeOptions, selectValueForDataType, hasCustomOption, CUSTOM_TYPE_SENTINEL } from '../utils/dataTypes'
import { CustomTypeInput } from './shared/CustomTypeInput'

const MIN_TEXTAREA_HEIGHT = 72

interface SelectedFileWithAgent {
  file: File
  targetAgent: string
  groupId?: string
  sampleNumber?: number
  dataType?: string
}

interface ChatRequestOwner {
  sessionId: string
  requestId: number
  controller: AbortController
  messageId: string
}

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'])

function isImageFile(filename: string): boolean {
  const dotIdx = filename.lastIndexOf('.')
  const ext = dotIdx >= 0 ? filename.toLowerCase().slice(dotIdx) : ''
  return IMAGE_EXTS.has(ext)
}

let fallbackChatIdentity = 0

function createChatMessageId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(bytes)
  } else {
    fallbackChatIdentity += 1
    const value = `${Date.now().toString(16)}${fallbackChatIdentity.toString(16)}`
    for (let index = 0; index < value.length && index < bytes.length * 2; index += 2) {
      bytes[index / 2] = Number.parseInt(value.slice(index, index + 2).padStart(2, '0'), 16)
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function ChatPanel() {
  const {
    language, currentSession,
    chatMessages, addChatMessage, setChatMessages,
    chatLoading, setChatLoading,
    chatDelegationActive, chatDelegationPlan,
    setChatDelegationActive, setChatDelegationPlan,
    setChatAbortController, authUser,
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
  const historyGenerationRef = useRef(0)
  const activeChatSessionRef = useRef<string | null>(currentSession?.id || null)
  const optimisticMessageIdsRef = useRef(new Map<string, Set<string>>())
  const sendRequestSequenceRef = useRef(0)
  const sendOwnerRef = useRef<ChatRequestOwner | null>(null)
  const [textareaHeight, setTextareaHeight] = useState(MIN_TEXTAREA_HEIGHT)
  const resizingRef = useRef(false)
  const resizeStartRef = useRef({ y: 0, height: 0 })

  // Retire all request-owned state before a different session can use it.
  useEffect(() => {
    const sessionId = currentSession?.id || null
    if (activeChatSessionRef.current === sessionId) return
    activeChatSessionRef.current = sessionId

    const owner = sendOwnerRef.current
    if (owner) {
      owner.controller.abort()
      sendOwnerRef.current = null
      if (useAppStore.getState().chatAbortController === owner.controller) {
        setChatAbortController(null)
      }
    }
    setChatLoading(false)
    setChatDelegationActive(false)
    setChatDelegationPlan(null)
    setError(null)
  }, [
    currentSession?.id,
    setChatAbortController,
    setChatDelegationActive,
    setChatDelegationPlan,
    setChatLoading,
  ])

  // Load chat history when session changes
  useEffect(() => {
    const generation = historyGenerationRef.current + 1
    historyGenerationRef.current = generation

    if (!currentSession?.id) {
      setChatMessages([])
      loadedSessionRef.current = null
      return
    }
    if (loadedSessionRef.current === currentSession.id) return
    const sessionId = currentSession.id
    loadedSessionRef.current = sessionId
    for (const cachedSessionId of optimisticMessageIdsRef.current.keys()) {
      if (cachedSessionId !== sessionId) {
        optimisticMessageIdsRef.current.delete(cachedSessionId)
      }
    }
    const optimisticIds = optimisticMessageIdsRef.current.get(sessionId) ?? new Set<string>()
    optimisticMessageIdsRef.current.set(sessionId, optimisticIds)
    setChatMessages(
      useAppStore.getState().chatMessages.filter(message => optimisticIds.has(message.id)),
    )
    const controller = new AbortController()
    const ownsResponse = () => (
      historyGenerationRef.current === generation
      && !controller.signal.aborted
      && useAppStore.getState().currentSession?.id === sessionId
    )

    apiFetch<{ messages: typeof chatMessages }>(
      `/session/${sessionId}/chat`,
      { signal: controller.signal },
      ownsResponse,
    )
      .then((data) => {
        if (ownsResponse()) {
          const current = useAppStore.getState().chatMessages
          const optimistic = current.filter(message => optimisticIds.has(message.id))
          setChatMessages(reconcileChatHistory(data.messages || [], optimistic))
        }
      })
      .catch(() => {
        if (ownsResponse()) {
          setChatMessages(
            useAppStore.getState().chatMessages.filter(
              message => optimisticIds.has(message.id),
            ),
          )
        }
      })

    return () => controller.abort()
  }, [currentSession?.id, setChatMessages])

  // Poll backend during loading as fallback (covers both delegation and slow figure generation)
  useEffect(() => {
    if (!chatLoading || !currentSession?.id) return
    const sessionId = currentSession.id
    let active = true
    const controllers = new Set<AbortController>()
    const interval = setInterval(async () => {
      const controller = new AbortController()
      controllers.add(controller)
      const ownsResponse = () => (
        active
        && !controller.signal.aborted
        && useAppStore.getState().currentSession?.id === sessionId
      )
      try {
        const data = await apiFetch<{ messages: typeof chatMessages }>(
          `/session/${sessionId}/chat`,
          { signal: controller.signal },
          ownsResponse,
        )
        if (!ownsResponse()) return
        const msgs = data.messages || []
        // Reconcile every poll so progressive figure updates are observed even
        // when the server-side message count stays constant.
        const currentChatMessages = useAppStore.getState().chatMessages
        setChatMessages(reconcileChatHistory(msgs, currentChatMessages))
        const lastMsg = msgs[msgs.length - 1]
        if (
          lastMsg
          && lastMsg.role === 'assistant'
          && !lastMsg.delegation_active
          && !lastMsg.pending_figures
        ) {
            setChatLoading(false)
            setChatDelegationActive(false)
            setChatDelegationPlan(null)
        }
      } catch (e) {
        if (ownsResponse()) console.debug('Chat poll failed:', e)
      } finally {
        controllers.delete(controller)
      }
    }, 5000)
    return () => {
      active = false
      clearInterval(interval)
      controllers.forEach(controller => controller.abort())
      controllers.clear()
    }
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
      setTextareaHeight(Math.max(MIN_TEXTAREA_HEIGHT, resizeStartRef.current.height + delta))
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
    setError(null)
  }

  const handleSend = async () => {
    if ((!input.trim() && selectedFiles.length === 0) || !currentSession?.id || chatLoading) return

    // Validate: files with custom type must have a non-empty, non-sentinel value
    const invalidFile = selectedFiles.find(sf =>
      hasCustomOption(getTypeOptions(sf.targetAgent)) &&
      (sf.dataType === CUSTOM_TYPE_SENTINEL || !sf.dataType)
    )
    if (invalidFile) {
      setError(t('customTypeRequired'))
      return
    }

    // Capture session ID at send time to detect cross-session pollution
    const sendSessionId = currentSession.id

    const message = input.trim() || (selectedFiles.length > 0 ? `[${selectedFiles.map(sf => sf.file.name).join(', ')}]` : '')
    setInput('')
    setError(null)
    setTextareaHeight(MIN_TEXTAREA_HEIGHT)

    const filesToSend = [...selectedFiles]
    setSelectedFiles([])

    // Create AbortController for this request
    const abortController = new AbortController()
    const requestId = sendRequestSequenceRef.current + 1
    sendRequestSequenceRef.current = requestId
    const ownsRequest = () => {
      const owner = sendOwnerRef.current
      return Boolean(
        owner
        && owner.sessionId === sendSessionId
        && owner.requestId === requestId
        && owner.controller === abortController
        && useAppStore.getState().currentSession?.id === sendSessionId
      )
    }
    setChatAbortController(abortController)

    // Optimistic: add user message immediately
    const clientMessageId = createChatMessageId()
    const sessionOptimisticIds = optimisticMessageIdsRef.current.get(sendSessionId) ?? new Set<string>()
    sessionOptimisticIds.add(clientMessageId)
    optimisticMessageIdsRef.current.set(sendSessionId, sessionOptimisticIds)
    sendOwnerRef.current = {
      sessionId: sendSessionId,
      requestId,
      controller: abortController,
      messageId: clientMessageId,
    }
    const userMsg = {
      id: clientMessageId,
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
      formData.append('message_id', clientMessageId)
      filesToSend.forEach((sf) => formData.append('files', sf.file))

      const data = await apiFetch<any>(`/session/${sendSessionId}/chat`, {
        method: 'POST',
        body: formData,
        signal: abortController.signal,
      }, ownsRequest)

      // Guard: discard response if user switched session during the request
      if (!ownsRequest()) {
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

      const responseMessage: ChatMessage = {
        id: data.id || createChatMessageId(),
        role: 'assistant',
        content: data.content || '',
        timestamp: data.timestamp || new Date().toISOString(),
        figures: data.figures,
        pending_figures: data.pending_figures,
        reply_to: data.reply_to || clientMessageId,
      }
      setChatMessages(upsertChatMessage(
        useAppStore.getState().chatMessages,
        responseMessage,
      ))
    } catch (e: any) {
      if (!ownsRequest()) return
      if (e.name === 'AbortError') {
        addChatMessage({
          id: createChatMessageId(),
          role: 'assistant',
          content: t('chatCancelled'),
          timestamp: new Date().toISOString(),
          reply_to: clientMessageId,
        })
      } else {
        setError(t('chatError'))
      }
    } finally {
      if (!ownsRequest()) return
      sendOwnerRef.current = null
      const latest = useAppStore.getState()
      if (latest.chatAbortController === abortController) {
        setChatAbortController(null)
      }
      if (!latest.chatDelegationActive) {
        setChatLoading(false)
      }
    }
  }

  const handleStop = async () => {
    if (!currentSession?.id) return

    const stopSessionId = currentSession.id
    const stopRequestSequence = sendRequestSequenceRef.current
    const stoppedOwner = sendOwnerRef.current
    const { chatAbortController: controller, chatDelegationActive: isDelegation } = useAppStore.getState()
    const ownsStop = () => {
      const latest = useAppStore.getState()
      return Boolean(
        latest.currentSession?.id === stopSessionId
        && sendRequestSequenceRef.current === stopRequestSequence
        && sendOwnerRef.current === stoppedOwner
        && latest.chatAbortController === controller
      )
    }

    if (isDelegation) {
      // Cancel delegation via backend endpoint
      try {
        await apiFetch(`/session/${stopSessionId}/chat/cancel`, { method: 'POST' })
      } catch (e) {
        console.error('Failed to cancel chat delegation:', e)
      }
    } else if (controller) {
      // Cancel simple fetch request
      controller.abort()
    }

    if (!ownsStop()) return

    // Immediate UI feedback
    if (useAppStore.getState().chatAbortController === controller) {
      setChatAbortController(null)
    }
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
      <div className="flex-1 overflow-auto p-4 space-y-4 bg-surface-50 dark:bg-surface-950">
        {chatMessages.length === 0 ? (
          <div className="flex-1 flex items-center justify-center h-full min-h-[300px]">
            <div className="text-center text-slate-400">
              <Bot className="w-16 h-16 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium mb-1">{t('chatEmpty')}</p>
              <p className="text-sm">{t('chatEmptyDesc')}</p>
            </div>
          </div>
        ) : (
          chatMessages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="shrink-0 w-8 h-8 rounded-full bg-accent-100 dark:bg-accent-900/60 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-accent-600 dark:text-accent-400" />
                </div>
              )}
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-accent-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-100'
                }`}
              >
                {/* File attachments */}
                {msg.files && msg.files.length > 0 && (
                  <div className="mb-2 space-y-1">
                    {msg.files.map((f: any, j: number) => (
                      <div
                        key={j}
                        className={`flex items-center gap-1.5 text-xs rounded-lg px-2 py-1 ${
                          msg.role === 'user' ? 'bg-accent-500/30' : 'bg-slate-200 dark:bg-slate-700'
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
                            msg.role === 'user' ? 'bg-accent-400/40' : 'bg-accent-100 dark:bg-accent-900/60 text-accent-700 dark:text-accent-300'
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
                {msg.pending_figures && (
                  <div className="mt-2 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-accent-500" />
                    <span>Generating figures...</span>
                  </div>
                )}
                {/* Referenced / generated figures */}
                {msg.figures && msg.figures.length > 0 && (
                  <div className="mt-3 space-y-3">
                    {msg.figures.map((fig, idx) => (
                      <div key={idx} className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden bg-white dark:bg-surface-900">
                        <img
                          src={fig.base64
                            ? base64ToDataUri(fig.base64)
                            : `/api/file?path=${encodeURIComponent(fig.path || '')}`}
                          alt={fig.name}
                          className="publication-canvas w-full h-auto"
                          loading="lazy"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                        />
                        <div className="px-3 py-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
                          <p className="text-xs font-medium text-slate-700 dark:text-slate-200">{fig.name}</p>
                          <div className="flex gap-1.5">
                            {fig.pdfPath && (
                              <a
                                href={`/api/file?path=${encodeURIComponent(fig.pdfPath)}`}
                                download={`${fig.name}.pdf`}
                                className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-accent-700 dark:text-accent-300 bg-accent-50 dark:bg-accent-900/40 hover:bg-accent-100 dark:hover:bg-accent-900/60 rounded-md transition-colors"
                              >
                                <Download className="w-3 h-3" />
                                PDF
                              </a>
                            )}
                            {fig.path && (
                              <a
                                href={`/api/file?path=${encodeURIComponent(fig.path)}`}
                                download={`${fig.name}.png`}
                                className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-md transition-colors"
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
                <div className="shrink-0 w-8 h-8 rounded-full bg-accent-600 flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
              )}
            </div>
          ))
        )}
        {(chatLoading || chatDelegationActive)
          && !(
            chatLoading
            && !chatDelegationActive
            && chatMessages[chatMessages.length - 1]?.role === 'assistant'
            && chatMessages[chatMessages.length - 1]?.pending_figures
          ) && (
          <div className="flex gap-3 justify-start">
            <div className="shrink-0 w-8 h-8 rounded-full bg-accent-100 dark:bg-accent-900/60 flex items-center justify-center">
              <Bot className="w-4 h-4 text-accent-600 dark:text-accent-400" />
            </div>
            <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl px-4 py-3">
              {chatDelegationActive ? (
                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                  <Loader2 className="w-4 h-4 animate-spin text-accent-500 shrink-0" />
                  <div>
                    <div className="font-medium text-accent-700 dark:text-accent-300">{chatDelegationPlan || 'Coordinating agents...'}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{language === 'zh' ? '分析进度请查看 Analysis 标签页' : 'Progress in Analysis tab'}</div>
                  </div>
                </div>
              ) : (
                <Loader2 className="w-4 h-4 animate-spin text-accent-500" />
              )}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 flex items-center gap-2 p-2 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400 text-xs">
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
                className="bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 dark:text-slate-200 space-y-1"
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
                    className="ml-auto shrink-0 px-1.5 py-0.5 border border-slate-300 dark:border-slate-600 rounded text-[11px] text-slate-600 dark:text-slate-300 bg-white dark:bg-surface-900 focus:ring-1 focus:ring-accent-500"
                  >
                    {AGENT_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {t(opt.labelKey)}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={() => handleRemoveFile(i)}
                    className="p-0.5 hover:bg-slate-200 dark:hover:bg-slate-700 rounded shrink-0"
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
                      className="px-1 py-0.5 border border-slate-200 dark:border-slate-700 rounded text-[11px] bg-white dark:bg-surface-900 max-w-[100px]"
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
                      className="px-1 py-0.5 border border-slate-200 dark:border-slate-700 rounded text-[11px] bg-white dark:bg-surface-900 disabled:opacity-40 max-w-[60px]"
                    >
                      <option value="_none">-</option>
                      {Array.from({ length: maxSamples }, (_, j) => (
                        <option key={j + 1} value={String(j + 1)}>S{j + 1}</option>
                      ))}
                    </select>

                    {/* Data type */}
                    <select
                      value={hasCustomOption(typeOptions) ? selectValueForDataType(sf.dataType, typeOptions) : (sf.dataType || typeOptions[0] || '')}
                      onChange={(e) => handleFileTypeChange(i, e.target.value)}
                      className="px-1 py-0.5 border border-slate-200 dark:border-slate-700 rounded text-[11px] bg-white dark:bg-surface-900 max-w-[90px]"
                    >
                      {typeOptions.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                    {hasCustomOption(typeOptions) && (
                      <CustomTypeInput
                        dataType={sf.dataType || ''}
                        typeOptions={typeOptions}
                        onCustomChange={(val) => handleFileTypeChange(i, val)}
                        language={language}
                      />
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-slate-200 dark:border-slate-700 p-4 bg-white dark:bg-surface-900">
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
            disabled={chatLoading || authUser?.status === 'readonly'}
            className="shrink-0 p-3 text-slate-500 dark:text-slate-400 hover:text-accent-600 dark:hover:text-accent-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors disabled:opacity-50"
            title={t('chatAttach')}
          >
            <Paperclip className="w-5 h-5" />
          </button>

          {/* Default agent selector for new files */}
          <select
            value={defaultAgent}
            onChange={(e) => setDefaultAgent(e.target.value)}
            disabled={chatLoading || authUser?.status === 'readonly'}
            className="shrink-0 px-2 py-3 border border-slate-300 dark:border-slate-600 rounded-xl text-xs text-slate-600 dark:text-slate-300 bg-white dark:bg-surface-900 focus:ring-2 focus:ring-accent-500 focus:border-accent-500 disabled:opacity-50"
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
              <div className="w-8 h-1 rounded-full bg-slate-300 dark:bg-slate-600 group-hover:bg-accent-400 transition-colors" />
            </div>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={authUser?.status === 'readonly' ? 'Read-only mode' : t('chatPlaceholder')}
              className="w-full resize-none px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-xl bg-white dark:bg-surface-900 text-surface-800 dark:text-surface-100 focus:ring-2 focus:ring-accent-500 focus:border-accent-500 text-sm"
              style={{ height: textareaHeight + 'px' }}
              disabled={chatLoading || authUser?.status === 'readonly'}
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
              disabled={(!input.trim() && selectedFiles.length === 0) || authUser?.status === 'readonly'}
              className="shrink-0 p-3 bg-accent-600 hover:bg-accent-700 text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
