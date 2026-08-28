import { useEffect, useRef, useState } from 'react'

import { apiFetch } from '../../hooks/useApi'
import { useTranslation } from '../../i18n/translations'
import { useAppStore } from '../../stores/appStore'
import { reconcileChatHistory } from '../../utils/chatMessages'

export const MIN_TEXTAREA_HEIGHT = 72

export interface SelectedFileWithAgent {
  file: File
  targetAgent: string
  groupId?: string
  sampleNumber?: number
  dataType?: string
}

export interface ChatRequestOwner {
  sessionId: string
  requestId: number
  controller: AbortController
  messageId: string
}

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'])

export function isImageFile(filename: string): boolean {
  const dotIdx = filename.lastIndexOf('.')
  const ext = dotIdx >= 0 ? filename.toLowerCase().slice(dotIdx) : ''
  return IMAGE_EXTS.has(ext)
}

let fallbackChatIdentity = 0

export function createChatMessageId(): string {
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

export function useChatHistory() {
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


  return {
    language, currentSession, chatMessages, addChatMessage, setChatMessages, chatLoading,
    setChatLoading, chatDelegationActive, chatDelegationPlan, setChatDelegationActive,
    setChatDelegationPlan, setChatAbortController, authUser, t, input, setInput, error,
    setError, selectedFiles, setSelectedFiles, defaultAgent, setDefaultAgent, filePreviews,
    setFilePreviews, messagesEndRef, textareaRef, fileInputRef, optimisticMessageIdsRef,
    sendRequestSequenceRef, sendOwnerRef, textareaHeight, setTextareaHeight, resizingRef,
    resizeStartRef,
  }
}

export type ChatHistoryController = ReturnType<typeof useChatHistory>
