import { useCallback, useEffect } from 'react'

import { apiFetch } from '../../hooks/useApi'
import { useAppStore } from '../../stores/appStore'
import { CUSTOM_TYPE_SENTINEL, getTypeOptions, hasCustomOption } from '../../utils/dataTypes'
import { upsertChatMessage } from '../../utils/chatMessages'
import type { ChatMessage } from '../../types'
import {
  createChatMessageId, MIN_TEXTAREA_HEIGHT, type ChatHistoryController,
} from './useChatHistory'

function useChatAttachmentActions(history: ChatHistoryController) {
  const {
    defaultAgent, fileInputRef, resizeStartRef, resizingRef, selectedFiles, setError,
    setFilePreviews, setSelectedFiles, setTextareaHeight, textareaHeight,
  } = history
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


  return {
    handleResizeMouseDown, handleAddFiles, handleRemoveFile, handleFileAgentChange,
    handleFileGroupChange, handleFileSampleChange, handleFileTypeChange,
  }
}

export function useChatSend(history: ChatHistoryController) {
  const {
    addChatMessage, chatLoading, currentSession, input, optimisticMessageIdsRef,
    selectedFiles, sendOwnerRef, sendRequestSequenceRef, setChatAbortController,
    setChatDelegationActive, setChatDelegationPlan, setChatLoading, setChatMessages,
    setError, setInput, setSelectedFiles, setTextareaHeight, t,
  } = history
  const attachmentActions = useChatAttachmentActions(history)
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


  return { ...attachmentActions, handleSend, handleStop, handleKeyDown }
}

export type ReturnTypeOfChatSend = ReturnType<typeof useChatSend>
