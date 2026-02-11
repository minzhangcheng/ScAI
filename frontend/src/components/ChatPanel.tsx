import { useState, useEffect, useRef } from 'react'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import { apiFetch } from '../hooks/useApi'
import ReactMarkdown from 'react-markdown'
import { Send, Loader2, User, Bot, AlertCircle } from 'lucide-react'

export function ChatPanel() {
  const {
    language, currentSession,
    chatMessages, addChatMessage, setChatMessages,
    chatLoading, setChatLoading,
  } = useAppStore()
  const { t } = useTranslation(language)
  const [input, setInput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const loadedSessionRef = useRef<string | null>(null)

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
        // Session may not have chat history yet
        setChatMessages([])
      })
  }, [currentSession?.id, setChatMessages])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  // Auto-resize textarea
  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px'
    }
  }

  const handleSend = async () => {
    if (!input.trim() || !currentSession?.id || chatLoading) return

    const message = input.trim()
    setInput('')
    setError(null)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    // Optimistic: add user message immediately
    const userMsg = {
      role: 'user' as const,
      content: message,
      timestamp: new Date().toISOString(),
    }
    addChatMessage(userMsg)
    setChatLoading(true)

    try {
      const response = await apiFetch<{ role: string; content: string }>(
        `/session/${currentSession.id}/chat`,
        {
          method: 'POST',
          body: JSON.stringify({ message, files: [] }),
        }
      )

      addChatMessage({
        role: 'assistant',
        content: response.content,
        timestamp: new Date().toISOString(),
      })
    } catch (e: any) {
      setError(t('chatError'))
    } finally {
      setChatLoading(false)
    }
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
                {msg.role === 'assistant' ? (
                  <div className="prose prose-sm prose-slate max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                )}
                {msg.files && msg.files.length > 0 && (
                  <div className="mt-2 text-xs opacity-70">
                    {msg.files.map((f, j) => (
                      <span key={j} className="inline-block mr-2">
                        {f.filename}
                      </span>
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
        {chatLoading && (
          <div className="flex gap-3 justify-start">
            <div className="shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center">
              <Bot className="w-4 h-4 text-violet-600" />
            </div>
            <div className="bg-slate-100 rounded-2xl px-4 py-3">
              <Loader2 className="w-4 h-4 animate-spin text-violet-500" />
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

      {/* Input area */}
      <div className="border-t border-slate-200 p-4">
        <div className="flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              adjustTextareaHeight()
            }}
            onKeyDown={handleKeyDown}
            placeholder={t('chatPlaceholder')}
            rows={1}
            className="flex-1 resize-none px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-all text-sm"
            disabled={chatLoading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || chatLoading}
            className="shrink-0 p-3 bg-violet-600 hover:bg-violet-700 text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {chatLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
