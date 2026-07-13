import { useState, useEffect } from 'react'
import { X, Send, Clock, MessageSquare } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { useCountdownTimer } from '../hooks/useCountdownTimer'

interface TextInputDialogProps {
  title: string
  description: string
  suggestions?: string[]
  placeholder?: string
  language: 'zh' | 'en'
  countdownSeconds: number
  resetKey?: string
  onSubmit: (input: string) => Promise<void>
  onClose?: () => void
}

export function TextInputDialog({
  title,
  description,
  suggestions,
  placeholder,
  language,
  countdownSeconds,
  onSubmit,
  onClose,
  resetKey,
}: TextInputDialogProps) {
  const [input, setInput] = useState('')
  const [submitting, setSubmitting] = useState(false)

  // Backend governs the actual timeout; the dialog only displays a countdown.
  const { timeRemaining, hasCountdown } = useCountdownTimer({
    countdownSeconds,
    resetKey,
  })

  // Reset the input field whenever the dialog is reused for a new prompt.
  useEffect(() => {
    setInput('')
  }, [resetKey])

  const handleSubmit = async () => {
    if (!input.trim() || submitting) return
    setSubmitting(true)
    try {
      await onSubmit(input.trim())
      onClose?.()
    } finally {
      setSubmitting(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full mx-4 max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-accent-600" />
            <h3 className="font-display font-semibold text-slate-900">{title}</h3>
          </div>
          <div className="flex items-center gap-3">
            {hasCountdown && (
              <div className="flex items-center gap-1 text-sm text-slate-500">
                <Clock className="w-4 h-4" />
                <span>{timeRemaining}s</span>
              </div>
            )}
            {onClose && (
              <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-lg">
                <X className="w-4 h-4 text-slate-400" />
              </button>
            )}
          </div>
        </div>

        <div className="px-6 py-4 overflow-y-auto flex-1 space-y-4">
          <div className="prose prose-sm prose-slate max-w-none">
            <ReactMarkdown>{description}</ReactMarkdown>
          </div>

          {Array.isArray(suggestions) && suggestions.length > 0 && (
            <div className="bg-slate-50 rounded-lg p-3 space-y-1">
              <p className="text-xs font-medium text-slate-500 uppercase">
                {language === 'zh' ? '建议选项' : 'Suggestions'}
              </p>
              {suggestions.map((s, i) => (
                <p key={i} className="text-sm text-slate-700">
                  {i + 1}. {s}
                </p>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder || (language === 'zh' ? '请输入...' : 'Type your response...')}
              className="flex-1 px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent-500 focus:border-transparent"
              autoFocus
              disabled={submitting}
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || submitting}
              className="px-4 py-2 bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1 text-sm"
            >
              <Send className="w-4 h-4" />
              {language === 'zh' ? '发送' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}