import { useState, useEffect, useRef, useCallback } from 'react'
import { AlertTriangle, X, ChevronDown, ChevronUp } from 'lucide-react'
import { useTranslation, type TranslationKey } from '../i18n/translations'
import type { LlmFailureInfo as LlmFailureInfoType } from '../stores/appStore'

interface LlmTimeoutDialogProps {
  open: boolean
  llmFailureInfo: LlmFailureInfoType
  timeRemaining: number
  hasCountdown: boolean
  onRetry: () => void
  onModifyAgent: () => void
  onTerminate: () => void
  onSkip: () => void
  onClose: () => void
  language: 'zh' | 'en'
}

const AGENT_LABEL_KEYS: Record<string, TranslationKey> = {
  pi: 'piAgent',
  pathology: 'pathologyAgent',
  bioinformatics: 'bioinformaticsAgent',
  biochemistry: 'biochemistryAgent',
  literature: 'literatureAgent',
}

export default function LlmTimeoutDialog({
  open,
  llmFailureInfo,
  timeRemaining,
  hasCountdown,
  onRetry,
  onModifyAgent,
  onTerminate,
  onSkip,
  onClose,
  language,
}: LlmTimeoutDialogProps) {
  const { t } = useTranslation(language)
  const [showErrorDetail, setShowErrorDetail] = useState(false)
  const dialogRef = useRef<HTMLDivElement>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)

  // Focus trap: save previous focus and focus first button on open
  useEffect(() => {
    if (open) {
      previousFocusRef.current = document.activeElement as HTMLElement
      const timer = setTimeout(() => {
        dialogRef.current?.querySelector<HTMLButtonElement>('button[data-autofocus]')?.focus()
      }, 50)
      return () => clearTimeout(timer)
    } else {
      previousFocusRef.current?.focus()
    }
  }, [open])

  // Escape key to close + Tab trap
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.stopPropagation()
      onClose()
    }
    if (e.key === 'Tab') {
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
      if (!focusable || focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
  }, [onClose])

  if (!open) return null

  const agentLabelKey = AGENT_LABEL_KEYS[llmFailureInfo.agent] || 'piAgent'
  const isMarkerDb = llmFailureInfo.reason === 'marker_db_missing'
  const isAgentDisabled = llmFailureInfo.reason === 'agent_disabled'

  const title = isMarkerDb
    ? t('markerDbMissingTitle')
    : isAgentDisabled
      ? t('agentDisabledTitle')
      : t('llmCallFailed')

  const description = isMarkerDb ? t('markerDbMissingDesc') : null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
      onKeyDown={handleKeyDown}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        className="bg-white dark:bg-surface-900 rounded-2xl shadow-2xl w-full max-w-md mx-4 outline-none"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/60 flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" aria-hidden="true" />
            </div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                {title}
              </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label={t('llmClose')}
          >
            <X className="w-5 h-5" aria-hidden="true" />
          </button>
        </div>

        {description && (
          <div className="px-6 py-4 space-y-3">
            <p className="text-sm text-gray-600">{description}</p>
            {/* Which source failed and why. The marker-DB branch hides the info
                section below, so without this the dialog says only that a
                download failed and the reason -- the host, the transport error
                -- stays in a server log the user cannot open. That reason is
                what the choice turns on: a dropped TLS handshake is worth
                retrying, a 404 is not. */}
            <div>
              <button
                onClick={() => setShowErrorDetail(!showErrorDetail)}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors"
                aria-expanded={showErrorDetail}
              >
                {showErrorDetail
                  ? <ChevronUp className="w-3 h-3" aria-hidden="true" />
                  : <ChevronDown className="w-3 h-3" aria-hidden="true" />
                }
                {t('llmErrorDetails')}
              </button>
              {showErrorDetail && (
                <div className="mt-2 p-3 bg-gray-50 rounded-lg text-xs text-gray-600 font-mono break-words whitespace-pre-wrap max-h-32 overflow-y-auto" role="log">
                  {llmFailureInfo.error}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Info Section */}
        {!isMarkerDb && (<div className="px-6 py-4 space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">{t('llmAgent')}</span>
            <span className="font-medium text-gray-900">
              {t(agentLabelKey as any)}
            </span>
          </div>
          {llmFailureInfo.reason === 'agent_disabled' && (
            <div className="text-sm text-gray-600 bg-amber-50 p-3 rounded-lg">
              {t('agentDisabledBody', { agent: t(agentLabelKey as any) })}
            </div>
          )}
          {llmFailureInfo.reason !== 'agent_disabled' && (
            <>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">{t('llmModel')}</span>
                <span className="font-medium text-gray-900 font-mono text-xs">
                  {llmFailureInfo.model}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">{t('llmServer')}</span>
                <span className="font-medium text-gray-900 font-mono text-xs truncate max-w-[200px]">
                  {llmFailureInfo.llmServer}
                </span>
              </div>
            </>
          )}

          {/* Expandable Error Detail */}
          <div>
            <button
              onClick={() => setShowErrorDetail(!showErrorDetail)}
              className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors"
              aria-expanded={showErrorDetail}
            >
              {showErrorDetail
                ? <ChevronUp className="w-3 h-3" aria-hidden="true" />
                : <ChevronDown className="w-3 h-3" aria-hidden="true" />
              }
              {t('llmErrorDetails')}
            </button>
            {showErrorDetail && (
              <div className="mt-2 p-3 bg-gray-50 rounded-lg text-xs text-gray-600 font-mono break-all max-h-32 overflow-y-auto" role="log">
                {llmFailureInfo.error}
              </div>
            )}
          </div>
        </div>
        )}

        {/* Action Buttons */}
        <div className="px-6 pb-4 space-y-2">
          {isMarkerDb ? (
            <>
              <button
                onClick={onRetry}
                data-autofocus
                className="w-full flex items-center justify-between px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
              >
                <span>{t('markerDbFetch')}</span>
              </button>
              <button
                onClick={onSkip}
                className="w-full px-4 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-colors"
              >
                {t('llmSkip')}
              </button>
              <button
                onClick={onTerminate}
                className="w-full px-4 py-3 bg-red-50 hover:bg-red-100 text-red-700 rounded-xl font-medium transition-colors border border-red-200 text-left"
              >
                {t('llmTerminate')}
              </button>
            </>
          ) : isAgentDisabled ? (
            <>
              <button
                onClick={onModifyAgent}
                data-autofocus
                className="w-full flex items-center justify-between px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
              >
                <span>{t('agentDisabledModifyAgent')}</span>
                <span className="text-xs text-blue-200">
                  {t('llmRetryHint')}
                </span>
              </button>
              <button
                onClick={onRetry}
                className="w-full px-4 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-colors"
              >
                {t('llmRetry')}
              </button>
              <button
                onClick={onSkip}
                className="w-full px-4 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-colors"
              >
                {t('llmSkip')}
              </button>
              <button
                onClick={onTerminate}
                className="w-full px-4 py-3 bg-red-50 hover:bg-red-100 text-red-700 rounded-xl font-medium transition-colors border border-red-200 text-left"
              >
                {t('llmTerminate')}
              </button>
            </>
          ) : (
            <>
              <button
                onClick={onRetry}
                data-autofocus
                className="w-full flex items-center justify-between px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
              >
                <span>{t('llmRetry')}</span>
                <span className="text-xs text-blue-200">
                  {t('llmRetryHint')}
                </span>
              </button>
              <button
                onClick={onModifyAgent}
                className="w-full px-4 py-3 bg-amber-50 hover:bg-amber-100 text-amber-700 rounded-xl font-medium transition-colors border border-amber-200 text-left"
              >
                {t('llmModifyAgent')}
              </button>
              <button
                onClick={onTerminate}
                className="w-full px-4 py-3 bg-red-50 hover:bg-red-100 text-red-700 rounded-xl font-medium transition-colors border border-red-200 text-left"
              >
                {t('llmTerminate')}
              </button>
              <button
                onClick={onSkip}
                className="w-full flex items-center justify-between px-4 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-colors"
              >
                <span>{t('llmSkip')}</span>
                {hasCountdown && (
                  <span className="text-xs text-gray-500" aria-live="polite">
                    {t('llmAutoSkip', { seconds: timeRemaining })}
                  </span>
                )}
              </button>
            </>
          )}
        </div>

        <div className="px-6 pb-4 text-center text-xs text-gray-400">
          {t('llmReopenHint')}
        </div>
      </div>
    </div>
  )
}
