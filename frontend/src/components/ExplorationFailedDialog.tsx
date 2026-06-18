import { useEffect, useRef, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { AlertTriangle, X, Database, Play, Square } from 'lucide-react'
import { useTranslation } from '../i18n/translations'
import type { ExplorationFailedInfo } from '../stores/appStore'

interface ExplorationFailedDialogProps {
  open: boolean
  info: ExplorationFailedInfo | null
  language: 'zh' | 'en'
  onGoToData: () => void
  onContinue: () => void
  onStop: () => void
}

/**
 * Prompts the user when exploration (Stage 1: QC/clustering/annotation) did not
 * complete. Per the product decision in the analysis-flow-fixes plan, this is
 * most often a data problem, so we do NOT silently auto-retry — we ask the user
 * to review/re-upload data, retry Stage 1, or stop.
 */
export default function ExplorationFailedDialog({
  open,
  info,
  language,
  onGoToData,
  onContinue,
  onStop,
}: ExplorationFailedDialogProps) {
  const { t } = useTranslation(language)
  const reasonText =
    info?.reason === 'did_not_converge' ? t('explorationFailedReason_did_not_converge')
    : info?.reason === 'analysis_error' ? t('explorationFailedReason_analysis_error')
    : info?.reason
  const dialogRef = useRef<HTMLDivElement>(null)

  // Escape to dismiss (treated as "Continue" — the workflow is already paused
  // at the round gate; closing the dialog leaves control with the user).
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.stopPropagation()
      onContinue()
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
  }, [onContinue])

  useEffect(() => {
    if (open) {
      const timer = setTimeout(() => {
        dialogRef.current?.querySelector<HTMLButtonElement>('button[data-autofocus]')?.focus()
      }, 50)
      return () => clearTimeout(timer)
    }
  }, [open])

  if (!open || !info) return null

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onKeyDown={handleKeyDown}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="exploration-failed-title"
        className="bg-white rounded-2xl shadow-modal w-full max-w-md mx-4 overflow-hidden animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <h2 id="exploration-failed-title" className="text-lg font-display font-semibold text-surface-900">
              {t('explorationFailedTitle')}
            </h2>
          </div>
          <button
            onClick={onContinue}
            className="p-1 hover:bg-surface-100 rounded-lg transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-surface-400" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-5 space-y-3">
          <div className="flex gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
            <p className="text-sm text-surface-700 leading-relaxed">
              {info.message || t('explorationFailedMessage')}
            </p>
          </div>
          {info.reason && (
            <div className="text-xs text-surface-500 bg-surface-50 rounded-lg px-3 py-2">
              <span className="font-medium">{t('explorationFailedReasonLabel')}: </span>
              <span>{reasonText}</span>
              {typeof info.round === 'number' && (
                <span className="ml-2 text-surface-400">· round {info.round}</span>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-2 px-6 py-4 border-t border-surface-200 bg-surface-50">
          <button
            data-autofocus
            onClick={onGoToData}
            className="px-4 py-2 text-sm font-medium text-white bg-accent-600 hover:bg-accent-700 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Database className="w-4 h-4" />
            {t('explorationFailedGoToData')}
          </button>
          <div className="flex gap-2">
            <button
              onClick={onContinue}
              className="flex-1 px-4 py-2 text-sm font-medium text-surface-700 bg-white hover:bg-surface-100 border border-surface-200 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              {t('explorationFailedContinue')}
            </button>
            <button
              onClick={onStop}
              className="flex-1 px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Square className="w-4 h-4" />
              {t('explorationFailedStop')}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}