import { useCallback, useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { useCountdownTimer } from '../hooks/useCountdownTimer'
import { useTranslation } from '../i18n/translations'
import { ErrorBanner } from './shared/ErrorBanner'

export interface IntegrationCandidate {
  direction: string
  direction_id?: string
  [k: string]: unknown
}

export interface IntegrationGatePanelProps {
  cycle: number
  candidates: IntegrationCandidate[]
  onSubmit: (selected: IntegrationCandidate[]) => Promise<void>
  onFinalize: () => Promise<void>
  onLetPiDecide: () => Promise<void>
  countdownSeconds: number  // -1=manual, 0=no-wait, >0=countdown
  language: string
  reportUrl?: string | null
  onClose: () => void
  open?: boolean
}

export function IntegrationGatePanel({
  cycle,
  candidates,
  onSubmit,
  onFinalize,
  onLetPiDecide,
  countdownSeconds,
  language,
  reportUrl,
  onClose,
  open = true,
}: IntegrationGatePanelProps) {
  const { t } = useTranslation(language === 'zh' ? 'zh' : 'en')
  const [sel, setSel] = useState<Record<string, boolean>>({})
  const [submitting, setSubmitting] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const dialogRef = useRef<HTMLDivElement>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)
  const candidateKey = candidates
    .map((c, i) => `${c.direction_id ?? String(i)}:${c.direction}`)
    .join('|')
  useEffect(() => {
    setSel({})
    setActionError(null)
  }, [cycle, candidateKey])
  const toggle = (id: string) => setSel((s) => ({ ...s, [id]: !s[id] }))
  const runAction = async (action: () => Promise<void>) => {
    if (submitting) return
    setSubmitting(true)
    setActionError(null)
    try {
      await action()
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setActionError(t('integrationGateActionFailed', { message }))
    } finally {
      setSubmitting(false)
    }
  }

  const submit = async () => {
    const selected = candidates.filter((c, i) => sel[c.direction_id ?? String(i)])
    if (selected.length === 0) {
      setActionError(t('integrationGateSelectRequired'))
      return
    }
    await runAction(() => onSubmit(selected))
  }

  const handleFinalize = async () => {
    await runAction(onFinalize)
  }

  const handleLetPiDecide = async () => {
    await runAction(onLetPiDecide)
  }

  // Countdown timer: on expire, let PI decide (same pattern as ExplorationReportDialog)
  const expiredRef = useRef(false)
  const noWaitAttemptedRef = useRef(false)
  useEffect(() => {
    expiredRef.current = false
    noWaitAttemptedRef.current = false
  }, [cycle, candidateKey])
  const handleAutoDecide = () => {
    if (!submitting && !expiredRef.current) {
      expiredRef.current = true
      handleLetPiDecide()
    }
  }
  const { timeRemaining, hasCountdown } = useCountdownTimer({
    countdownSeconds,
    onExpire: handleAutoDecide,
    resetKey: `${cycle}-${candidateKey}`,
  })

  const minutes = Math.floor(timeRemaining / 60)
  const seconds = timeRemaining % 60
  const timerDisplay = hasCountdown ? `${minutes}:${seconds.toString().padStart(2, '0')}` : null

  // No-wait mode: auto-trigger PI decide immediately (at most once per gate identity)
  const isNoWait = countdownSeconds === 0
  useEffect(() => {
    if (isNoWait && !submitting && !noWaitAttemptedRef.current) {
      noWaitAttemptedRef.current = true
      void handleLetPiDecide()
    }
  }, [isNoWait, submitting])

  useEffect(() => {
    if (open) {
      previousFocusRef.current = document.activeElement as HTMLElement
      dialogRef.current?.querySelector<HTMLElement>('[data-autofocus]')?.focus()
      return () => {
        previousFocusRef.current?.focus()
      }
    } else {
      previousFocusRef.current?.focus()
    }
  }, [open])

  const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.stopPropagation()
      onClose()
      return
    }
    if (event.key !== 'Tab') return

    const focusable = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) ?? []).filter((element) => (
      !element.matches(':disabled')
      && !element.closest('[hidden], [inert], [aria-hidden="true"]')
    ))
    if (focusable.length === 0) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }, [onClose])

  const titleId = `integration-gate-title-${cycle}`

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        ref={dialogRef}
        onKeyDown={handleKeyDown}
        className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl bg-white dark:bg-surface-900 shadow-2xl"
      >
        <div className="flex items-start justify-between border-b border-slate-200 dark:border-slate-700 px-6 py-4">
          <h2 id={titleId} className="text-xl font-semibold text-slate-900 dark:text-slate-100">
            {t('integrationGateTitle', { cycle })}
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label={language === 'zh' ? '关闭' : 'Close'}
            data-autofocus
            className="rounded-lg p-2 text-slate-500 dark:text-slate-400 transition-colors hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-700 dark:hover:text-slate-200"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>
        <div className="space-y-4 overflow-y-auto px-6 py-5">
          {reportUrl && (
            <a
              href={reportUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex font-medium text-primary-600 hover:text-primary-700 hover:underline"
            >
              {language === 'zh' ? '查看整合报告' : 'View Integration Report'}
            </a>
          )}
          {timerDisplay && (
            <div className="text-sm text-slate-500 dark:text-slate-400">
              {t('integrationGateCountdown')} {timerDisplay}
            </div>
          )}
          {actionError && (
            <ErrorBanner message={actionError} onDismiss={() => setActionError(null)} />
          )}
          <ul className="space-y-3">
            {candidates.map((candidate, index) => {
              const id = candidate.direction_id ?? String(index)
              return (
                <li key={id}>
                  <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-slate-200 dark:border-slate-700 p-4 transition-colors hover:border-primary-300 hover:bg-primary-50/40 dark:hover:bg-primary-950/30">
                    <input
                      type="checkbox"
                      checked={!!sel[id]}
                      onChange={() => toggle(id)}
                      className="mt-1 h-4 w-4 rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="text-sm leading-6 text-slate-700 dark:text-slate-200">{candidate.direction}</span>
                  </label>
                </li>
              )
            })}
          </ul>
        </div>
        <div className="flex flex-wrap justify-end gap-3 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-surface-950 px-6 py-4">
          <button
            type="button"
            onClick={handleLetPiDecide}
            disabled={submitting}
            className="rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-surface-900 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {t('integrationGateLetPiDecide')}
          </button>
          <button
            type="button"
            onClick={handleFinalize}
            disabled={submitting}
            className="rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-surface-900 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {t('integrationGateFinalize')}
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={submitting}
            className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {t('integrationGateContinue')}
          </button>
        </div>
      </div>
    </div>
  )
}
