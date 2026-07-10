import { useEffect, useRef, useState } from 'react'
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
}: IntegrationGatePanelProps) {
  const { t } = useTranslation(language === 'zh' ? 'zh' : 'en')
  const [sel, setSel] = useState<Record<string, boolean>>({})
  const [submitting, setSubmitting] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
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

  return (
    <div className="integration-gate-panel">
      <h3>{t('integrationGateTitle', { cycle })}</h3>
      {reportUrl && (
        <a href={reportUrl} target="_blank" rel="noopener noreferrer"
           style={{ display: 'inline-block', marginBottom: '0.5rem', color: '#4f46e5' }}>
          {language === 'zh' ? '查看整合报告' : 'View Integration Report'}
        </a>
      )}
      {timerDisplay && (
        <div className="countdown-timer" style={{ marginBottom: '0.5rem', color: '#888' }}>
          {t('integrationGateCountdown')} {timerDisplay}
        </div>
      )}
      {actionError && (
        <ErrorBanner message={actionError} onDismiss={() => setActionError(null)} />
      )}
      <ul>
        {candidates.map((c, i) => {
          const id = c.direction_id ?? String(i)
          return (
            <li key={id}>
              <label>
                <input type="checkbox" checked={!!sel[id]} onChange={() => toggle(id)} />
                {c.direction}
              </label>
            </li>
          )
        })}
      </ul>
      <div className="actions" style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
        <button onClick={submit} disabled={submitting}>{t('integrationGateContinue')}</button>
        <button onClick={handleLetPiDecide} disabled={submitting}>{t('integrationGateLetPiDecide')}</button>
        <button onClick={handleFinalize} disabled={submitting}>{t('integrationGateFinalize')}</button>
      </div>
    </div>
  )
}
