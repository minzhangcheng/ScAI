import { useEffect, useRef, useState } from 'react'
import { useCountdownTimer } from '../hooks/useCountdownTimer'

export interface IntegrationCandidate {
  direction: string
  direction_id?: string
  [k: string]: unknown
}

export interface IntegrationGatePanelProps {
  cycle: number
  candidates: IntegrationCandidate[]
  onSubmit: (selected: IntegrationCandidate[]) => void
  onFinalize: () => void
  onLetPiDecide: () => void
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
  const [sel, setSel] = useState<Record<string, boolean>>({})
  const [submitting, setSubmitting] = useState(false)
  const candidateKey = candidates
    .map((c, i) => `${c.direction_id ?? String(i)}:${c.direction}`)
    .join('|')
  useEffect(() => {
    setSel({})
  }, [cycle, candidateKey])
  const toggle = (id: string) => setSel((s) => ({ ...s, [id]: !s[id] }))
  const submit = async () => {
    if (submitting) return
    setSubmitting(true)
    try {
      onSubmit(candidates.filter((c, i) => sel[c.direction_id ?? String(i)]))
    } finally {
      setSubmitting(false)
    }
  }
  const handleFinalize = async () => {
    if (submitting) return
    setSubmitting(true)
    try {
      await onFinalize()
    } finally {
      setSubmitting(false)
    }
  }
  const handleLetPiDecide = async () => {
    if (submitting) return
    setSubmitting(true)
    try {
      await onLetPiDecide()
    } finally {
      setSubmitting(false)
    }
  }

  // Countdown timer: on expire, let PI decide (same pattern as ExplorationReportDialog)
  const expiredRef = useRef(false)
  useEffect(() => {
    expiredRef.current = false
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

  // No-wait mode: auto-trigger PI decide immediately
  const isNoWait = countdownSeconds === 0
  const onLetPiDecideRef = useRef(onLetPiDecide)
  onLetPiDecideRef.current = onLetPiDecide
  useEffect(() => {
    if (isNoWait && !submitting) {
      onLetPiDecideRef.current()
    }
  }, [isNoWait, submitting])

  return (
    <div className="integration-gate-panel">
      <h3>{language === 'zh'
        ? `整合分析 (第 ${cycle} 次) — 选择新的研究方向`
        : `Integration Analysis (Cycle ${cycle}) — Select New Research Direction`}</h3>
      {reportUrl && (
        <a href={reportUrl} target="_blank" rel="noopener noreferrer"
           style={{ display: 'inline-block', marginBottom: '0.5rem', color: '#4f46e5' }}>
          {language === 'zh' ? '查看整合报告' : 'View Integration Report'}
        </a>
      )}
      {timerDisplay && (
        <div className="countdown-timer" style={{ marginBottom: '0.5rem', color: '#888' }}>
          {language === 'zh' ? '倒计时' : 'Countdown'}: {timerDisplay}
        </div>
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
        <button onClick={submit} disabled={submitting}>{language === 'zh' ? '继续分析所选方向' : 'Continue with Selected'}</button>
        <button onClick={handleLetPiDecide} disabled={submitting}>{language === 'zh' ? '让 PI 决定' : 'Let PI Decide'}</button>
        <button onClick={handleFinalize} disabled={submitting}>{language === 'zh' ? '进入报告撰写' : 'Write Report'}</button>
      </div>
    </div>
  )
}
