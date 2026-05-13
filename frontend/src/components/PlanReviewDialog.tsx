import { useState, useEffect, useRef } from 'react'
import {
  CheckCircle2,
  Clock,
  ChevronDown,
  ChevronUp,
  FileDown,
  Edit3,
} from 'lucide-react'

interface PlanStep {
  id: number
  title: string
  method: string
  input: string
  expected_output: string
  rationale: string
  depends_on: number[]
  status: string
}

interface AnalysisPlan {
  id: string
  phase: string
  stage: string
  overview: string
  steps: PlanStep[]
  key_decision_points: string[]
  expected_runtime: string
  status: string
}

interface PlanReviewDialogProps {
  open: boolean
  plan: AnalysisPlan | null
  dialogueHtml: string
  phase: string
  round: number
  stage: string
  countdownSeconds: number
  onApprove: () => Promise<void>
  onSkip: () => Promise<void>
  onModify: (userInput: string) => Promise<void>
}

export function PlanReviewDialog({
  open,
  plan,
  dialogueHtml,
  phase,
  round,
  stage,
  countdownSeconds,
  onApprove,
  onSkip,
  onModify,
}: PlanReviewDialogProps) {
  const [userInput, setUserInput] = useState('')
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())
  const [countdown, setCountdown] = useState(countdownSeconds)
  const [submitting, setSubmitting] = useState(false)
  const [showModifyInput, setShowModifyInput] = useState(false)

  useEffect(() => {
    setCountdown(countdownSeconds)
  }, [countdownSeconds, open])

  const countdownRef = useRef<number | null>(null)
  const submittingRef = useRef(submitting)
  submittingRef.current = submitting

  useEffect(() => {
    if (countdown <= 0 || countdownSeconds <= 0) return
    if (countdownRef.current) clearInterval(countdownRef.current)
    countdownRef.current = window.setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          if (countdownRef.current) clearInterval(countdownRef.current)
          if (open && !submittingRef.current) {
            onSkip()
          }
          return 0
        }
        return prev - 1
      })
    }, 1000)
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current)
    }
  }, [countdownSeconds, open]) // eslint-disable-line react-hooks/exhaustive-deps

  const phaseLabel = phase === 'exploration' ? '探索阶段' : '创新阶段'
  const stageLabel = stage === 'stage1' ? '阶段 1: QC+聚类+注释'
                     : stage === 'stage2' ? '阶段 2: 下游分析' : '全量分析'

  const toggleStep = (stepId: number) => {
    setExpandedSteps(prev => {
      const next = new Set(prev)
      if (next.has(stepId)) {
        next.delete(stepId)
      } else {
        next.add(stepId)
      }
      return next
    })
  }

  const handleDownloadDialogue = () => {
    if (!dialogueHtml) return
    const blob = new Blob([dialogueHtml], { type: 'text/html' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `plan-dialogue-round${round}-${stage}.html`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleApprove = async () => {
    setSubmitting(true)
    try {
      await onApprove()
    } finally {
      setSubmitting(false)
    }
  }

  const handleSkip = async () => {
    setSubmitting(true)
    try {
      await onSkip()
    } finally {
      setSubmitting(false)
    }
  }

  const handleSubmitModify = async () => {
    if (!userInput.trim()) return
    setSubmitting(true)
    try {
      await onModify(userInput.trim())
    } finally {
      setSubmitting(false)
      setShowModifyInput(false)
    }
  }

  if (!plan) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-2xl shadow-modal w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-gray-900">
              分析计划审核 — {phaseLabel} — Round {round} — {stageLabel}
            </h2>
            {countdown > 0 && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 text-sm font-medium">
                <Clock className="w-4 h-4" />
                {countdown}s
              </span>
            )}
          </div>
          <button
            onClick={handleDownloadDialogue}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <FileDown className="w-4 h-4" />
            下载对话记录
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {/* Overview */}
          {plan.overview && (
            <div className="bg-blue-50 rounded-lg p-4">
              <h3 className="text-sm font-medium text-blue-900 mb-1">概述</h3>
              <p className="text-sm text-blue-800">{plan.overview}</p>
            </div>
          )}

          {/* Steps */}
          <div>
            <h3 className="text-sm font-medium text-gray-900 mb-2">
              分析步骤 ({plan.steps.length} 步)
            </h3>
            <div className="space-y-2">
              {plan.steps.map((step) => {
                const isExpanded = expandedSteps.has(step.id)
                const statusColors: Record<string, string> = {
                  completed: 'bg-green-100 text-green-800',
                  in_progress: 'bg-blue-100 text-blue-800',
                  skipped: 'bg-gray-100 text-gray-600',
                  pending: 'bg-gray-50 text-gray-700',
                }
                const statusBadge = statusColors[step.status] || statusColors.pending

                return (
                  <div key={step.id} className="border border-gray-200 rounded-lg overflow-hidden">
                    <button
                      onClick={() => toggleStep(step.id)}
                      className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-50 transition-colors"
                    >
                      <span className="flex items-center gap-2 flex-1 min-w-0">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${statusBadge}`}>
                          Step {step.id}
                        </span>
                        <span className="text-sm font-medium text-gray-900 truncate">
                          {step.title}
                        </span>
                      </span>
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-gray-400 shrink-0" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-400 shrink-0" />
                      )}
                    </button>
                    {isExpanded && (
                      <div className="px-4 pb-3 space-y-2 text-sm">
                        <div>
                          <span className="text-gray-500">方法：</span>
                          <span className="text-gray-900">{step.method}</span>
                        </div>
                        {step.input && (
                          <div>
                            <span className="text-gray-500">输入：</span>
                            <span className="text-gray-900">{step.input}</span>
                          </div>
                        )}
                        {step.expected_output && (
                          <div>
                            <span className="text-gray-500">预期输出：</span>
                            <span className="text-gray-900">{step.expected_output}</span>
                          </div>
                        )}
                        {step.rationale && (
                          <div>
                            <span className="text-gray-500">理由：</span>
                            <span className="text-gray-900">{step.rationale}</span>
                          </div>
                        )}
                        {step.depends_on.length > 0 && (
                          <div>
                            <span className="text-gray-500">依赖步骤：</span>
                            <span className="text-gray-900">{step.depends_on.join(', ')}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Key Decision Points */}
          {plan.key_decision_points.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-2">关键决策点</h3>
              <div className="flex flex-wrap gap-2">
                {plan.key_decision_points.map((dp, i) => (
                  <span key={i} className="inline-flex items-center px-2.5 py-1 rounded-full bg-amber-100 text-amber-800 text-xs font-medium">
                    {dp}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Expected Runtime */}
          {plan.expected_runtime && (
            <div className="text-sm text-gray-500">
              预计运行时间: {plan.expected_runtime}
            </div>
          )}

          {/* Modify Input */}
          {showModifyInput && (
            <div className="space-y-3">
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder="输入对计划的修改建议..."
                rows={3}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none resize-none"
              />
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => { setShowModifyInput(false); setUserInput('') }}
                  className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  取消
                </button>
                <button
                  onClick={handleSubmitModify}
                  disabled={submitting || !userInput.trim()}
                  className="px-4 py-1.5 text-sm bg-amber-600 text-white rounded-lg hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {submitting ? '提交中...' : '提交修改意见'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 bg-gray-50">
          {!showModifyInput && (
            <button
              onClick={() => setShowModifyInput(true)}
              className="flex items-center gap-1.5 px-4 py-2 text-sm text-amber-700 hover:bg-amber-50 rounded-lg transition-colors"
            >
              <Edit3 className="w-4 h-4" />
              修改
            </button>
          )}
          <button
            onClick={handleSkip}
            disabled={submitting}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          >
            跳过（让 PI 自行决定）
          </button>
          <button
            onClick={handleApprove}
            disabled={submitting}
            className="flex items-center gap-1.5 px-4 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
          >
            <CheckCircle2 className="w-4 h-4" />
            批准方案
          </button>
        </div>
      </div>
    </div>
  )
}