import { useState, useCallback } from 'react'
import { SupplementaryAnalysisModal } from './SupplementaryAnalysisModal'
import {
  X,
  Send,
  Brain,
  AlertTriangle,
  TrendingUp,
  Dna,
  ChevronDown,
  ChevronUp,
  Plus,
  Trash2,
  ExternalLink,
  Clock,
  CheckSquare,
  Square,
} from 'lucide-react'
import { Language } from '../i18n/translations'
import { useCountdownTimer } from '../hooks/useCountdownTimer'

interface ExplorationReport {
  cluster_overview: string
  data_quality_warnings: string[]
  strongest_heterogeneity: string[]
  key_pathway_changes: string[]
  abnormal_markers: string[]
  low_confidence_annotations: string[]
  pi_direction_assessment: string
  research_direction_candidates: Array<{
    direction: string
    rationale: string
    expected_innovation?: string
    expected_novelty?: string
    data_reliability?: string
  }>
}

export interface SelectedDirection {
  direction: string
  notes: string
  isCustom: boolean
}

interface ExplorationReportDialogProps {
  report: ExplorationReport
  roundSummary: string
  explorationReportUrl?: string
  detailedReportUrl?: string
  language: Language
  countdownSeconds: number  // -1=manual, 0=no wait, >0=countdown
  onChooseDirections: (directions: SelectedDirection[]) => Promise<void>
  onLetPiDecide: () => Promise<void>
  onRequestSupplementary: (requestText: string) => Promise<void>
  onClose: () => void
  isSupplementaryRunning?: boolean
}

function ScoreBadge({ text }: { text?: string }) {
  if (!text) return <span className="text-xs text-slate-400">N/A</span>
  const lower = text.toLowerCase()
  let score = ''
  let colorClasses = ''
  if (lower.startsWith('high')) {
    score = 'High'
    colorClasses = 'bg-emerald-100 text-emerald-700'
  } else if (lower.startsWith('medium')) {
    score = 'Medium'
    colorClasses = 'bg-amber-100 text-amber-700'
  } else if (lower.startsWith('low')) {
    score = 'Low'
    colorClasses = 'bg-red-100 text-red-700'
  } else {
    score = text.substring(0, 15)
    colorClasses = 'bg-slate-100 text-slate-600'
  }
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${colorClasses}`}>
      {score}
    </span>
  )
}

export function ExplorationReportDialog({
  report,
  roundSummary,
  explorationReportUrl,
  detailedReportUrl,
  language,
  countdownSeconds,
  onChooseDirections,
  onLetPiDecide,
  onRequestSupplementary,
  onClose,
  isSupplementaryRunning = false,
}: ExplorationReportDialogProps) {
  const isZh = language === 'zh'

  // Multi-select state: set of selected candidate indices
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set())
  // Notes per candidate direction (keyed by index)
  const [candidateNotes, setCandidateNotes] = useState<Record<number, string>>({})
  // Custom directions
  const [customDirections, setCustomDirections] = useState<Array<{ direction: string; notes: string }>>([])

  const [submitting, setSubmitting] = useState(false)
  const [showFullReport, setShowFullReport] = useState(false)
  const [showSupplementaryInput, setShowSupplementaryInput] = useState(false)

  const handlePiDecide = useCallback(async () => {
    setSubmitting(true)
    try {
      await onLetPiDecide()
    } finally {
      setSubmitting(false)
    }
  }, [onLetPiDecide])

  const { timeRemaining, hasCountdown } = useCountdownTimer({
    countdownSeconds,
    onExpire: handlePiDecide,
  })

  const toggleCandidate = (index: number) => {
    setSelectedIndices(prev => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  const updateCandidateNote = (index: number, note: string) => {
    setCandidateNotes(prev => ({ ...prev, [index]: note }))
  }

  const addCustomDirection = () => {
    setCustomDirections(prev => [...prev, { direction: '', notes: '' }])
  }

  const updateCustomDirection = (index: number, field: 'direction' | 'notes', value: string) => {
    setCustomDirections(prev => prev.map((d, i) => i === index ? { ...d, [field]: value } : d))
  }

  const removeCustomDirection = (index: number) => {
    setCustomDirections(prev => prev.filter((_, i) => i !== index))
  }

  const totalSelected = selectedIndices.size + customDirections.filter(d => d.direction.trim()).length
  const canSubmit = totalSelected > 0 && !submitting

  const handleSubmit = useCallback(async () => {
    setSubmitting(true)
    try {
      const directions: SelectedDirection[] = []

      // Add selected candidate directions
      for (const idx of selectedIndices) {
        const c = report.research_direction_candidates[idx]
        if (c) {
          directions.push({
            direction: c.direction,
            notes: candidateNotes[idx] || '',
            isCustom: false,
          })
        }
      }

      // Add custom directions (filter empty)
      for (const d of customDirections) {
        if (d.direction.trim()) {
          directions.push({
            direction: d.direction.trim(),
            notes: d.notes.trim(),
            isCustom: true,
          })
        }
      }

      if (directions.length > 0) {
        await onChooseDirections(directions)
      }
    } finally {
      setSubmitting(false)
    }
  }, [selectedIndices, candidateNotes, customDirections, report, onChooseDirections])

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-modal max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b bg-gradient-to-r from-accent-50 to-accent-100 rounded-t-2xl">
          <div className="flex-1">
            <h2 className="text-lg font-display font-semibold text-slate-800">
              {isZh ? '探索期分析完成 — 选择研究方向' : 'Exploration Complete — Choose Research Directions'}
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              {roundSummary
                ? roundSummary.substring(0, 120) + (roundSummary.length > 120 ? '...' : '')
                : (isZh ? 'Round 1 全面探索已完成，请查看报告并选择研究方向（可多选）' : 'Round 1 exploration is complete. Review the report and choose direction(s) (multi-select).')}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {explorationReportUrl && (
              <a
                href={explorationReportUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-accent-700 bg-accent-100 hover:bg-accent-200 rounded-lg transition-colors"
              >
                <ExternalLink className="w-3.5 h-3.5" />
                {isZh ? '查看详细报告' : 'View Detailed Report'}
              </a>
            )}
            <button onClick={onClose} className="p-2 hover:bg-white/60 rounded-lg">
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
          {/* Cluster Overview */}
          {report.cluster_overview && (
            <div className="p-4 bg-slate-50 rounded-xl">
              <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2 mb-2">
                <Dna className="w-4 h-4 text-accent-500" />
                {isZh ? '亚群概况' : 'Cluster Overview'}
              </h3>
              <p className="text-sm text-slate-600 whitespace-pre-wrap">{report.cluster_overview}</p>
            </div>
          )}

          {/* Data Quality Warnings */}
          {(report.data_quality_warnings || []).length > 0 && (
            <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl">
              <h3 className="text-sm font-semibold text-amber-800 flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4" />
                {isZh ? '数据质量警告' : 'Data Quality Warnings'}
              </h3>
              <ul className="text-sm text-amber-700 space-y-1">
                {(report.data_quality_warnings || []).map((w, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-amber-400 mt-0.5">•</span>
                    {w}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Key Findings (collapsible) */}
          <div>
            <button
              onClick={() => setShowFullReport(!showFullReport)}
              className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-800"
            >
              {showFullReport ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              {isZh ? '查看完整分析结果' : 'View Full Analysis Results'}
            </button>

            {showFullReport && (
              <div className="mt-3 space-y-3">
                {(report.strongest_heterogeneity || []).length > 0 && (
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <h4 className="text-xs font-semibold text-blue-700 mb-1">
                      {isZh ? '最强异质性' : 'Strongest Heterogeneity'}
                    </h4>
                    <ul className="text-xs text-blue-600 space-y-0.5">
                      {(report.strongest_heterogeneity || []).map((h, i) => <li key={i}>• {h}</li>)}
                    </ul>
                  </div>
                )}

                {(report.key_pathway_changes || []).length > 0 && (
                  <div className="p-3 bg-green-50 rounded-lg">
                    <h4 className="text-xs font-semibold text-green-700 mb-1">
                      {isZh ? '关键通路变化' : 'Key Pathway Changes'}
                    </h4>
                    <ul className="text-xs text-green-600 space-y-0.5">
                      {(report.key_pathway_changes || []).map((p, i) => <li key={i}>• {p}</li>)}
                    </ul>
                  </div>
                )}

                {(report.abnormal_markers || []).length > 0 && (
                  <div className="p-3 bg-red-50 rounded-lg">
                    <h4 className="text-xs font-semibold text-red-700 mb-1">
                      {isZh ? '异常 Marker' : 'Abnormal Markers'}
                    </h4>
                    <ul className="text-xs text-red-600 space-y-0.5">
                      {(report.abnormal_markers || []).map((m, i) => <li key={i}>• {m}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* PI Direction Assessment */}
          {report.pi_direction_assessment && (
            <div className="p-4 bg-accent-50 border border-accent-200 rounded-xl">
              <h3 className="text-sm font-semibold text-accent-800 flex items-center gap-2 mb-2">
                <Brain className="w-4 h-4" />
                {isZh ? 'PI 研究方向评估' : 'PI Direction Assessment'}
              </h3>
              <p className="text-sm text-accent-700 whitespace-pre-wrap">{report.pi_direction_assessment}</p>
            </div>
          )}

          {/* Research Direction Candidates — multi-select checkboxes */}
          {(report.research_direction_candidates || []).length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-accent-500" />
                {isZh ? '推荐研究方向（可多选）' : 'Recommended Directions (multi-select)'}
              </h3>
              <div className="space-y-2">
                {(report.research_direction_candidates || []).map((c, i) => {
                  const isSelected = selectedIndices.has(i)
                  return (
                    <div key={i} className={`rounded-xl border-2 transition-all ${
                      isSelected
                        ? 'border-accent-400 bg-accent-50/50'
                        : 'border-slate-200 hover:border-accent-200'
                    }`}>
                      {/* Checkbox + direction header */}
                      <button
                        onClick={() => toggleCandidate(i)}
                        className="w-full text-left p-4 flex items-start gap-3"
                      >
                        <div className="mt-0.5 flex-shrink-0">
                          {isSelected
                            ? <CheckSquare className="w-5 h-5 text-accent-600" />
                            : <Square className="w-5 h-5 text-slate-300" />
                          }
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-sm text-slate-800">{c.direction}</div>
                          {/* Score badges */}
                          <div className="flex flex-wrap gap-2 mt-1.5">
                            <span className="text-xs text-slate-500">{isZh ? '创新性' : 'Innovation'}:</span>
                            <ScoreBadge text={c.expected_innovation} />
                            <span className="text-xs text-slate-500 ml-1">{isZh ? '新颖性' : 'Novelty'}:</span>
                            <ScoreBadge text={c.expected_novelty} />
                            <span className="text-xs text-slate-500 ml-1">{isZh ? '可靠性' : 'Reliability'}:</span>
                            <ScoreBadge text={c.data_reliability} />
                          </div>
                          {/* Show rationale when selected */}
                          {isSelected && (
                            <div className="text-xs text-slate-500 mt-2 leading-relaxed">{c.rationale}</div>
                          )}
                        </div>
                      </button>
                      {/* Notes input when selected */}
                      {isSelected && (
                        <div className="px-4 pb-4 pl-12">
                          <textarea
                            value={candidateNotes[i] || ''}
                            onChange={e => updateCandidateNote(i, e.target.value)}
                            placeholder={isZh ? '备注（可选）：提示 PI 需注意或调整的地方...' : 'Notes (optional): Tips for PI to focus on or adjust...'}
                            className="w-full border border-accent-200 rounded-lg px-3 py-2 text-xs resize-none focus:outline-none focus:ring-2 focus:ring-accent-300 bg-white"
                            rows={2}
                            onClick={e => e.stopPropagation()}
                          />
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Custom Directions */}
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">
              {isZh ? '自定义研究方向' : 'Custom Research Directions'}
            </h3>
            <div className="space-y-2">
              {customDirections.map((d, i) => (
                <div key={i} className="flex gap-2 items-start p-3 border border-slate-200 rounded-xl bg-slate-50/50">
                  <div className="flex-1 space-y-2">
                    <input
                      type="text"
                      value={d.direction}
                      onChange={e => updateCustomDirection(i, 'direction', e.target.value)}
                      placeholder={isZh ? '研究方向...' : 'Research direction...'}
                      className="w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-300"
                    />
                    <textarea
                      value={d.notes}
                      onChange={e => updateCustomDirection(i, 'notes', e.target.value)}
                      placeholder={isZh ? '备注（可选）...' : 'Notes (optional)...'}
                      className="w-full border rounded-lg px-3 py-2 text-xs resize-none focus:outline-none focus:ring-2 focus:ring-accent-300"
                      rows={1}
                    />
                  </div>
                  <button
                    onClick={() => removeCustomDirection(i)}
                    className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors mt-1"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
              <button
                onClick={addCustomDirection}
                className="flex items-center gap-2 px-3 py-2 text-sm text-accent-600 hover:bg-accent-50 rounded-lg transition-colors"
              >
                <Plus className="w-4 h-4" />
                {isZh ? '添加研究方向' : 'Add Direction'}
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t bg-slate-50 rounded-b-2xl">
          <div className="flex items-center gap-3">
            {/* Countdown timer */}
            {hasCountdown && (
              <div className="flex items-center gap-1.5 text-sm text-slate-500">
                <Clock className="w-4 h-4" />
                <span className={timeRemaining <= 30 ? 'text-amber-600 font-medium' : ''}>
                  {formatTime(timeRemaining)}
                </span>
              </div>
            )}
            <button
              onClick={handlePiDecide}
              disabled={submitting || isSupplementaryRunning}
              className="px-5 py-2.5 rounded-xl text-sm font-medium border border-accent-300 text-accent-700 hover:bg-accent-50 transition-all disabled:opacity-50"
            >
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4" />
                {isZh ? '让 PI 决定' : 'Let PI Decide'}
              </div>
            </button>
            {detailedReportUrl && (
              <a
                href={detailedReportUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-accent-600 hover:text-accent-800 flex items-center gap-1"
              >
                <ExternalLink className="w-3.5 h-3.5" />
                {isZh ? '查看详细报告' : 'View Detailed Report'}
              </a>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowSupplementaryInput(true)}
              disabled={submitting || isSupplementaryRunning}
              className="px-4 py-2.5 rounded-xl text-sm font-medium border border-amber-300 text-amber-700 hover:bg-amber-50 transition-all disabled:opacity-50"
            >
              {isSupplementaryRunning
                ? (isZh ? '补充分析中...' : 'Analyzing...')
                : (isZh ? '请求补充分析' : 'Request More Analysis')}
            </button>
            {totalSelected > 0 && (
              <span className="text-xs text-slate-500">
                {isZh ? `已选 ${totalSelected} 个研究方向` : `${totalSelected} direction(s) selected`}
              </span>
            )}
            <button
              onClick={handleSubmit}
              disabled={!canSubmit || isSupplementaryRunning}
              className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all flex items-center gap-2 ${
                !canSubmit || isSupplementaryRunning
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-accent-600 text-white hover:bg-accent-700 shadow-md'
              }`}
            >
              <Send className="w-4 h-4" />
              {submitting
                ? (isZh ? '提交中...' : 'Submitting...')
                : (isZh ? '开始创新分析' : 'Start Innovation Analysis')}
            </button>
          </div>
        </div>
      </div>

      {showSupplementaryInput && (
        <SupplementaryAnalysisModal
          language={language}
          placeholder={isZh
            ? '例如：请进一步搜索关于XXX通路在这个组织中的文献，以帮助判断研究方向'
            : 'e.g., Please search for more literature about XXX pathway to help determine research direction'}
          onSubmit={onRequestSupplementary}
          onClose={() => setShowSupplementaryInput(false)}
        />
      )}
    </div>
  )
}
