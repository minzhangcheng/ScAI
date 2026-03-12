import { useState } from 'react'
import {
  X,
  Send,
  Brain,
  AlertTriangle,
  TrendingUp,
  Dna,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { Language } from '../i18n/translations'

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

interface ExplorationReportDialogProps {
  report: ExplorationReport
  roundSummary: string
  language: Language
  onChooseDirection: (direction: string) => Promise<void>
  onLetPiDecide: () => Promise<void>
  onClose: () => void
}

export function ExplorationReportDialog({
  report,
  roundSummary,
  language,
  onChooseDirection,
  onLetPiDecide,
  onClose
}: ExplorationReportDialogProps) {
  const [customDirection, setCustomDirection] = useState('')
  const [selectedCandidate, setSelectedCandidate] = useState<number | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [showFullReport, setShowFullReport] = useState(false)

  const handleSubmitDirection = async () => {
    setSubmitting(true)
    try {
      const direction = selectedCandidate !== null
        ? report.research_direction_candidates[selectedCandidate].direction
        : customDirection
      await onChooseDirection(direction)
    } finally {
      setSubmitting(false)
    }
  }

  const handlePiDecide = async () => {
    setSubmitting(true)
    try {
      await onLetPiDecide()
    } finally {
      setSubmitting(false)
    }
  }

  const isZh = language === 'zh'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b bg-gradient-to-r from-violet-50 to-blue-50 rounded-t-2xl">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">
              {isZh ? '探索期分析完成 — 选择研究方向' : 'Exploration Complete — Choose Research Direction'}
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              {roundSummary
                ? roundSummary.substring(0, 120) + (roundSummary.length > 120 ? '...' : '')
                : (isZh ? 'Round 1 全面探索已完成，请查看报告并选择 Round 2 的研究方向' : 'Round 1 exploration is complete. Review the report and choose a direction for Round 2.')}
            </p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white/60 rounded-lg">
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
          {/* Cluster Overview */}
          {report.cluster_overview && (
            <div className="p-4 bg-slate-50 rounded-xl">
              <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2 mb-2">
                <Dna className="w-4 h-4 text-violet-500" />
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
            <div className="p-4 bg-violet-50 border border-violet-200 rounded-xl">
              <h3 className="text-sm font-semibold text-violet-800 flex items-center gap-2 mb-2">
                <Brain className="w-4 h-4" />
                {isZh ? 'PI 研究方向评估' : 'PI Direction Assessment'}
              </h3>
              <p className="text-sm text-violet-700 whitespace-pre-wrap">{report.pi_direction_assessment}</p>
            </div>
          )}

          {/* Research Direction Candidates */}
          {(report.research_direction_candidates || []).length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-violet-500" />
                {isZh ? '推荐研究方向（点击选择）' : 'Recommended Directions (click to select)'}
              </h3>
              <div className="space-y-2">
                {(report.research_direction_candidates || []).map((c, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setSelectedCandidate(selectedCandidate === i ? null : i)
                      setCustomDirection('')
                    }}
                    className={`w-full text-left p-4 rounded-xl border-2 transition-all ${
                      selectedCandidate === i
                        ? 'border-violet-400 bg-violet-50'
                        : 'border-slate-200 hover:border-violet-200 hover:bg-slate-50'
                    }`}
                  >
                    <div className="font-medium text-sm text-slate-800">{c.direction}</div>
                    <div className="text-xs text-slate-500 mt-1">{c.rationale}</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Custom direction input */}
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">
              {isZh ? '或自定义研究方向' : 'Or specify a custom direction'}
            </h3>
            <textarea
              value={customDirection}
              onChange={e => {
                setCustomDirection(e.target.value)
                if (e.target.value) setSelectedCandidate(null)
              }}
              placeholder={isZh ? '例如：重点分析巨噬细胞的极化状态...' : 'e.g., Focus on macrophage polarization state...'}
              className="w-full border rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-violet-300"
              rows={2}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t bg-slate-50 rounded-b-2xl">
          <button
            onClick={handlePiDecide}
            disabled={submitting}
            className="px-5 py-2.5 rounded-xl text-sm font-medium border border-violet-300 text-violet-700 hover:bg-violet-50 transition-all disabled:opacity-50"
          >
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              {isZh ? '让 PI 决定' : 'Let PI Decide'}
            </div>
          </button>

          <button
            onClick={handleSubmitDirection}
            disabled={submitting || (selectedCandidate === null && !customDirection.trim())}
            className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all flex items-center gap-2 ${
              submitting || (selectedCandidate === null && !customDirection.trim())
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-violet-600 text-white hover:bg-violet-700 shadow-md'
            }`}
          >
            <Send className="w-4 h-4" />
            {submitting
              ? (isZh ? '提交中...' : 'Submitting...')
              : (isZh ? '确认方向，开始分析' : 'Start Innovation Analysis')}
          </button>
        </div>
      </div>
    </div>
  )
}
