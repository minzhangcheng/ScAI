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

export interface ExplorationReport {
  cluster_overview: string
  data_quality_warnings: string[]
  strongest_heterogeneity: string[]
  key_pathway_changes: string[]
  abnormal_markers: string[]
  low_confidence_annotations: string[]
  pi_direction_assessment: string
  direction_literature_evidence?: DirectionLiteratureEvidence[]
  research_direction_candidates: DirectionCandidate[]
}

export interface PubMedEvidenceItem {
  pmid: string
  title: string
  year?: string
  url?: string
  relevance: string
  supports: string
  source?: string
  species_context?: string
  tissue_context?: string
  why_it_matters: string
}

export interface DirectionSearchQuery {
  category: string
  query: string
  result_count?: number
  relevant_count?: number
  degraded?: boolean
}

export interface DirectionLiteratureEvidence {
  direction_id: string
  direction: string
  status: string
  search_tier?: string
  confidence?: string
  negative_validation_confidence?: string
  direct_evidence_count?: number
  analogous_evidence_count?: number
  contradictory_evidence_count?: number
  methodological_precedent_count?: number
  direct_evidence?: PubMedEvidenceItem[]
  analogous_evidence?: PubMedEvidenceItem[]
  contradictory_evidence?: PubMedEvidenceItem[]
  methodological_precedent?: PubMedEvidenceItem[]
  species_tissue_qualification?: string
  queries?: DirectionSearchQuery[]
  summary?: string
  limitations?: string[]
}

export interface DirectionCandidate {
  direction_id?: string
  direction: string
  rationale: string
  expected_innovation?: string
  expected_novelty?: string
  data_reliability?: string
  literature_novelty?: string
  functional_impact?: string
  overall_priority?: string
  evidence_ref?: string
  biology_expectedness?: string
  data_novelty?: string
  mechanistic_importance?: string
  data_sufficiency?: string
  methodological_gap?: string
  risk_flags?: string[]
  prerequisite_dependencies?: string[]
  species_tissue_qualification?: string
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
  const [expandedEvidence, setExpandedEvidence] = useState<Set<number>>(new Set())
  const toggleEvidence = (index: number) => {
    setExpandedEvidence(prev => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

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
                            <span className="text-xs text-slate-500">{isZh ? '优先级' : 'Priority'}:</span>
                            <ScoreBadge text={c.overall_priority || c.expected_innovation} />
                            <span className="text-xs text-slate-500 ml-1">{isZh ? '数据可靠性' : 'Data'}:</span>
                            <ScoreBadge text={c.data_reliability} />
                            <span className="text-xs text-slate-500 ml-1">{isZh ? '文献新颖性' : 'Lit Novelty'}:</span>
                            <ScoreBadge text={c.literature_novelty || c.expected_novelty} />
                            <span className="text-xs text-slate-500 ml-1">{isZh ? '功能影响' : 'Impact'}:</span>
                            <ScoreBadge text={c.functional_impact} />
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
                      {/* Evidence section */}
                      {(() => {
                        const evidence: DirectionLiteratureEvidence | undefined = report.direction_literature_evidence
                          ? report.direction_literature_evidence.find(e => e.direction_id === c.evidence_ref || e.direction_id === c.direction_id) || report.direction_literature_evidence[i]
                          : undefined
                        if (!evidence) return null
                        const evidenceFailed = evidence.status === 'failed' || evidence.status === 'skipped'
                        const speciesQual = evidence.species_tissue_qualification
                        const qualLabels: Record<string, string> = {
                          matched_species_tissue_available: isZh ? '匹配物种/组织证据可用' : 'Matched species/tissue evidence available',
                          no_matched_species_tissue_evidence: isZh ? '无匹配物种/组织证据' : 'No matched species/tissue evidence',
                          cross_tissue_only: isZh ? '仅跨组织证据' : 'Cross-tissue evidence only',
                          cross_species_only: isZh ? '仅跨物种证据' : 'Cross-species evidence only',
                          mouse_liver_available: isZh ? '匹配物种/组织证据可用' : 'Matched species/tissue evidence available',
                          no_mouse_liver_evidence: isZh ? '无匹配物种/组织证据' : 'No matched species/tissue evidence',
                        }
                        return (
                          <>
                            {/* Evidence status pills */}
                            <div className="px-4 pb-1 pl-12 flex flex-wrap gap-1.5">
                              {evidence.status === 'completed' && (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700">
                                  {isZh ? '证据已验证' : 'Evidence verified'}
                                </span>
                              )}
                              {evidence.status === 'partial' && (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-700">
                                  {isZh ? '部分证据' : 'Partial evidence'}
                                </span>
                              )}
                              {(evidence.status === 'failed' || evidence.status === 'skipped') && (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
                                  {isZh ? '文献验证失败' : 'Literature verification failed'}
                                </span>
                              )}
                              {evidence.search_tier === 'deep' && (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700">
                                  {isZh ? '深度搜索' : 'Deep search used'}
                                </span>
                              )}
                              {speciesQual && qualLabels[speciesQual] && (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-700">
                                  {qualLabels[speciesQual]}
                                </span>
                              )}
                              {(evidence.negative_validation_confidence === 'medium' || evidence.negative_validation_confidence === 'high') && (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-cyan-100 text-cyan-700">
                                  {isZh ? '负向验证' : 'Negative validation'}
                                </span>
                              )}
                            </div>
                            {/* Risk flags */}
                            {c.risk_flags && c.risk_flags.length > 0 && (
                              <div className="px-4 pb-1 pl-12 flex flex-wrap gap-1.5">
                                {c.risk_flags.map((flag, fi) => (
                                  <span key={fi} className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-orange-100 text-orange-700">
                                    {flag === 'requires_new_experiment'
                                      ? (isZh ? '需新实验' : 'Requires new experiment')
                                      : flag}
                                  </span>
                                ))}
                              </div>
                            )}
                            {/* Prerequisite dependencies */}
                            {c.prerequisite_dependencies && c.prerequisite_dependencies.length > 0 && (
                              <div className="px-4 pb-1 pl-12 flex flex-wrap gap-1.5">
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-600">
                                  {isZh ? '未满足的前提条件' : 'Unmet prerequisite'}
                                </span>
                              </div>
                            )}
                            {/* View evidence button */}
                            <div className="px-4 pb-1 pl-12">
                              <button
                                type="button"
                                onClick={e => { e.stopPropagation(); toggleEvidence(i) }}
                                className="text-xs text-accent-600 hover:text-accent-700 flex items-center gap-1"
                              >
                                {expandedEvidence.has(i) ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                {isZh ? '查看文献证据' : 'View evidence'}
                              </button>
                            </div>
                            {/* Expandable evidence details */}
                            {expandedEvidence.has(i) && (
                              <div className="mx-4 mb-4 ml-12 p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-2">
                                {evidenceFailed && (
                                  <div className="text-red-700 bg-red-50 border border-red-100 rounded p-2">
                                    {isZh ? '文献验证失败或不完整；该方向已保守评分。' : 'Literature verification failed or is incomplete; this direction was scored conservatively.'}
                                  </div>
                                )}
                                {evidence.summary && <div>{evidence.summary}</div>}
                                <div className="grid grid-cols-2 gap-2">
                                  <div>{isZh ? '直接证据' : 'Direct'}: {evidence.direct_evidence_count || 0}</div>
                                  <div>{isZh ? '相似证据' : 'Analogous'}: {evidence.analogous_evidence_count || 0}</div>
                                  <div>{isZh ? '反证' : 'Contradictory'}: {evidence.contradictory_evidence_count || 0}</div>
                                  <div>{isZh ? '方法先例' : 'Methods'}: {evidence.methodological_precedent_count || 0}</div>
                                </div>
                                {[
                                  ...(evidence.direct_evidence || []),
                                  ...(evidence.analogous_evidence || []),
                                  ...(evidence.contradictory_evidence || []),
                                  ...(evidence.methodological_precedent || []),
                                ].slice(0, 5).map(item => (
                                  <div key={item.pmid} className="border-t border-slate-200 pt-2">
                                    PMID: {item.pmid} — {item.title}
                                    {item.why_it_matters ? <span className="block text-slate-400">{item.why_it_matters}</span> : null}
                                  </div>
                                ))}
                                {evidence.confidence && (
                                  <div className="text-slate-500">{isZh ? '文献置信度' : 'Literature confidence'}: {evidence.confidence}</div>
                                )}
                                {(evidence.queries || []).length > 0 && (
                                  <div className="border-t border-slate-200 pt-2">
                                    <div className="font-medium">{isZh ? '检索式' : 'Queries'}</div>
                                    {(evidence.queries || []).slice(0, 5).map((q, qi) => (
                                      <div key={qi} className="text-slate-400 truncate">{q.category}: {q.query}</div>
                                    ))}
                                  </div>
                                )}
                                {(evidence.limitations || []).length > 0 && (
                                  <div className="border-t border-slate-200 pt-2">
                                    <div className="font-medium">{isZh ? '局限' : 'Limitations'}</div>
                                    {(evidence.limitations || []).map((lim, li) => (
                                      <div key={li} className="text-slate-400">{lim}</div>
                                    ))}
                                  </div>
                                )}
                                <div className="border-t border-slate-200 pt-2 grid grid-cols-2 gap-1 text-slate-500">
                                  <div>{isZh ? '生物学预期' : 'Biology expectedness'}: {c.biology_expectedness || 'N/A'}</div>
                                  <div>{isZh ? '数据新颖性' : 'Data novelty'}: {c.data_novelty || 'N/A'}</div>
                                  <div>{isZh ? '机制重要性' : 'Mechanistic importance'}: {c.mechanistic_importance || 'N/A'}</div>
                                  <div>{isZh ? '数据充分性' : 'Data sufficiency'}: {c.data_sufficiency || 'N/A'}</div>
                                  <div>{isZh ? '方法学缺口' : 'Methodological gap'}: {c.methodological_gap || 'N/A'}</div>
                                </div>
                              </div>
                            )}
                          </>
                        )
                      })()}
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
