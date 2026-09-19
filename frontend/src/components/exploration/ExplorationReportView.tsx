import {
  AlertTriangle, Brain, CheckSquare, ChevronDown, ChevronUp, Clock, Dna,
  Plus, Send, Square, Trash2, TrendingUp, X,
} from 'lucide-react'

import DirectionEvidence from '../analysis/DirectionEvidence'
import { StageReviewReportDownload } from '../report/StageReviewReportDownload'
import { SupplementaryAnalysisModal } from '../SupplementaryAnalysisModal'
import type { ExplorationReportController } from './useExplorationReportController'

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

function RecommendationBadge({ value, isZh }: { value?: string; isZh: boolean }) {
  if (!value) return null
  const labelMap: Record<string, string> = {
    prioritize: isZh ? '优先' : 'Prioritize',
    secondary: isZh ? '次要' : 'Secondary',
    needs_more_data: isZh ? '需更多数据' : 'Needs more data',
    avoid: isZh ? '暂不建议' : 'Avoid',
  }
  const cls = value === 'prioritize'
    ? 'bg-emerald-100 text-emerald-700'
    : value === 'secondary'
      ? 'bg-blue-100 text-blue-700'
      : value === 'needs_more_data'
        ? 'bg-amber-100 text-amber-700'
        : 'bg-red-100 text-red-700'
  return <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${cls}`}>{labelMap[value] || value}</span>
}

function DetailList({ title, items }: { title: string; items?: string[] }) {
  if (!items || items.length === 0) return null
  return (
    <div>
      <div className="font-medium text-slate-600">{title}</div>
      <ul className="list-disc pl-4 space-y-0.5">
        {items.map((item, idx) => <li key={idx}>{item}</li>)}
      </ul>
    </div>
  )
}


interface ExplorationReportViewProps {
  controller: ExplorationReportController
}

function DirectionCandidates({ controller }: ExplorationReportViewProps) {
  const {
    candidateNotes, expandedEvidence, isZh, language, report, selectedIndices,
    toggleCandidate, toggleEvidence, updateCandidateNote,
  } = controller
  return (
    <>
          {(report.research_direction_candidates || []).length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-accent-500" />
                {isZh ? '推荐研究方向（可多选）' : 'Recommended Directions (multi-select)'}
              </h3>
              <div className="space-y-2">
                {(report.research_direction_candidates || []).map((c, i) => {
                  const isSelected = selectedIndices.has(i)
                  const lowConfidence = c.recommendation === 'avoid' || c.recommendation === 'needs_more_data'
                  return (
                    <div key={i} className={`rounded-xl border-2 transition-all ${
                      isSelected
                        ? 'border-accent-400 bg-accent-50/50'
                        : 'border-slate-200 hover:border-accent-200'
                    } ${lowConfidence && !isSelected ? 'opacity-80 bg-slate-50/60' : ''}`}>
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
                            {c.contradicts_stated_premise && (
                              <span className="inline-block px-2 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-700">
                                {isZh ? '前提冲突' : 'Premise conflict'}
                              </span>
                            )}
                            <span className="text-xs text-slate-500 ml-1">{isZh ? '建议' : 'Recommendation'}:</span>
                            <RecommendationBadge value={c.recommendation} isZh={isZh} />
                            <span className="text-xs text-slate-500 ml-1">{isZh ? '生物学合理性' : 'Plausibility'}:</span>
                            <ScoreBadge text={c.biological_plausibility} />
                            <span className="text-xs text-slate-500 ml-1">{isZh ? '伪影风险' : 'Artifact risk'}:</span>
                            <ScoreBadge text={c.artifact_risk} />
                          </div>
                          {/* Show rationale when selected */}
                          {isSelected && (
                            <div className="text-xs text-slate-500 mt-2 leading-relaxed">{c.rationale}</div>
                          )}
                          {/* Expanded scientific details */}
                          {isSelected && (
                            <div className="mt-3 space-y-2 text-xs text-slate-600 leading-relaxed">
                              {(c.recommendation === 'needs_more_data' || c.artifact_risk === 'high') && (
                                <div className="bg-amber-50 border border-amber-200 rounded p-2 text-amber-800 mb-2">
                                  {isZh
                                    ? '建议：在将此方向用于 Stage 2 之前，考虑请求补充分析。'
                                    : 'Consider requesting supplementary analysis before choosing this direction for Stage 2.'}
                                </div>
                              )}
                              {c.core_observation && <div><span className="font-medium text-slate-700">{isZh ? '核心观察' : 'Core observation'}:</span> {c.core_observation}</div>}
                              {c.main_hypothesis && <div><span className="font-medium text-slate-700">{isZh ? '主假说' : 'Main hypothesis'}:</span> {c.main_hypothesis}</div>}
                              {c.supporting_data && c.supporting_data.length > 0 && (
                                <div>
                                  <div className="font-medium text-slate-600">{isZh ? '支撑数据' : 'Supporting data'}</div>
                                  {c.supporting_data.map((d, idx) => (
                                    <div key={idx} className="ml-2 text-slate-500">
                                      [{d.source || 'unknown'}] {d.evidence || ''}{d.strength ? ` (strength: ${d.strength})` : ''}
                                    </div>
                                  ))}
                                </div>
                              )}
                              <DetailList title={isZh ? '替代假说' : 'Alternative hypotheses'} items={c.alternative_hypotheses} />
                              <DetailList title={isZh ? '伪影风险' : 'Artifact risks'} items={c.artifact_risks} />
                              <DetailList title={isZh ? '缺失或矛盾证据' : 'Contradictory or missing evidence'} items={c.contradictory_or_missing_evidence} />
                              {c.validation_plan && (
                                <div className="space-y-1">
                                  <div className="font-medium text-slate-600">{isZh ? '验证计划' : 'Validation plan'}</div>
                                  <DetailList title={isZh ? '下一步分析' : 'Next analysis'} items={c.validation_plan.next_analysis} />
                                  <DetailList title={isZh ? '湿实验验证' : 'Wet-lab validation'} items={c.validation_plan.wet_lab_validation} />
                                  <DetailList title={isZh ? '判定标准' : 'Decision criteria'} items={c.validation_plan.decision_criteria} />
                                </div>
                              )}
                              {c.pi_review_summary && <div><span className="font-medium text-slate-700">{isZh ? 'PI评审' : 'PI review'}:</span> {c.pi_review_summary}</div>}
                            </div>
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
                      <DirectionEvidence
                        candidate={c}
                        evidence={report.direction_literature_evidence
                          ? report.direction_literature_evidence.find(e => e.direction_id === c.evidence_ref || e.direction_id === c.direction_id) || report.direction_literature_evidence[i]
                          : undefined}
                        expanded={expandedEvidence.has(i)}
                        onToggle={() => toggleEvidence(i)}
                        language={language}
                      />
                    </div>
                  )
                })}
              </div>
            </div>
          )}
    </>
  )
}

export function ExplorationReportView({ controller }: ExplorationReportViewProps) {
  const {
    addCustomDirection, canSubmit, customDirections, explorationReportUrl,
    formatTime, handlePiDecide, handleSubmit, hasCountdown, isSupplementaryRunning, isZh,
    language, onClose, onRequestSupplementary, removeCustomDirection, report, roundSummary,
    setShowFullReport, setShowSupplementaryInput, showFullReport, showSupplementaryInput,
    submitting, timeRemaining, totalSelected, updateCustomDirection,
  } = controller
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col animate-scale-in">
        <div className="flex items-center justify-between px-6 py-4 border-b bg-gradient-to-r from-accent-50 to-accent-100 rounded-t-2xl">
          <div className="flex-1">
            <h2 className="text-lg font-display font-semibold text-slate-800 dark:text-surface-100">
              {isZh ? '探索期分析完成 — 选择研究方向' : 'Exploration Complete — Choose Research Directions'}
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              {roundSummary
                ? roundSummary.substring(0, 120) + (roundSummary.length > 120 ? '...' : '')
                : (isZh ? 'Round 1 全面探索已完成，请查看报告并选择研究方向（可多选）' : 'Round 1 exploration is complete. Review the report and choose direction(s) (multi-select).')}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <StageReviewReportDownload url={explorationReportUrl} language={language} stage={2} />
            <button onClick={onClose} className="p-2 hover:bg-white/60 rounded-lg">
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
          {report.cluster_overview && (
            <div className="p-4 bg-slate-50 rounded-xl">
              <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2 mb-2">
                <Dna className="w-4 h-4 text-accent-500" />
                {isZh ? '亚群概况' : 'Cluster Overview'}
              </h3>
              <p className="text-sm text-slate-600 whitespace-pre-wrap">{report.cluster_overview}</p>
            </div>
          )}
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
          {report.pi_direction_assessment && (
            <div className="p-4 bg-accent-50 border border-accent-200 rounded-xl">
              <h3 className="text-sm font-semibold text-accent-800 flex items-center gap-2 mb-2">
                <Brain className="w-4 h-4" />
                {isZh ? 'PI 研究方向评估' : 'PI Direction Assessment'}
              </h3>
              <p className="text-sm text-accent-700 whitespace-pre-wrap">{report.pi_direction_assessment}</p>
            </div>
          )}
          <DirectionCandidates controller={controller} />
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
        <div className="flex items-center justify-between px-6 py-4 border-t bg-slate-50 rounded-b-2xl">
          <div className="flex items-center gap-3">
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
