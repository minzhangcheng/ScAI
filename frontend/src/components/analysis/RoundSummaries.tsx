// Per-round analysis progress list. Split out of AnalysisPanel.tsx; JSX copied verbatim.
import {
  AlertCircle,
  ChevronDown,
  ChevronUp,
  FlaskConical,
  GitMerge,
  Loader2,
  Microscope,
} from 'lucide-react'
import type { Session } from '../../stores/appStore'
import type { RoundSummary } from '../../types'
import { resultIdentityKey } from '../../utils/roundKey'

interface RoundSummariesProps {
  roundSummaries: RoundSummary[]
  currentSession: Session
  expandedRound: string | null
  setExpandedRound: (value: string | null) => void
  isRunning: boolean
  language: string
}

export function RoundSummaries({
  roundSummaries,
  currentSession,
  expandedRound,
  setExpandedRound,
  isRunning,
  language,
}: RoundSummariesProps) {
  return (
        <div>
          <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
            {language === 'zh' ? '分析进度' : 'Analysis Progress'}
          </h3>
          
          {roundSummaries.length === 0 && isRunning && (
            <div className="p-6 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded-xl text-center">
              <Loader2 className="w-8 h-8 text-blue-500 mx-auto mb-3 animate-spin" />
              <p className="font-medium text-blue-900 dark:text-blue-200">
                {language === 'zh' ? 'Round 1 分析进行中...' : 'Round 1 analysis in progress...'}
              </p>
              <p className="text-sm text-blue-600 dark:text-blue-400 mt-1">
                {language === 'zh' 
                  ? 'PI Agent 正在协调各专家进行分析，请稍候' 
                  : 'PI Agent is coordinating analysis, please wait'}
              </p>
            </div>
          )}
          
          {roundSummaries.length === 0 && !isRunning && currentSession.status === 'pending' && (
            <div className="p-6 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl text-center">
              <p className="text-slate-600 dark:text-slate-300">
                {language === 'zh' 
                  ? '点击"开始分析"启动多智能体分析流程' 
                  : 'Click "Start Analysis" to begin the multi-agent analysis workflow'}
              </p>
            </div>
          )}
          
          <div className="space-y-3">
            {roundSummaries.map((round) => (
              <div
                key={resultIdentityKey(round)}
                className={`border rounded-xl overflow-hidden transition-all ${
                  round.status === 'waiting_input'
                    ? 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/40'
                    : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-surface-900'
                }`}
              >
                {/* Round Header */}
                <button
                  onClick={() => setExpandedRound(
                    expandedRound === resultIdentityKey(round)
                      ? null
                      : resultIdentityKey(round)
                  )}
                  className="w-full flex items-center justify-between p-4 text-left hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className={`px-2.5 py-1 rounded-full text-sm font-medium ${
                      round.phase === 'integration'
                        ? 'bg-purple-200 dark:bg-purple-800/70 text-purple-800 dark:text-purple-200'
                        : round.status === 'waiting_input'
                        ? 'bg-amber-200 dark:bg-amber-800/70 text-amber-800 dark:text-amber-200'
                        : round.status === 'analyzing'
                        ? 'bg-blue-200 dark:bg-blue-800/70 text-blue-800 dark:text-blue-200'
                        : 'bg-green-200 dark:bg-green-800/70 text-green-800 dark:text-green-200'
                    }`}>
                      {round.phase === 'integration'
                        ? (language === 'zh' ? `整合轮 ${round.round}` : `Integration ${round.round}`)
                        : `Round ${round.round}`}
                    </span>
                    <span className="text-sm text-slate-600 dark:text-slate-300 line-clamp-1">
                      {(round.summary || '').substring(0, 80)}...
                    </span>
                  </div>
                  {expandedRound === resultIdentityKey(round) ? (
                    <ChevronUp className="w-5 h-5 text-slate-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-slate-400" />
                  )}
                </button>

                {/* Round Details */}
                {expandedRound === resultIdentityKey(round) && (
                  <div className="px-4 pb-4 border-t border-slate-200 dark:border-slate-700">
                    {/* Summary */}
                    <div className="mt-3">
                      <h4 className="text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
                        {language === 'zh' ? '本轮总结' : 'Round Summary'}
                      </h4>
                      <p className="text-sm text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800 p-3 rounded-lg">
                        {round.summary || ''}
                      </p>
                    </div>

                    {/* Findings */}
                    {round.findings.length > 0 && (
                      <div className="mt-3">
                        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
                          {language === 'zh' ? '主要发现' : 'Key Findings'}
                        </h4>
                        <ul className="text-sm text-slate-600 dark:text-slate-300 space-y-1">
                          {round.findings.slice(0, 5).map((finding, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-green-500 mt-0.5">•</span>
                              <span>{finding}</span>
                            </li>
                          ))}
                          {round.findings.length > 5 && (
                            <li className="text-slate-400 italic">
                              +{round.findings.length - 5} {language === 'zh' ? '更多发现...' : 'more findings...'}
                            </li>
                          )}
                        </ul>
                      </div>
                    )}

                    {/* Integration Report Summary */}
                    {round.phase === 'integration' && round.integrationReportSummary && (
                      <div className="mt-3 p-3 bg-purple-50 dark:bg-purple-950/40 border border-purple-200 dark:border-purple-800 rounded-lg">
                        <h4 className="text-sm font-medium text-purple-800 dark:text-purple-200 mb-1">
                          {language === 'zh' ? '整合分析摘要' : 'Integration Summary'}
                        </h4>
                        <p className="text-sm text-purple-700 dark:text-purple-300 whitespace-pre-line">
                          {round.integrationReportSummary}
                        </p>
                      </div>
                    )}

                    {/* Integration Candidates */}
                    {round.phase === 'integration' && round.integrationCandidates && round.integrationCandidates.length > 0 && (
                      <div className="mt-3 p-3 bg-purple-50 dark:bg-purple-950/40 border border-purple-200 dark:border-purple-800 rounded-lg">
                        <h4 className="text-sm font-medium text-purple-800 dark:text-purple-200 mb-2 flex items-center gap-2">
                          <GitMerge className="w-4 h-4" />
                          {language === 'zh' ? '候选研究方向' : 'Candidate Research Directions'}
                        </h4>
                        <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-1">
                          {round.integrationCandidates.map((c, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-purple-400 mt-0.5">•</span>
                              <span>{c.direction}{c.score ? ` (score: ${c.score.toFixed(2)})` : ''}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Data Requests */}
                    {(round.dataRequests.pathology.length > 0 ||
                      round.dataRequests.biochemistry.length > 0 ||
                      round.dataRequests.other.length > 0) && (
                      <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded-lg">
                        <h4 className="text-sm font-medium text-amber-800 dark:text-amber-200 mb-2 flex items-center gap-2">
                          <AlertCircle className="w-4 h-4" />
                          {language === 'zh' ? '需要补充的数据' : 'Additional Data Requested'}
                        </h4>

                        {round.dataRequests.pathology.length > 0 && (
                          <div className="mb-2">
                            <span className="text-xs font-medium text-red-700 dark:text-red-300 flex items-center gap-1">
                              <Microscope className="w-3 h-3" />
                              {language === 'zh' ? '病理染色:' : 'Pathology:'}
                            </span>
                            <ul className="text-sm text-slate-700 dark:text-slate-200 ml-4">
                              {round.dataRequests.pathology.map((item, idx) => (
                                <li key={idx}>• {item}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {round.dataRequests.biochemistry.length > 0 && (
                          <div className="mb-2">
                            <span className="text-xs font-medium text-amber-700 dark:text-amber-300 flex items-center gap-1">
                              <FlaskConical className="w-3 h-3" />
                              {language === 'zh' ? '生化指标:' : 'Biochemistry:'}
                            </span>
                            <ul className="text-sm text-slate-700 dark:text-slate-200 ml-4">
                              {round.dataRequests.biochemistry.map((item, idx) => (
                                <li key={idx}>• {item}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {round.dataRequests.other.length > 0 && (
                          <div>
                            <span className="text-xs font-medium text-slate-700 dark:text-slate-200">
                              {language === 'zh' ? '其他:' : 'Other:'}
                            </span>
                            <ul className="text-sm text-slate-700 dark:text-slate-200 ml-4">
                              {round.dataRequests.other.map((item, idx) => (
                                <li key={idx}>• {item}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
  )
}
