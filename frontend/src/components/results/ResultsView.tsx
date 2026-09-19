import {
  BookOpen, ChevronDown, ChevronRight, Code2, Download, FileText, FlaskConical,
  GitMerge, Microscope,
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import { resultIdentityKey } from '../../utils/roundKey'
import { reportBranchQuery, withReportBranch } from '../../utils/reportBranch'
import { DownloadButtons, FigureGallery, GraphicAbstract, IntegratedReport } from '../analysis'
import type { ResultsController } from './useResultsController'

type ResultRound = NonNullable<NonNullable<ResultsController['currentSession']>['roundResults']>[number]
type Translate = ResultsController['t']

interface ResultsRoundProps {
  expandedRounds: Set<string>
  language: string
  round: ResultRound
  t: Translate
  toggleRound: (key: string) => void
}
function ResultsRound({ expandedRounds, language, round, t, toggleRound }: ResultsRoundProps) {
            const key = resultIdentityKey(round)
            const isExpanded = expandedRounds.has(key)
            const roundStatusLabel = {
              completed: t('roundStatusCompleted'),
              partial: t('roundStatusPartial'),
              failed: t('roundStatusFailed'),
              skipped: t('roundStatusSkipped'),
            }[round.roundStatus]
            const evidenceStatusLabel = {
              substantive: t('evidenceStatusSubstantive'),
              empty: t('evidenceStatusEmpty'),
              failed: t('evidenceStatusFailed'),
            }[round.evidenceStatus]

            return (
              <div
                key={key}
                className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden bg-white dark:bg-surface-900"
              >
                {/* Round Header */}
                <button
                  onClick={() => toggleRound(key)}
                  className="w-full flex items-center justify-between p-4 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                >
                  <div className="flex flex-wrap items-center gap-3">
                    {isExpanded ? (
                      <ChevronDown className="w-5 h-5 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-slate-400" />
                    )}
                    <span className={`font-semibold ${round.phase === 'integration' ? 'text-purple-800 dark:text-purple-200' : 'text-slate-900 dark:text-slate-100'}`}>
                      {round.phase === 'integration'
                        ? (language === 'zh' ? `整合轮 ${round.round}` : `Integration ${round.round}`)
                        : `Round ${round.round}`}
                    </span>
                    <span className="text-sm text-slate-500 dark:text-slate-400">
                      {(round.pathologyFindings?.length || 0) + (round.bioinformaticsFindings?.length || 0) + (round.biochemistryFindings?.length || 0) + (round.literatureFindings?.length || 0)} {t('findings')}
                    </span>
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      round.roundStatus === 'completed'
                        ? 'bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300'
                        : round.roundStatus === 'partial'
                          ? 'bg-amber-100 dark:bg-amber-900/60 text-amber-700 dark:text-amber-300'
                          : 'bg-rose-100 dark:bg-rose-900/60 text-rose-700 dark:text-rose-300'
                    }`}>
                      {roundStatusLabel}
                    </span>
                    <span className={`rounded-full px-2 py-0.5 text-xs ${
                      round.evidenceStatus === 'substantive'
                        ? 'bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300'
                    }`}>
                      {t('evidenceBadge', { status: evidenceStatusLabel })}
                    </span>
                    {round.directionCoverageEligible != null && (
                      <span className="rounded-full bg-slate-100 dark:bg-slate-800 px-2 py-0.5 text-xs text-slate-600 dark:text-slate-300">
                        {round.directionCoverageEligible
                          ? t('coverageEligible')
                          : t('coverageWithheld')}
                      </span>
                    )}
                    {round.lineageId !== 'main' && (
                      <span
                        className="max-w-40 truncate rounded-full bg-violet-100 dark:bg-violet-900/60 px-2 py-0.5 text-xs text-violet-700 dark:text-violet-300"
                        title={round.branchLabel || round.lineageId}
                      >
                        {round.branchLabel || round.lineageId}
                      </span>
                    )}
                  </div>
                </button>

                {/* Round Content */}
                {isExpanded && (
                  <div className="px-4 pb-4 space-y-4">
                    {/* PI Summary */}
                    {round.summary && (
                      <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                        <p className="text-sm text-slate-700 dark:text-slate-200">{round.summary}</p>
                      </div>
                    )}

                    {round.evidenceNotes.length > 0 && (
                      <div className="rounded-lg border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 p-3">
                        <h4 className="mb-1 text-sm font-medium text-amber-800 dark:text-amber-200">
                          {t('evidenceLimitations')}
                        </h4>
                        <ul className="space-y-1 text-sm text-amber-800 dark:text-amber-200">
                          {round.evidenceNotes.map((note, index) => (
                            <li key={`${round.resultIdentityKey}-evidence-note-${index}`}>• {note}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Integration Report Summary */}
                    {round.phase === 'integration' && round.integrationReportSummary && (
                      <div className="p-3 bg-purple-50 dark:bg-purple-950/40 border border-purple-200 dark:border-purple-800 rounded-lg mt-3">
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
                      <div className="p-3 bg-purple-50 dark:bg-purple-950/40 border border-purple-200 dark:border-purple-800 rounded-lg mt-3">
                        <div className="flex items-center gap-2 mb-2">
                          <GitMerge className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                          <span className="text-sm font-medium text-purple-900 dark:text-purple-200">
                            {language === 'zh' ? '候选研究方向' : 'Candidate Research Directions'}
                          </span>
                        </div>
                        <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-1">
                          {round.integrationCandidates.map((c, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-purple-400 mt-1">•</span>
                              <span>{c.direction}{c.score ? ` (score: ${c.score.toFixed(2)})` : ''}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Findings by Agent */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Pathology */}
                      {round.pathologyFindings?.length > 0 && (
                        <div className="p-3 border border-red-100 dark:border-red-900 bg-red-50/50 dark:bg-red-950/40 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Microscope className="w-4 h-4 text-red-600 dark:text-red-400" />
                            <span className="text-sm font-medium text-red-900 dark:text-red-200">{t('pathologyFindings')}</span>
                          </div>
                          <ul className="text-sm text-red-800 dark:text-red-200 space-y-1">
                            {round.pathologyFindings.map((f, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-red-400 mt-1">•</span>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Bioinformatics */}
                      {round.bioinformaticsFindings?.length > 0 && (
                        <div className="p-3 border border-green-100 dark:border-green-900 bg-green-50/50 dark:bg-green-950/40 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Code2 className="w-4 h-4 text-green-600 dark:text-green-400" />
                            <span className="text-sm font-medium text-green-900 dark:text-green-200">{t('bioinformaticsFindings')}</span>
                          </div>
                          <ul className="text-sm text-green-800 dark:text-green-200 space-y-1">
                            {round.bioinformaticsFindings.map((f, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-green-400 mt-1">•</span>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Biochemistry */}
                      {round.biochemistryFindings?.length > 0 && (
                        <div className="p-3 border border-amber-100 dark:border-amber-900 bg-amber-50/50 dark:bg-amber-950/40 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <FlaskConical className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                            <span className="text-sm font-medium text-amber-900 dark:text-amber-200">{t('biochemistryFindings')}</span>
                          </div>
                          <ul className="text-sm text-amber-800 dark:text-amber-200 space-y-1">
                            {round.biochemistryFindings.map((f, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-amber-400 mt-1">•</span>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Literature */}
                      {round.literatureFindings?.length > 0 && (
                        <div className="p-3 border border-blue-100 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-950/40 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <BookOpen className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                            <span className="text-sm font-medium text-blue-900 dark:text-blue-200">{t('literatureFindings')}</span>
                          </div>
                          <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                            {round.literatureFindings.map((f, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-blue-400 mt-1">•</span>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {(() => {
                        const displayable = (round.crossModalCandidates ?? []).filter(
                          (c) => c.report_recommendation !== 'do_not_emphasize'
                        )
                        if (displayable.length === 0) return null
                        return (
                          <div className="p-3 border border-purple-100 dark:border-purple-900 bg-purple-50/50 dark:bg-purple-950/40 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-sm font-medium text-purple-900 dark:text-purple-200">{t('crossModalCandidates')}</span>
                            </div>
                            <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-1 list-disc ml-5">
                              {displayable.map((c, i) => (
                                <li key={i}>
                                  <span className="font-medium">{c.theme}</span> — {c.mechanism_statement}{' '}
                                  <span className="text-gray-500 dark:text-gray-400">
                                    ({t('crossModalConfidence')}: {c.confidence}; {t('crossModalRelation')}: {c.relation_to_current_direction})
                                  </span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )
                      })()}
                    </div>
                  </div>
                )}
              </div>
            )

}


interface ResultsViewProps {
  controller: ResultsController
}

export function ResultsView({ controller }: ResultsViewProps) {
  const {
    currentSession, curatedCount, downloadReport, error, expandedRounds,
    graphicAbstract, integratedReport, language, loadReport, loading,
    orderedFigures, publicationFigures, publicationTables, t, toggleRound,
  } = controller
  const [targetReportNumber, setTargetReportNumber] = useState<number | null>(null)
  const [targetRequestId, setTargetRequestId] = useState(0)
  const requestFigureReference = useCallback((number: number) => {
    setTargetReportNumber(number)
    setTargetRequestId(requestId => requestId + 1)
  }, [])
  useEffect(() => {
    setTargetReportNumber(null)
    setTargetRequestId(0)
  }, [currentSession?.id, currentSession?.restoreBranchId])
  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 dark:bg-slate-800 rounded-full mb-6">
          <FileText className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="font-display text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
          {t('analysisResults')}
        </h2>
        <p className="text-slate-500 dark:text-slate-400 max-w-md">
          {t('waitingResultsDesc')}
        </p>
      </div>
    )
  }

  if (currentSession.status !== 'completed' && (!currentSession.roundResults || currentSession.roundResults.length === 0)) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 dark:bg-slate-800 rounded-full mb-6">
          <FileText className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="font-display text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
          {t('waitingResults')}
        </h2>
        <p className="text-slate-500 dark:text-slate-400 max-w-md">
          {t('waitingResultsDesc')}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {error && (
        <div
          role="alert"
          className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200"
        >
          <span>
            {language === 'zh'
              ? `报告加载失败：${error}`
              : `Failed to load the report: ${error}`}
          </span>
          <button
            type="button"
            onClick={() => void loadReport()}
            disabled={loading}
            className="rounded-lg bg-red-600 px-3 py-1.5 font-medium text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading
              ? (language === 'zh' ? '重试中…' : 'Retrying…')
              : (language === 'zh' ? '重试' : 'Retry')}
          </button>
        </div>
      )}
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-display text-xl font-semibold text-slate-900 dark:text-slate-100">{t('analysisResults')}</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {t('completedRounds', { n: currentSession.roundResults?.length || 0 })}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {currentSession.explorationReportUrl && (
            <a
              href={withReportBranch(
                currentSession.explorationReportUrl,
                currentSession.restoreBranchId,
              )}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-accent-50 dark:bg-accent-900/40 text-accent-700 dark:text-accent-300 rounded-xl font-medium hover:bg-accent-100 dark:hover:bg-accent-900/60 transition-colors"
            >
              <FileText className="w-4 h-4" />
              {language === 'zh' ? '查看探索报告' : 'View Exploration Report'}
            </a>
          )}
          {currentSession.integrationReportUrl && (
            <a
              href={withReportBranch(
                currentSession.integrationReportUrl,
                currentSession.restoreBranchId,
              )}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-accent-50 dark:bg-accent-900/40 text-accent-700 dark:text-accent-300 rounded-xl font-medium hover:bg-accent-100 dark:hover:bg-accent-900/60 transition-colors"
            >
              <FileText className="w-4 h-4" />
              {language === 'zh' ? '查看整合报告' : 'View Integration Report'}
            </a>
          )}

          {currentSession.status === 'completed' && (
            <a
              href={`/api/session/${currentSession.id}/report/html${reportBranchQuery(currentSession.restoreBranchId)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-accent-50 dark:bg-accent-900/40 text-accent-700 dark:text-accent-300 rounded-xl font-medium hover:bg-accent-100 dark:hover:bg-accent-900/60 transition-colors"
            >
              <FileText className="w-4 h-4" />
              {language === 'zh' ? '查看完整报告' : 'View Full Report'}
            </a>
          )}

          {currentSession.finalSummary && (
            <button
              onClick={downloadReport}
              className="flex items-center gap-2 px-4 py-2 bg-accent-600 text-white rounded-xl font-medium hover:bg-accent-700 transition-colors"
            >
              <Download className="w-4 h-4" />
              {t('downloadReport')}
            </button>
          )}
        </div>
      </div>
      {/* Loading indicator */}
      {loading && (
        <div className="flex items-center justify-center py-4">
          <div className="w-5 h-5 border-2 border-accent-600 border-t-transparent rounded-full animate-spin" />
          <span className="ml-2 text-sm text-slate-500 dark:text-slate-400">{language === 'zh' ? '加载报告中...' : 'Loading report...'}</span>
        </div>
      )}

      {/* Publication Outputs — graphic abstract, then the report, then the figures */}
      {(currentSession.status === 'completed' || publicationFigures.length > 0) && (
        <div className="space-y-4">
          {graphicAbstract && (
            <GraphicAbstract
              graphicAbstract={graphicAbstract}
              sessionId={currentSession?.id}
              branchId={currentSession?.restoreBranchId}
            />
          )}

          {integratedReport && (
            <IntegratedReport
              report={integratedReport}
              onFigureReference={requestFigureReference}
            />
          )}

          <FigureGallery
            figures={orderedFigures}
            curatedCount={curatedCount}
            targetReportNumber={targetReportNumber}
            targetRequestId={targetRequestId}
          />

          {publicationFigures.length > 0 && (
            <DownloadButtons
              figures={publicationFigures}
              tables={publicationTables}
            />
          )}
        </div>
      )}

      {/* Round Results */}
      <div>
        <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
          {t('roundDetails')}
        </h3>
        <div className="space-y-3">

          {(currentSession.roundResults || []).map(round => (
            <ResultsRound
              key={resultIdentityKey(round)}
              expandedRounds={expandedRounds}
              language={language}
              round={round}
              t={t}
              toggleRound={toggleRound}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
