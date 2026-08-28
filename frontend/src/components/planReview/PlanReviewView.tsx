import {
  CheckCircle2, ChevronDown, ChevronUp, Clock, Edit3, FileDown, Pause, Play,
} from 'lucide-react'

import type { PlanReviewController } from './usePlanReviewController'

interface PlanReviewViewProps {
  controller: PlanReviewController
}

export function PlanReviewView({ controller }: PlanReviewViewProps) {
  const {
    expandedSteps, handleApprove, handleDownloadDialogue, handleSkip, handleSubmitModify,
    hasCountdown, isPaused, phaseLabel, plan, round, setShowModifyInput, setUserInput,
    showModifyInput, stageLabel, submitting, t, timerDisplay, timerUrgent, toggleStep,
    toggleTimerPause, userInput,
  } = controller
  if (!plan) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {t('planReviewTitle')} — {phaseLabel} — Round {round} — {stageLabel}
            </h2>
          </div>
          <div className="flex items-center gap-3">
            {hasCountdown && (
              <>
                <div className="flex items-center gap-2">
                  <Clock className={`w-4 h-4 ${timerUrgent ? 'text-red-500' : 'text-slate-500 dark:text-slate-400'}`} />
                  <span className={`font-mono text-lg tabular-nums ${
                    timerUrgent ? 'text-red-600 animate-pulse' :
                    isPaused ? 'text-amber-600 dark:text-amber-400' : 'text-slate-700 dark:text-slate-200'
                  }`}>
                    {timerDisplay}
                  </span>
                  {isPaused && (
                    <span className="text-xs text-amber-600 font-medium">{t('timerPaused')}</span>
                  )}
                </div>
                <button
                  onClick={toggleTimerPause}
                  className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-slate-100 transition-colors"
                  title={isPaused ? t('resumeTimer') : t('pauseTimer')}
                >
                  {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                </button>
              </>
            )}
            <button
              onClick={handleDownloadDialogue}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              <FileDown className="w-4 h-4" />
              {t('planReviewDownloadLog')}
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {plan.overview && (
            <div className="bg-blue-50 rounded-lg p-4">
              <h3 className="text-sm font-medium text-blue-900 mb-1">{t('planReviewOverview')}</h3>
              <p className="text-sm text-blue-800">{plan.overview}</p>
            </div>
          )}
          <div>
            <h3 className="text-sm font-medium text-gray-900 mb-2">
              {t('planReviewSteps', { n: plan.steps.length })}
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
                          <span className="text-gray-500">{t('planReviewMethod')}</span>
                          <span className="text-gray-900">{step.method}</span>
                        </div>
                        {step.input && (
                          <div>
                            <span className="text-gray-500">{t('planReviewInput')}</span>
                            <span className="text-gray-900">{step.input}</span>
                          </div>
                        )}
                        {step.expected_output && (
                          <div>
                            <span className="text-gray-500">{t('planReviewExpectedOutput')}</span>
                            <span className="text-gray-900">{step.expected_output}</span>
                          </div>
                        )}
                        {step.figure_requirements && step.figure_requirements.length > 0 && (
                          <div>
                            <span className="text-gray-500">{t('planReviewRequiredFigures')}</span>
                            <span className="text-gray-900">
                              {step.figure_requirements.map((requirement) => (
                                `${requirement.figure_type} ×${requirement.minimum_count}` +
                                (requirement.description ? ` — ${requirement.description}` : '') +
                                (requirement.platform_baseline ? ` [platform: ${requirement.platform_baseline}]` : '')
                              )).join('; ')}
                            </span>
                          </div>
                        )}
                        {step.allow_text_only && (
                          <div>
                            <span className="text-gray-500">{t('planReviewVisualPolicy')}</span>
                            <span className="text-gray-900">
                              {t('planReviewTextOnlyException')}
                              {step.text_only_reason ? ` — ${step.text_only_reason}` : ''}
                            </span>
                          </div>
                        )}
                        {step.rationale && (
                          <div>
                            <span className="text-gray-500">{t('planReviewRationale')}</span>
                            <span className="text-gray-900">{step.rationale}</span>
                          </div>
                        )}
                        {step.depends_on.length > 0 && (
                          <div>
                            <span className="text-gray-500">{t('planReviewDependsOn')}</span>
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
          {plan.key_decision_points.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-2">{t('planReviewKeyDecisions')}</h3>
              <div className="flex flex-wrap gap-2">
                {plan.key_decision_points.map((dp, i) => (
                  <span key={i} className="inline-flex items-center px-2.5 py-1 rounded-full bg-amber-100 text-amber-800 text-xs font-medium">
                    {dp}
                  </span>
                ))}
              </div>
            </div>
          )}
          {plan.expected_runtime && (
            <div className="text-sm text-gray-500">
              {t('planReviewEstimatedTime')} {plan.expected_runtime}
            </div>
          )}
          {showModifyInput && (
            <div className="space-y-3">
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder={t('planReviewPlaceholder')}
                rows={3}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none resize-none"
              />
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => { setShowModifyInput(false); setUserInput('') }}
                  className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  {t('planReviewCancel')}
                </button>
                <button
                  onClick={handleSubmitModify}
                  disabled={submitting || !userInput.trim()}
                  className="px-4 py-1.5 text-sm bg-amber-600 text-white rounded-lg hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {submitting ? t('planReviewSubmitting') : t('planReviewSubmitFeedback')}
                </button>
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 bg-gray-50">
          {!showModifyInput && (
            <button
              onClick={() => setShowModifyInput(true)}
              className="flex items-center gap-1.5 px-4 py-2 text-sm text-amber-700 hover:bg-amber-50 rounded-lg transition-colors"
            >
              <Edit3 className="w-4 h-4" />
              {t('planReviewModify')}
            </button>
          )}
          <button
            onClick={handleSkip}
            disabled={submitting}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          >
            {t('planReviewSkip')}
          </button>
          <button
            onClick={handleApprove}
            disabled={submitting}
            className="flex items-center gap-1.5 px-4 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
          >
            <CheckCircle2 className="w-4 h-4" />
            {t('planReviewApprove')}
          </button>
        </div>
      </div>
    </div>
  )
}
