import { AlertCircle, CheckCircle2, Loader2, Pause } from 'lucide-react'
import { MarkdownContent } from '../MarkdownContent'

import { RoundSummaries } from '../analysis/RoundSummaries'
import type { AnalysisPanelController } from './useAnalysisPanelController'

interface AnalysisStatusViewProps {
  controller: AnalysisPanelController
}

export function AnalysisStatusView({ controller }: AnalysisStatusViewProps) {
  const {
    agents, currentSession, expandedRound, getAgentStatus, language, roundSummaries,
    setExpandedRound, t, unresolvedErrors,
  } = controller
  if (!currentSession) return null
  const isRunning = currentSession.status === 'running'

  return (
    <>
      {/* Agent Status - Compact Row */}
      <div className="py-3 border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-2 flex-wrap">
          {agents.map((agent) => {
            const status = getAgentStatus(agent.id)
            const Icon = agent.icon

            return (
              <div
                key={agent.id}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm whitespace-nowrap ${
                  status === 'working'
                    ? 'bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300'
                    : status === 'done'
                    ? 'bg-green-100 dark:bg-green-900/60 text-green-700 dark:text-green-300'
                    : status === 'error'
                    ? 'bg-red-100 dark:bg-red-900/60 text-red-700 dark:text-red-300'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300'
                }`}
              >
                {status === 'working' ? (
                  currentSession.status === 'paused' ? (
                    <Pause className="w-3.5 h-3.5" />
                  ) : (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  )
                ) : (
                  <Icon className="w-3.5 h-3.5" />
                )}
                <span>{agent.name}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Agent Errors Display */}
      {(() => {
        if (unresolvedErrors.length === 0) return null

        return (
          <div className="py-3 border-b border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="w-4 h-4 text-red-500 dark:text-red-400" />
              <span className="text-sm font-medium text-red-700 dark:text-red-300">
                {language === 'zh' ? '分析错误' : 'Analysis Errors'} ({unresolvedErrors.length})
              </span>
            </div>
            <div className="space-y-2">
              {unresolvedErrors.map((error) => {
                const agentInfo = agents.find(a => a.id === error.agent)
                return (
                  <div
                    key={error.id}
                    className="flex items-start gap-2 p-2 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 rounded-lg text-sm"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          error.agent === 'pathology' ? 'bg-rose-100 dark:bg-rose-900/60 text-rose-700 dark:text-rose-300' :
                          error.agent === 'bioinformatics' ? 'bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300' :
                          error.agent === 'biochemistry' ? 'bg-amber-100 dark:bg-amber-900/60 text-amber-700 dark:text-amber-300' :
                          error.agent === 'literature' ? 'bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300' :
                          'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                        }`}>
                          {agentInfo?.name || error.agent}
                        </span>
                        <span className="text-xs text-slate-500 dark:text-slate-400">Round {error.round}</span>
                        <span className="text-xs text-slate-400 dark:text-slate-500">{error.errorType}</span>
                      </div>
                      <p className="text-red-700 dark:text-red-300 mt-1">{error.message}</p>
                      {error.details && (
                        <details className="mt-1">
                          <summary className="text-xs text-red-500 dark:text-red-400 cursor-pointer hover:text-red-700 dark:hover:text-red-300">
                            {language === 'zh' ? '查看详情' : 'View Details'}
                          </summary>
                          <pre className="text-xs text-red-600 dark:text-red-300 mt-1 whitespace-pre-wrap bg-red-100 dark:bg-red-950/40 p-2 rounded max-h-32 overflow-y-auto">
                            {error.details}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )
      })()}

      {/* Main Content Area - Scrollable */}
      <div className="flex-1 overflow-y-auto py-4 space-y-4">
        <RoundSummaries
          roundSummaries={roundSummaries}
          currentSession={currentSession}
          expandedRound={expandedRound}
          setExpandedRound={setExpandedRound}
          isRunning={isRunning}
          language={language}
        />

        {/* Brief Analysis Result */}
        {currentSession.finalSummary && (
          <div className="bg-gradient-to-br from-accent-50 dark:from-accent-950/40 to-accent-100 dark:to-accent-900/40 border border-accent-200 dark:border-accent-800 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-accent-100 dark:bg-accent-900/60 rounded-lg">
                <CheckCircle2 className="w-5 h-5 text-accent-600 dark:text-accent-400" />
              </div>
              <h3 className="font-display font-semibold text-accent-900 dark:text-accent-200">{t('finalReport')}</h3>
            </div>
            <div className="text-slate-700 dark:text-slate-200">
              <MarkdownContent content={currentSession.finalSummary} />
            </div>
          </div>
        )}
      </div>
    </>
  )
}
