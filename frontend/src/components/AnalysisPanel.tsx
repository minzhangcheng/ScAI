import { RoundSummaries } from './analysis/RoundSummaries'
import ReactMarkdown from 'react-markdown'
import { RoundFeedbackDialog } from './RoundFeedbackDialog'
import { AnnotationConfirmationDialog } from './AnnotationConfirmationDialog'
import { ExplorationReportDialog } from './ExplorationReportDialog'
import { IntegrationGatePanel } from './IntegrationGatePanel'
import { PlanReviewDialog } from './PlanReviewDialog'
import { ValidationDialog } from './ValidationDialog'
import { StopDialog } from './StopDialog'
import LlmTimeoutDialog from './LlmTimeoutDialog'
import LlmModifySubDialog from './LlmModifySubDialog'
import ExplorationFailedDialog from './ExplorationFailedDialog'
import { SnapshotPanel } from './SnapshotPanel'
import { SnapshotFailureDialog } from './snapshots/SnapshotFailureDialog'
import { RuntimeWarningBanner } from './snapshots/RuntimeWarningBanner'
import { TextInputDialog } from './TextInputDialog'
import { ConfirmDialog } from './ConfirmDialog'
import { useAnalysisPanelController } from './analysisPanel/useAnalysisPanelController'
import {
  Play,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Users,
  Pause,
  Square,
  Camera,
  Trash2
} from 'lucide-react'

export function buildDirectionsPayload(
  directions: Array<{ direction: string; notes: string; isCustom?: boolean }>,
) {
  return directions.map(d => ({
    direction: d.direction,
    notes: d.notes ?? '',
    is_custom: !!d.isCustom,
  }))
}

export function AnalysisPanel() {
  const {
    agents, annotationData, clearSpeciesInputState, currentSession, expandedRound,
    explorationFailedInfo, explorationReportData, feedbackRoundData, finalizeFromIntegration,
    getAgentStatus, handleChooseDirections, handleConfirmAnnotations, handleDialogSend,
    handleDialogSkip, handleExplorationFailedContinue, handleExplorationFailedGoToData,
    handleExplorationFailedStop, handleIgnoreValidationError, handleLetPiDecide,
    handleLetPiDecideIntegration, handleLlmModifyAgent, handleLlmResumeAfterModify,
    handleLlmRetry, handleLlmReturnToSettings, handleLlmReturnToTimeout, handleLlmSkip,
    handleLlmTerminate, handlePause, handlePlanApprove, handlePlanModify, handlePlanSkip,
    handleReconfigureFromValidation, handleReopenDialog, handleRerun, handleResume,
    handleSnapshotFailureManage, handleSnapshotFailureProceed, handleSnapshotFailureRetry,
    handleSnapshotFailureSkip, handleSnapshotFailureTerminate, handleSpeciesInput, handleStart,
    handleStop, handleStopDialogCancel, handleSupplementaryAnalysis, integrationGateData,
    isCurrentIntegrationGate, isForwardResumeBlocked, isRerunEnabled, isSupplementaryRunning,
    language, llmFailureInfo,
    llmModifySaved, maxRounds, planReviewData, readMode, roundSummaries, roundWaitSeconds,
    runtimeRestoreStatus, runtimeRestoreWarning, setExpandedRound, setShowAnnotationDialog,
    setShowExplorationDialog, setShowIntegrationGateDialog, setShowLlmTimeoutDialog,
    setShowSnapshotCleanupReminder, setShowSnapshotPanel, setShowStopDialog,
    showAnnotationDialog, showExplorationDialog, showExplorationFailedDialog,
    showFeedbackDialog, showIntegrationGateDialog, showLlmTimeoutDialog, showPlanReviewDialog,
    showSnapshotCleanupReminder, showSnapshotPanel, showSpeciesInputDialog, showStopDialog,
    showValidationDialog, snapshotCount, snapshotFailure, snapshotV2Enabled, speciesInputInfo,
    stopping, submitIntegrationDirections, t, timeRemaining, unresolvedErrors, validating,
    validationMessage, validationUncertain,
  } = useAnalysisPanelController()

  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 dark:bg-slate-800 rounded-full mb-6">
          <Users className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="font-display text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
          {t('startNewAnalysis')}
        </h2>
        <p className="text-slate-500 dark:text-slate-400 max-w-md">
          {t('welcomeDesc')}
        </p>
      </div>
    )
  }

  const isPaused = currentSession.status === 'paused'
  const isWaitingInput = currentSession.status === 'waiting_input' || currentSession.status === 'waiting_annotation' || currentSession.status === 'waiting_plan_review' || currentSession.status === 'waiting_species_input'
  const isRunning = currentSession.status === 'running'
  const isCompleted = currentSession.status === 'completed'
  const isFailed = currentSession.status === 'failed'

  return (
    <div className="flex flex-col h-full">
      {/* Validating spinner overlay */}
      {validating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div className="bg-white dark:bg-surface-900 rounded-xl shadow-lg p-6 flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-accent-600 dark:text-accent-400" />
            <span className="text-slate-700 dark:text-slate-200 font-medium">
              {t('validatingApiKey')}
            </span>
          </div>
        </div>
      )}

      {/* Validation failure dialog */}
      {showValidationDialog && (
        <ValidationDialog
          message={validationMessage}
          uncertain={validationUncertain}
          onIgnore={handleIgnoreValidationError}
          onReconfigure={handleReconfigureFromValidation}
          language={language}
        />
      )}

      {/* Runtime restore safety banner (persists across reload).
          Not gated on snapshotV2Enabled: an ordinary session *import* sets
          runtime_restore_status='unavailable' with Snapshot V2 switched off, and
          hiding the banner there left the Resume button disabled with the
          explanation suppressed. */}
      {(runtimeRestoreStatus === 'unavailable' || runtimeRestoreStatus === 'intentionally_kept') && (
        <div className="mb-2">
          <RuntimeWarningBanner
            runtimeRestoreStatus={runtimeRestoreStatus}
            warning={runtimeRestoreWarning}
            isZh={language === 'zh'}
          />
        </div>
      )}

      {/* Header - Compact */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-700">
        <div>
          <h2 className="font-display text-lg font-semibold text-slate-900 dark:text-slate-100">{t('analysisControl')}</h2>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {currentSession.status === 'pending' && (
            <button
              onClick={handleStart}
              disabled={readMode || isForwardResumeBlocked()}
              className={`flex items-center gap-2 px-4 py-1.5 text-white rounded-lg font-medium transition-colors ${
                readMode || isForwardResumeBlocked()
                  ? 'bg-slate-400 dark:bg-slate-600 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
              title={
                isForwardResumeBlocked()
                  ? (language === 'zh' ? '运行时不可用或快照失败，无法启动' : 'Blocked by runtime/snapshot failure')
                  : readMode
                    ? t('readModeRestriction')
                    : undefined
              }
            >
              <Play className="w-4 h-4" />
              {t('startAnalysis')}
            </button>
          )}

          {isRerunEnabled() && !isRunning && (
            <button
              onClick={handleRerun}
              disabled={readMode}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                readMode
                  ? 'bg-slate-200 dark:bg-slate-700 text-slate-400 cursor-not-allowed'
                  : 'bg-indigo-100 dark:bg-indigo-900/60 hover:bg-indigo-200 dark:hover:bg-indigo-800/70 text-indigo-800 dark:text-indigo-200'
              }`}
              title={language === 'zh' ? '从原始数据重新运行' : 'Re-run analysis from raw inputs'}
            >
              <Play className="w-3.5 h-3.5" />
              {language === 'zh' ? '重新运行' : 'Re-run'}
            </button>
          )}
          
          {isRunning && (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2 px-3 py-1 bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300 rounded-lg">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="font-medium">Round {currentSession.currentRound}</span>
              </div>
              <button
                onClick={handlePause}
                className="flex items-center gap-1.5 px-3 py-1 bg-amber-100 dark:bg-amber-900/60 hover:bg-amber-200 dark:hover:bg-amber-800/70 text-amber-700 dark:text-amber-300 rounded-lg text-sm font-medium transition-colors"
              >
                <Pause className="w-4 h-4" />
                {language === 'zh' ? '暂停' : 'Pause'}
              </button>
              <button
                onClick={() => setShowStopDialog(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-red-100 dark:bg-red-900/60 hover:bg-red-200 dark:hover:bg-red-800/70 text-red-700 dark:text-red-300 rounded-lg text-sm font-medium transition-colors"
              >
                <Square className="w-3.5 h-3.5" />
                {language === 'zh' ? '停止' : 'Stop'}
              </button>
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
            </div>
          )}

          {isPaused && (
            <div className="flex items-center gap-2">
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
              <div className="flex items-center gap-2 px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 rounded-lg">
                <Pause className="w-4 h-4" />
                <span className="font-medium">
                  {language === 'zh' ? '已暂停' : 'Paused'}
                </span>
              </div>
              <button
                onClick={handleResume}
                disabled={isForwardResumeBlocked()}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                  isForwardResumeBlocked()
                    ? 'bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
                    : 'bg-green-100 dark:bg-green-900/60 hover:bg-green-200 dark:hover:bg-green-800/70 text-green-700 dark:text-green-300'
                }`}
                title={
                  isForwardResumeBlocked()
                    ? (language === 'zh' ? '运行时不可用，无法继续' : 'Resume blocked — runtime unavailable')
                    : undefined
                }
              >
                <Play className="w-3.5 h-3.5" />
                {language === 'zh' ? '继续' : 'Resume'}
              </button>
            </div>
          )}

          {isWaitingInput && (
            <div className="flex items-center gap-2">
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
              {llmFailureInfo ? (
                <button
                  onClick={() => setShowLlmTimeoutDialog(true)}
                  className="flex items-center gap-2 px-3 py-1 bg-red-100 dark:bg-red-900/60 hover:bg-red-200 dark:hover:bg-red-800/70 text-red-700 dark:text-red-300 rounded-lg transition-colors"
                  aria-label={t('llmCallFailedBadge')}
                >
                  <AlertCircle className="w-4 h-4" aria-hidden="true" />
                  <span className="font-medium">
                    {t('llmCallFailedBadge')}
                  </span>
                </button>
              ) : (
                <button
                  onClick={handleReopenDialog}
                  className={`flex items-center gap-2 px-3 py-1 rounded-lg transition-colors cursor-pointer ${
                    currentSession.status === 'waiting_annotation'
                      ? 'bg-accent-100 dark:bg-accent-900/60 text-accent-700 dark:text-accent-300 hover:bg-accent-200 dark:hover:bg-accent-800/70'
                      : 'bg-amber-100 dark:bg-amber-900/60 text-amber-700 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-800/70'
                  }`}
                  title={language === 'zh' ? '点击重新打开交互窗口' : 'Click to reopen interaction dialog'}
                >
                  <AlertCircle className="w-4 h-4" />
                  <span className="font-medium">
                    {currentSession.status === 'waiting_annotation'
                      ? (language === 'zh' ? '等待注释确认' : 'Waiting for Annotation')
                      : currentSession.status === 'waiting_species_input'
                      ? t('waitingSpeciesInput')
                      : (language === 'zh' ? '等待输入' : 'Waiting for Input')}
                  </span>
                </button>
              )}
              <button
                onClick={() => setShowStopDialog(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-red-100 dark:bg-red-900/60 hover:bg-red-200 dark:hover:bg-red-800/70 text-red-700 dark:text-red-300 rounded-lg text-sm font-medium transition-colors"
              >
                <Square className="w-3.5 h-3.5" />
                {language === 'zh' ? '停止' : 'Stop'}
              </button>
            </div>
          )}

          {isCompleted && (
            <div className="flex items-center gap-2">
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
              <div className="flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900/60 text-green-700 dark:text-green-300 rounded-lg">
                <CheckCircle2 className="w-4 h-4" />
                <span className="font-medium">{t('analysisComplete')}</span>
              </div>
            </div>
          )}

          {isFailed && (
            <div className="flex items-center gap-2">
              {snapshotV2Enabled && <button
                onClick={() => setShowSnapshotPanel(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-lg text-sm font-medium transition-colors"
                title={language === 'zh' ? '快照管理' : 'Snapshots'}
              >
                <Camera className="w-3.5 h-3.5" />
                {language === 'zh' ? '快照' : 'Snapshot'}
              </button>}
              <div className="flex items-center gap-2 px-3 py-1 bg-red-100 dark:bg-red-900/60 text-red-700 dark:text-red-300 rounded-lg">
                <AlertCircle className="w-4 h-4" />
                <span className="font-medium">{language === 'zh' ? '分析失败' : 'Analysis Failed'}</span>
              </div>
            </div>
          )}
        </div>
      </div>

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
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
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
            <div className="prose prose-slate prose-sm max-w-none">
              <ReactMarkdown>{currentSession.finalSummary}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* Annotation Confirmation Dialog — shown during exploration when annotations need review */}
      {showAnnotationDialog && annotationData && currentSession && (
        <AnnotationConfirmationDialog
          annotations={annotationData.annotations || []}
          tissue={annotationData.tissue || ''}
          organism={annotationData.organism || ''}
          tissueSummary={annotationData.tissue_summary || ''}
          reportHtmlUrl={annotationData.report_html_url || undefined}
          detailedReportUrl={annotationData.detailed_report_url || undefined}
          validationStatus={annotationData.validation_status}
          validationSummary={annotationData.validation_summary}
          validationTextSummary={annotationData.validation_text_summary}
          validationResult={annotationData.validation_result}
          reclusteringRecommendations={annotationData.reclustering_recommendations}
          proportionGroupLabels={annotationData.proportion_group_labels || []}
          proportionDimension={annotationData.proportion_dimension || 'group'}
          language={language}
          countdownSeconds={roundWaitSeconds}
          onConfirm={handleConfirmAnnotations}
          onRequestSupplementary={(text) => handleSupplementaryAnalysis(text, 'annotation')}
          isSupplementaryRunning={isSupplementaryRunning}
          onClose={() => setShowAnnotationDialog(false)}
        />
      )}

      {/* Plan Review Dialog — shown when analysis plan needs user approval */}
      {showPlanReviewDialog && planReviewData && currentSession && (
        <PlanReviewDialog
          open={showPlanReviewDialog}
          plan={planReviewData.plan || null}
          dialogueHtml={planReviewData.dialogue_html || ''}
          phase={planReviewData.phase || ''}
          round={planReviewData.round || currentSession.currentRound || 1}
          stage={planReviewData.stage || 'full'}
          countdownSeconds={planReviewData.wait_seconds != null ? planReviewData.wait_seconds : roundWaitSeconds}
          onApprove={handlePlanApprove}
          onSkip={handlePlanSkip}
          onModify={handlePlanModify}
        />
      )}

      {/* Exploration Report Dialog — shown after Round 1 exploration completes */}
      {showExplorationDialog && explorationReportData && currentSession && (
        <ExplorationReportDialog
          report={(explorationReportData.report_a || explorationReportData) as ExplorationReport}
          roundSummary={explorationReportData.round_summary?.summary || ''}
          explorationReportUrl={explorationReportData.html_url || undefined}
          detailedReportUrl={explorationReportData.detailed_report_url || undefined}
          language={language}
          countdownSeconds={roundWaitSeconds}
          onChooseDirections={handleChooseDirections}
          onLetPiDecide={handleLetPiDecide}
          onRequestSupplementary={(text) => handleSupplementaryAnalysis(text, 'direction')}
          isSupplementaryRunning={isSupplementaryRunning}
          onClose={() => setShowExplorationDialog(false)}
        />
      )}

      {isCurrentIntegrationGate && integrationGateData && (
        <IntegrationGatePanel
          cycle={integrationGateData.cycle}
          candidates={integrationGateData.candidates}
          onSubmit={submitIntegrationDirections}
          onFinalize={finalizeFromIntegration}
          onLetPiDecide={handleLetPiDecideIntegration}
          countdownSeconds={roundWaitSeconds}
          language={language}
          reportUrl={integrationGateData.html_url || undefined}
          onClose={() => setShowIntegrationGateDialog(false)}
          open={showIntegrationGateDialog}
        />
      )}

      {/* Round Feedback Dialog — shown after each round completes (not during exploration
          report, and never at the integration gate, which has its own IntegrationGatePanel).
          The integration exclusion is also enforced upstream where showFeedbackDialog is set;
          this render-level guard defends against stale state on reconnect/rehydrate. */}
      {showFeedbackDialog && !showExplorationDialog
        && !(isCurrentIntegrationGate && integrationGateData)
        && feedbackRoundData && feedbackRoundData.phase !== 'integration'
        && currentSession && !llmFailureInfo && (
        <RoundFeedbackDialog
          roundNumber={feedbackRoundData.round}
          maxRounds={maxRounds}
          roundSummary={feedbackRoundData}
          language={language}
          countdownSeconds={roundWaitSeconds}
          treatmentGroups={currentSession.treatmentGroups || []}
          onSend={handleDialogSend}
          onSkip={handleDialogSkip}
          onRequestSupplementary={(text) => handleSupplementaryAnalysis(text, 'round')}
          isSupplementaryRunning={isSupplementaryRunning}
          onStopImmediate={() => handleStop('immediate')}
          onStopSummarize={() => handleStop('summarize')}
        />
      )}

      {/* Stop Confirmation Dialog */}
      {showStopDialog && (
        <StopDialog
          onStopImmediate={() => handleStop('immediate')}
          onStopSummarize={() => handleStop('summarize')}
          onClose={handleStopDialogCancel}
          stopping={stopping}
          language={language}
        />
      )}

      {/* LLM Timeout Dialog */}
      {llmFailureInfo && (
        <LlmTimeoutDialog
          open={showLlmTimeoutDialog}
          llmFailureInfo={llmFailureInfo}
          timeRemaining={timeRemaining}
          hasCountdown={roundWaitSeconds > 0}
          onRetry={handleLlmRetry}
          onModifyAgent={handleLlmModifyAgent}
          onTerminate={handleLlmTerminate}
          onSkip={handleLlmSkip}
          onClose={() => setShowLlmTimeoutDialog(false)}
          language={language}
        />
      )}

      {/* LLM Modify Sub-Dialog */}
      {llmModifySaved && llmFailureInfo && !showLlmTimeoutDialog && (
        <LlmModifySubDialog
          onResume={handleLlmResumeAfterModify}
          onReturnToSettings={handleLlmReturnToSettings}
          onReturnToTimeout={handleLlmReturnToTimeout}
          language={language}
        />
      )}

      {/* Species Input Dialog */}
      {showSpeciesInputDialog && speciesInputInfo && currentSession && (
        <TextInputDialog
          title={speciesInputInfo.source === 'pi_inference'
            ? t('speciesInputTitle')
            : t('speciesGeneValidationTitle')}
          description={speciesInputInfo.context || ''}
          suggestions={speciesInputInfo.suggestions}
          placeholder={speciesInputInfo.source === 'pi_inference'
            ? t('speciesInputPlaceholder')
            : t('speciesGeneValidationPlaceholder')}
          language={language}
          countdownSeconds={roundWaitSeconds}
          resetKey={speciesInputInfo.source}
          onSubmit={handleSpeciesInput}
          onClose={() => clearSpeciesInputState()}
        />
      )}

      {/* Exploration Failed Dialog (likely a data problem) */}
      {showExplorationFailedDialog && explorationFailedInfo && currentSession && (
        <ExplorationFailedDialog
          open={showExplorationFailedDialog}
          info={explorationFailedInfo}
          language={language}
          onGoToData={handleExplorationFailedGoToData}
          onContinue={handleExplorationFailedContinue}
          onStop={handleExplorationFailedStop}
        />
      )}

      {/* Snapshot failure HITL — Retry re-enters capture only; Skip continues
          without capturing at that boundary (skippable producers only) */}
      {(snapshotV2Enabled || (snapshotFailure?.failure.code || '').startsWith('runtime_persistence_')) && snapshotFailure && !showSnapshotPanel && (
        <SnapshotFailureDialog
          failure={snapshotFailure}
          isZh={language === 'zh'}
          onRetry={() => {
            void handleSnapshotFailureRetry()
          }}
          onProceed={() => {
            void handleSnapshotFailureProceed()
          }}
          onSkip={() => {
            void handleSnapshotFailureSkip()
          }}
          onManage={handleSnapshotFailureManage}
          onTerminate={() => {
            void handleSnapshotFailureTerminate()
          }}
        />
      )}

      {/* Snapshot Panel */}
      {snapshotV2Enabled && showSnapshotPanel && currentSession && (
        <SnapshotPanel
          sessionId={currentSession.id}
          language={language}
          onClose={() => setShowSnapshotPanel(false)}
          onRestore={() => {
            setShowSnapshotPanel(false)
            // Reload page to get restored state
            window.location.reload()
          }}
        />
      )}

      {/* Snapshot Cleanup Reminder */}
      {snapshotV2Enabled && <ConfirmDialog
        open={showSnapshotCleanupReminder}
        title={language === 'zh' ? '快照数量提醒' : 'Snapshot Count Reminder'}
        message={language === 'zh'
          ? `当前会话已有 ${snapshotCount} 个快照，建议清理不需要的快照以释放空间。`
          : `This session has ${snapshotCount} snapshots. Consider cleaning up unused snapshots to free space.`}
        confirmLabel={language === 'zh' ? '去清理' : 'Cleanup'}
        titleIcon={<Camera className="w-5 h-5 text-accent-600" />}
        confirmIcon={<Trash2 className="w-4 h-4" />}
        onConfirm={() => {
          setShowSnapshotCleanupReminder(false)
          setShowSnapshotPanel(true)
        }}
        onCancel={() => setShowSnapshotCleanupReminder(false)}
      />}
    </div>
  )
}
