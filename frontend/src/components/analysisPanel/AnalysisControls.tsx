import { AlertCircle, CheckCircle2, Loader2, Pause, Play, Square } from 'lucide-react'

import { ValidationDialog } from '../ValidationDialog'
import { RuntimePersistenceBanner } from '../runtime/RuntimePersistenceBanner'
import { RuntimeWarningBanner } from '../runtime/RuntimeWarningBanner'
import { SnapshotList } from '../runtime/SnapshotList'
import { chatOwnerKey } from '../../utils/chatOwnership'
import type { AnalysisPanelController } from './useAnalysisPanelController'

interface AnalysisControlsProps {
  controller: AnalysisPanelController
}

export function AnalysisControls({ controller }: AnalysisControlsProps) {
  const {
    currentSession, gateContinuationPending,
    handleIgnoreValidationError, handlePause, handleReconfigureFromValidation,
    handleReopenDialog, handleRerun, handleResume, handleStart, isForwardResumeBlocked,
    isRerunEnabled, language, llmFailureInfo, readMode, runtimeRestoreStatus,
    runtimeRestoreWarning, setShowLlmTimeoutDialog, setShowStopDialog,
    showValidationDialog, t, validating, validationMessage,
    validationUncertain,
  } = controller
  if (!currentSession) return null
  const failureBadgeKey = llmFailureInfo?.failureKind === 'annotation_tool_failure'
    ? 'annotationToolFailed'
    : llmFailureInfo?.failureKind === 'parse_error'
      ? 'llmOutputInvalid' : 'llmCallFailedBadge'
  const isPaused = currentSession.status === 'paused'
  const isWaitingInput = currentSession.status === 'waiting_input' || currentSession.status === 'waiting_annotation' || currentSession.status === 'waiting_plan_review' || currentSession.status === 'waiting_species_input'
  const isRunning = currentSession.status === 'running'
  const isCompleted = currentSession.status === 'completed'
  const isFailed = currentSession.status === 'failed'
  const currentOwnerKey = chatOwnerKey(
    currentSession.id, currentSession.restoreBranchId,
  )
  const gatePending = Boolean(
    gateContinuationPending
    && gateContinuationPending.ownerKey === currentOwnerKey,
  )

  return (
    <>
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

      {/* Runtime safety banner (persists across reload). A session imported
          without its runtime, or one whose runtime could not be rehydrated,
          sets runtime_restore_status='unavailable'; hiding the banner would
          leave the Resume button disabled with the explanation suppressed. */}
      {(runtimeRestoreStatus === 'unavailable' || runtimeRestoreStatus === 'intentionally_kept') && (
        <div className="mb-2">
          <RuntimeWarningBanner
            runtimeRestoreStatus={runtimeRestoreStatus}
            warning={runtimeRestoreWarning}
            isZh={language === 'zh'}
          />
        </div>
      )}

      {/* What the last committed generation promises (spec 5.1). Non-blocking:
          the capture gate that used to stop the analysis here is gone, and
          there is nothing a dialog could offer — the previous generation is
          intact and the run continues either way.

          The status goes in because the same record means different things on
          either side of it: while the run is live, "the state is newer than
          the newest committed runtime" is the expected state between successful
          save points and stays silent; failed saves and non-live gaps remain
          visible. */}
      <div className="mb-2 empty:mb-0">
        <RuntimePersistenceBanner
          record={currentSession.runtimePersistence as never}
          sessionStatus={currentSession.status}
          isZh={language === 'zh'}
        />
      </div>

      {/* Decision snapshots (Part B spec 4.9). Absent unless the administrator
          configured a snapshot directory. The summary remains expandable at
          zero so capture failures and the not-yet-captured state stay visible. */}
      <SnapshotList sessionId={currentSession.id} isZh={language === 'zh'} />

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
                  ? (language === 'zh' ? '运行时不可用，无法启动' : 'Blocked: the runtime is unavailable')
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
              title={language === 'zh'
                ? '回到最近一次保存的内核状态并继续；没有可用状态时才从原始数据重跑'
                : 'Continue from the last saved runtime; re-runs from the raw inputs only if none can be used'}
            >
              <Play className="w-3.5 h-3.5" />
              {language === 'zh' ? '恢复并继续' : 'Recover'}
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
            </div>
          )}

          {isPaused && (
            <div className="flex items-center gap-2">
              {gatePending ? (
                <div className="flex items-center gap-2 px-3 py-1 bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300 rounded-lg">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="font-medium">
                    {language === 'zh' ? '正在重新启动…' : 'Restarting…'}
                  </span>
                </div>
              ) : (
                <>
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
                </>
              )}
            </div>
          )}

          {isWaitingInput && (
            <div className="flex items-center gap-2">
              {llmFailureInfo ? (
                <button
                  onClick={() => setShowLlmTimeoutDialog(true)}
                  className="flex items-center gap-2 px-3 py-1 bg-red-100 dark:bg-red-900/60 hover:bg-red-200 dark:hover:bg-red-800/70 text-red-700 dark:text-red-300 rounded-lg transition-colors"
                  aria-label={t(failureBadgeKey)}
                >
                  <AlertCircle className="w-4 h-4" aria-hidden="true" />
                  <span className="font-medium">
                    {t(failureBadgeKey)}
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
              <div className="flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900/60 text-green-700 dark:text-green-300 rounded-lg">
                <CheckCircle2 className="w-4 h-4" />
                <span className="font-medium">{t('analysisComplete')}</span>
              </div>
            </div>
          )}

          {isFailed && (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2 px-3 py-1 bg-red-100 dark:bg-red-900/60 text-red-700 dark:text-red-300 rounded-lg">
                <AlertCircle className="w-4 h-4" />
                <span className="font-medium">{language === 'zh' ? '分析失败' : 'Analysis Failed'}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
