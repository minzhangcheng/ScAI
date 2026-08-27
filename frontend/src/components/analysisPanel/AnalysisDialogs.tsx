import { Camera, Trash2 } from 'lucide-react'

import { AnnotationConfirmationDialog } from '../AnnotationConfirmationDialog'
import { ConfirmDialog } from '../ConfirmDialog'
import ExplorationFailedDialog from '../ExplorationFailedDialog'
import { ExplorationReportDialog } from '../ExplorationReportDialog'
import { IntegrationGatePanel } from '../IntegrationGatePanel'
import LlmModifySubDialog from '../LlmModifySubDialog'
import LlmTimeoutDialog from '../LlmTimeoutDialog'
import { PlanReviewDialog } from '../PlanReviewDialog'
import { RoundFeedbackDialog } from '../RoundFeedbackDialog'
import { SnapshotPanel } from '../SnapshotPanel'
import { SnapshotFailureDialog } from '../snapshots/SnapshotFailureDialog'
import { StopDialog } from '../StopDialog'
import { TextInputDialog } from '../TextInputDialog'
import type { AnalysisPanelController } from './useAnalysisPanelController'

interface AnalysisDialogsProps {
  controller: AnalysisPanelController
}

export function AnalysisDialogs({ controller }: AnalysisDialogsProps) {
  const {
    annotationData, clearSpeciesInputState, currentSession, explorationFailedInfo,
    explorationReportData, feedbackRoundData, finalizeFromIntegration,
    handleChooseDirections, handleConfirmAnnotations, handleDialogSend, handleDialogSkip,
    handleExplorationFailedContinue, handleExplorationFailedGoToData,
    handleExplorationFailedStop, handleLetPiDecide, handleLetPiDecideIntegration,
    handleLlmModifyAgent, handleLlmResumeAfterModify, handleLlmRetry,
    handleLlmReturnToSettings, handleLlmReturnToTimeout, handleLlmSkip, handleLlmTerminate,
    handlePlanApprove, handlePlanModify, handlePlanSkip, handleSnapshotFailureManage,
    handleSnapshotFailureProceed, handleSnapshotFailureRetry, handleSnapshotFailureSkip,
    handleSnapshotFailureTerminate, handleSpeciesInput, handleStop, handleStopDialogCancel,
    handleSupplementaryAnalysis, integrationGateData, isCurrentIntegrationGate,
    isSupplementaryRunning, language, llmFailureInfo, llmModifySaved, maxRounds,
    planReviewData, roundWaitSeconds, setShowAnnotationDialog, setShowExplorationDialog,
    setShowIntegrationGateDialog, setShowLlmTimeoutDialog, setShowSnapshotCleanupReminder,
    setShowSnapshotPanel, showAnnotationDialog, showExplorationDialog,
    showExplorationFailedDialog, showFeedbackDialog, showIntegrationGateDialog,
    showLlmTimeoutDialog, showPlanReviewDialog, showSnapshotCleanupReminder,
    showSnapshotPanel, showSpeciesInputDialog, showStopDialog, snapshotCount,
    snapshotFailure, snapshotV2Enabled, speciesInputInfo, stopping, submitIntegrationDirections,
    timeRemaining,
  } = controller
  const isRunning = currentSession?.status === 'running'

  return (
    <>
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
    </>
  )
}
