import { AnnotationConfirmationDialog } from '../AnnotationConfirmationDialog'
import ExplorationFailedDialog from '../ExplorationFailedDialog'
import {
  ExplorationReportDialog,
  type ExplorationReport,
} from '../ExplorationReportDialog'
import { IntegrationGatePanel } from '../IntegrationGatePanel'
import LlmModifySubDialog from '../LlmModifySubDialog'
import LlmTimeoutDialog from '../LlmTimeoutDialog'
import { PlanReviewDialog } from '../PlanReviewDialog'
import { RoundFeedbackDialog } from '../RoundFeedbackDialog'
import { StopDialog } from '../StopDialog'
import { TextInputDialog } from '../TextInputDialog'
import type { AnalysisPanelController } from './useAnalysisPanelController'
import { withReportBranch } from '../../utils/reportBranch'

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
    handlePlanApprove, handlePlanModify, handlePlanSkip,
    handleSpeciesInput, handleStop, handleStopDialogCancel,
    handleSupplementaryAnalysis, integrationGateData, isCurrentIntegrationGate,
    isSupplementaryRunning, language, llmFailureInfo, llmModifySaved, llmWaitSeconds, maxRounds,
    planReviewData, roundWaitSeconds, setShowAnnotationDialog, setShowExplorationDialog,
    setShowIntegrationGateDialog, setShowLlmTimeoutDialog,
    showAnnotationDialog, showExplorationDialog,
    showExplorationFailedDialog, showFeedbackDialog, showIntegrationGateDialog,
    showLlmTimeoutDialog, showPlanReviewDialog,
    showSpeciesInputDialog, showStopDialog,
    speciesInputInfo, stopping, submitIntegrationDirections,
    t, timeRemaining,
  } = controller

  return (
    <>
      {/* Annotation Confirmation Dialog — shown during exploration when annotations need review */}
      {showAnnotationDialog && annotationData && currentSession && (
        <AnnotationConfirmationDialog
          annotations={annotationData.annotations || []}
          tissue={annotationData.tissue || ''}
          organism={annotationData.organism || ''}
          tissueSummary={annotationData.tissue_summary || ''}
          reportHtmlUrl={withReportBranch(annotationData.report_html_url || '', currentSession.restoreBranchId) || undefined}
          detailedReportUrl={withReportBranch(annotationData.detailed_report_url || '', currentSession.restoreBranchId) || undefined}
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
          explorationReportUrl={withReportBranch(explorationReportData.html_url || '', currentSession.restoreBranchId) || undefined}
          detailedReportUrl={withReportBranch(explorationReportData.detailed_report_url || '', currentSession.restoreBranchId) || undefined}
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
          reportUrl={withReportBranch(integrationGateData.html_url || '', currentSession?.restoreBranchId) || undefined}
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
          hasCountdown={llmWaitSeconds > 0}
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

    </>
  )
}
