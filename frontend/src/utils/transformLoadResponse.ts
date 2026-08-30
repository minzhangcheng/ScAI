import { parseSnapshotFailure, type ParsedSnapshotFailure } from './snapshotV2'
import { buildAnnotationDataFromState, type AnnotationData } from './annotationGate'
import type {
  SessionLoadResponse,
  AgentSettingsResponse,
  RawRoundResult,
  RawTreatmentGroup,
  RawDataFile,
  RawBiochemistryIndicator,
  RawPublicationFigure,
  RawPublicationTable,
} from '../types'
import { TASK_STATUSES } from '../stores/appStore'
import type {
  Session,
  RoundResult,
  FigureReviewStatus,
  LlmFailureInfo,
  AgentModels,
  AgentPrompts,
  AgentServerConfig,
  AgentTemperatures,
  AgentType,
  AgentVisionSettings,
} from '../stores/appStore'
import type { SnapshotTransactionRecord } from '../types/snapshotV2'
import { resultIdentityKey } from './roundKey'
import { normalizeImageBase64 } from './image'

function opt<K extends string, V>(key: K, value: V | null | undefined): Partial<Record<K, V>> {
  return value != null ? { [key]: value } as Partial<Record<K, V>> : {}
}

function hasInterventionTarget(
  value: unknown,
): value is Record<string, unknown> & { node_name: string; generation: number } {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Record<string, unknown>
  return (
    typeof candidate.node_name === 'string' &&
    typeof candidate.generation === 'number'
  )
}

export function transformRoundResults(raw: (RawRoundResult | null)[] | undefined): RoundResult[] {
  return (raw || []).map((r): RoundResult | null => {
    if (!r) return null
    const round = r.round ?? 0
    const phase: RoundResult['phase'] = (
      r.phase === 'innovation' || r.phase === 'integration' || r.phase === 'exploration'
        ? r.phase
        : 'exploration'
    )
    const lineageId = r.lineage_id || 'main'
    const directionKey = r.direction_key || r.direction || ''
    return {
      round,
      summary: r.pi_summary || r.summary || '',
      phase,
      pathologyFindings: Array.isArray(r.pathology_findings) ? r.pathology_findings : [],
      pathologyCoverage: Array.isArray(r.pathology_coverage) ? r.pathology_coverage : [],
      bioinformaticsFindings: Array.isArray(r.bioinformatics_findings) ? r.bioinformatics_findings : [],
      biochemistryFindings: Array.isArray(r.biochemistry_findings) ? r.biochemistry_findings : [],
      literatureFindings: Array.isArray(r.literature_findings) ? r.literature_findings : [],
      crossModalCandidates: Array.isArray(r.cross_modal_candidates) ? r.cross_modal_candidates : [],
      integrationCandidates: Array.isArray(r.integration_candidates) ? r.integration_candidates : [],
      roundStatus: r.round_status || 'completed',
      evidenceStatus: r.evidence_status || 'empty',
      ...(r.direction_coverage_eligible != null
        ? { directionCoverageEligible: r.direction_coverage_eligible }
        : {}),
      ...(r.bioinformatics_evidence_verified != null
        ? { bioinformaticsEvidenceVerified: r.bioinformatics_evidence_verified }
        : {}),
      evidenceExecutionIds: Array.isArray(r.evidence_execution_ids) ? r.evidence_execution_ids : [],
      evidenceNotes: Array.isArray(r.evidence_notes) ? r.evidence_notes : [],
      evidenceGateFindings: Array.isArray(r.evidence_gate_findings) ? r.evidence_gate_findings : [],
      evidenceGateDegraded: r.evidence_gate_degraded === true,
      ...(r.wait_policy ? { waitPolicy: r.wait_policy } : {}),
      lineageId,
      ...(r.branch_id ? { branchId: r.branch_id } : {}),
      ...(r.branch_label ? { branchLabel: r.branch_label } : {}),
      ...(r.restored_from_snapshot_id
        ? { restoredFromSnapshotId: r.restored_from_snapshot_id }
        : {}),
      directionKey,
      resultIdentityKey: resultIdentityKey({
        result_identity_key: r.result_identity_key,
        lineageId,
        phase,
        round,
        directionKey,
      }),
      ...(r.integration_report_summary ? { integrationReportSummary: r.integration_report_summary } : {}),
    }
  }).filter((r): r is RoundResult => r !== null)
}

function transformTreatmentGroups(raw: RawTreatmentGroup[] | undefined) {
  return (raw || []).map((g) => ({
    id: g.id, name: g.name, description: g.description || '',
    sampleCount: g.sampleCount ?? g.sample_count ?? 1, order: g.order ?? 0,
  }))
}

function transformDataFiles(raw: RawDataFile[] | undefined) {
  return (raw || []).map((f) => ({
    id: f.id, filename: f.filename, filePath: f.filePath ?? f.file_path ?? '',
    section: f.section, dataType: f.dataType ?? f.data_type ?? '',
    groupId: f.groupId ?? f.group_id, sampleNumber: f.sampleNumber ?? f.sample_number,
    description: f.description, timestamp: f.timestamp || '',
  }))
}

function transformBiochemistryIndicators(raw: RawBiochemistryIndicator[] | undefined) {
  return (raw || []).map((r) => ({
    id: r.id, indicator: r.indicator, value: r.value, unit: r.unit,
    referenceRange: r.referenceRange ?? r.reference_range,
    interpretation: r.interpretation,
    groupId: r.groupId ?? r.group_id, sampleNumber: r.sampleNumber ?? r.sample_number,
  }))
}

/**
 * Gallery badge fields carried by the load payload
 * (backend AnalysisState._serialize_publication_figures) and by the
 * integrated_report message. Declared here rather than on RawPublicationFigure
 * because they are optional on every other producer of a raw figure.
 */
export type RawFigureBadges = {
  round?: number | null
  round_number?: number | null
  layout_blocking?: boolean | null
  layoutBlocking?: boolean | null
  diagnostic?: boolean | null
}

function transformPublicationFigures(raw: (RawPublicationFigure & RawFigureBadges)[] | undefined) {
  return (raw || []).map((f) => ({
    id: f.id,
    name: f.name,
    description: f.description,
    figureType: f.figure_type || f.figureType || '',
    pngPath: f.png_path || f.pngPath || '',
    pdfPath: f.pdf_path || f.pdfPath || '',
    pngBase64: normalizeImageBase64(f.png_base64)
      ?? normalizeImageBase64(f.pngBase64),
    reviewStatus: (f.review_status || f.reviewStatus || 'pending') as FigureReviewStatus,
    reviewUnavailable: f.review_unavailable ?? f.reviewUnavailable ?? false,
    piFeedback: f.pi_feedback || f.piFeedback,
    revisionCount: f.revision_count ?? f.revisionCount ?? 0,
    createdAt: f.created_at || f.createdAt || '',
    // Same mapping ResultsPanel applies to a live figure message: a reload must
    // rebuild the round / layout-defect / diagnostic badges, not drop them and
    // wait for an integrated_report replay that may have aged out of history.
    round: f.round ?? f.round_number ?? undefined,
    layoutBlocking: f.layout_blocking ?? f.layoutBlocking ?? false,
    diagnostic: f.diagnostic ?? false,
    planStepId: f.plan_step_id ?? f.planStepId ?? undefined,
    planId: f.plan_id ?? f.planId ?? undefined,
    planRequirementKey: f.plan_requirement_key ?? f.planRequirementKey ?? undefined,
    submittedPlanStepId: f.submitted_plan_step_id ?? f.submittedPlanStepId ?? undefined,
    supervisorBindingDecision: f.supervisor_binding_decision ?? f.supervisorBindingDecision ?? undefined,
    supervisorBindingReason: f.supervisor_binding_reason ?? f.supervisorBindingReason ?? undefined,
    sourceExecutionId: f.source_execution_id ?? f.sourceExecutionId ?? undefined,
    purpose: f.purpose ?? undefined,
    isPublication: f.is_publication ?? f.isPublication ?? true,
  }))
}

function transformPublicationTables(raw: RawPublicationTable[] | undefined) {
  return (raw || []).map((t) => ({
    id: t.id,
    name: t.name,
    description: t.description,
    csvPath: t.csv_path || t.csvPath || '',
    columns: Array.isArray(t.columns) ? t.columns : [],
    previewData: Array.isArray(t.preview_data) ? t.preview_data : (t.previewData || []),
    reviewStatus: (t.review_status || t.reviewStatus || 'pending') as FigureReviewStatus,
    piFeedback: t.pi_feedback || t.piFeedback,
    createdAt: t.created_at || t.createdAt || '',
  }))
}

function transformGraphicAbstract(raw: SessionLoadResponse['graphic_abstract']) {
  if (!raw) return undefined
  return {
    id: raw.id,
    prompt: raw.prompt || '',
    imageBase64: normalizeImageBase64(raw.image_base64)
      ?? normalizeImageBase64(raw.imageBase64),
    textDiagram: raw.text_diagram || raw.textDiagram || undefined,
    isTextMode: raw.is_text_mode ?? raw.isTextMode ?? false,
    createdAt: raw.created_at || raw.createdAt || '',
  }
}

export interface LiteratureConfigStoreUpdates {
  literatureMode?: string
  deepResearchModel?: string
  deepResearchLlmBaseUrl?: string
  deepResearchApiKey?: string
  googleApiKey?: string
  pubmedEmail?: string
  literatureMaxSearchRounds?: number
}

export type StoreUpdates = {
  stateVersion?: number
  snapshotCount?: number
  roundWaitSeconds?: number
  /** Rebuilt from the authoritative annotation_state; null when no gate pending. */
  annotationGateData?: AnnotationData | null
  /** Pending LLM/disabled-agent intervention rebuilt from authoritative state. */
  llmFailureInfo?: LlmFailureInfo | null
  runtimeRestoreStatus?: 'complete' | 'unavailable' | 'intentionally_kept'
  runtimeRestoreWarning?: string | null
  snapshotFailure?: ParsedSnapshotFailure | null
  snapshotTransaction?: SnapshotTransactionRecord | null
  prompts?: AgentPrompts
  models?: AgentModels
  temperatures?: AgentTemperatures
  agentServerConfigs?: Partial<Record<AgentType, AgentServerConfig>>
  agentVisionSettings?: AgentVisionSettings
} & LiteratureConfigStoreUpdates

export interface TransformedLoadResponse {
  sessionUpdates: Partial<Session>
  storeUpdates: StoreUpdates
}

export function transformAgentSettings(data: AgentSettingsResponse): StoreUpdates {
  const agentServerConfigs = Object.fromEntries(
    Object.entries(data.agent_server_configs).map(([agent, config]) => [
      agent,
      {
        llmBaseUrl: config?.llm_base_url ?? '',
        apiKey: config?.api_key ?? '',
        ...(typeof config?.context_window === 'number'
          ? { contextWindow: config.context_window }
          : {}),
        ...(typeof config?.effective_context_window === 'number'
          ? { effectiveContextWindow: config.effective_context_window }
          : {}),
      },
    ]),
  ) as Partial<Record<AgentType, AgentServerConfig>>
  const literature = data.literature_config

  return {
    prompts: data.prompts,
    models: data.models,
    temperatures: data.temperatures,
    agentServerConfigs,
    agentVisionSettings: data.vision_settings,
    ...opt('literatureMode', literature.mode),
    ...opt('deepResearchModel', literature.deep_research_model),
    ...opt('deepResearchLlmBaseUrl', literature.deep_research_llm_base_url),
    ...opt('deepResearchApiKey', literature.deep_research_api_key),
    ...opt('googleApiKey', literature.google_api_key),
    ...opt('pubmedEmail', literature.pubmed_email),
    ...opt('literatureMaxSearchRounds', literature.max_search_rounds),
  }
}

export function transformLoadResponse(data: SessionLoadResponse): TransformedLoadResponse {
  const status: Session['status'] = TASK_STATUSES.includes(data.status as Session['status'])
    ? data.status as Session['status']
    : 'pending'

  const sessionUpdates: Partial<Session> = {
    ...opt('name', data.name),
    status,
    ...opt('currentRound', data.current_round),
    ...opt('instruction', data.initial_instruction),
    ...opt('treatmentGroups', data.treatment_groups != null ? transformTreatmentGroups(data.treatment_groups) : undefined),
    ...opt('dataFiles', data.data_files != null ? transformDataFiles(data.data_files) : undefined),
    ...opt('biochemistryIndicators', data.biochemistry_indicators != null ? transformBiochemistryIndicators(data.biochemistry_indicators) : undefined),
    ...opt('roundResults', data.round_results != null ? transformRoundResults(data.round_results) : undefined),
    ...opt('finalSummary', data.final_summary),
    ...opt('publicationFigures', data.publication_figures != null ? transformPublicationFigures(data.publication_figures) : undefined),
    ...opt('publicationTables', data.publication_tables != null ? transformPublicationTables(data.publication_tables) : undefined),
    ...opt('figureRevisionTasks', data.figure_revision_tasks != null ? data.figure_revision_tasks.map((task) => ({
      id: task.id,
      logicalFigureId: task.logical_figure_id,
      planId: task.plan_id ?? undefined,
      planRequirementKey: task.plan_requirement_key ?? undefined,
      currentFigureId: task.current_figure_id,
      sourceExecutionId: task.source_execution_id ?? undefined,
      status: task.status,
      reason: task.reason,
      mustFix: task.must_fix,
      suggestions: task.suggestions,
      attemptCount: task.attempt_count,
      maxAttempts: task.max_attempts,
      createdAt: task.created_at,
      updatedAt: task.updated_at,
    })) : undefined),
    ...opt('bulkInputProfiles', data.bulk_input_profiles != null ? data.bulk_input_profiles.map((profile) => ({
      id: profile.id,
      sourceFileId: profile.source_file_id,
      shape: profile.shape,
      annotationColumns: profile.annotation_columns,
      candidateSampleColumns: profile.candidate_sample_columns,
      numericFractionByColumn: profile.numeric_fraction_by_column,
      integerLikeFraction: profile.integer_like_fraction,
      negativeValueCount: profile.negative_value_count,
      missingValueSummary: profile.missing_value_summary,
      declaredScale: profile.declared_scale,
      profileStatus: profile.profile_status,
      complete: profile.complete,
      rowsScanned: profile.rows_scanned,
      errorCode: profile.error_code ?? undefined,
      createdAt: profile.created_at,
    })) : undefined),
    ...opt('methodDecisions', data.method_decisions != null ? data.method_decisions.map((decision) => ({
      id: decision.id,
      inputProfileId: decision.input_profile_id,
      claimLevel: decision.claim_level,
      method: decision.method,
      rationale: decision.rationale,
      executionId: decision.execution_id,
      decisionSource: decision.decision_source,
      roundNumber: decision.round_number,
      createdAt: decision.created_at,
    })) : undefined),
    ...opt('graphicAbstract', data.graphic_abstract != null ? transformGraphicAbstract(data.graphic_abstract) : undefined),
    ...opt('integratedReport', data.integrated_report ?? undefined),
    ...opt('explorationReportUrl', data.exploration_report_url),
    ...opt('integrationReportUrl', data.integration_report_url),
    ...opt('restoreBranchId', data.restore_branch_id ?? undefined),
    ...opt(
      'runtimePersistence',
      data.runtime_persistence && typeof data.runtime_persistence === 'object'
        ? data.runtime_persistence
        : undefined,
    ),
    ...opt('analysisPhase', data.analysis_phase),
    ...opt('species', data.species),
    ...opt('speciesResolved', data.species_resolved),
    ...opt('speciesGenesValidated', data.species_genes_validated),
  }

  const storeUpdates: TransformedLoadResponse['storeUpdates'] = {
    ...opt('stateVersion', data.state_version),
    ...opt('snapshotCount', data.snapshot_count),
    ...opt('roundWaitSeconds', typeof data.round_wait_seconds === 'number' ? data.round_wait_seconds : undefined),
    annotationGateData: buildAnnotationDataFromState(data.annotation_state, {
      ...opt('annotation_report_url', data.annotation_report_url),
      ...opt('detailed_report_url', data.detailed_report_url),
    }),
    llmFailureInfo: null,
  }

  if (
    status === 'waiting_input' &&
    (data.wait_input_reason === 'llm_failure' || data.wait_input_reason === 'marker_db_missing') &&
    hasInterventionTarget(data.llm_failure_info)
  ) {
    const info = data.llm_failure_info
    storeUpdates.llmFailureInfo = {
      agent: typeof info.agent === 'string' ? info.agent : '',
      model: typeof info.model === 'string' ? info.model : '',
      llmServer: typeof info.llm_server === 'string' ? info.llm_server : '',
      error: typeof info.error === 'string' ? info.error : '',
      nodeName: info.node_name,
      generation: info.generation,
      reason: data.wait_input_reason,
    }
  } else if (
    status === 'waiting_input' &&
    data.wait_input_reason === 'agent_disabled' &&
    hasInterventionTarget(data.agent_disabled_info)
  ) {
    const info = data.agent_disabled_info
    const agent = typeof info.agent === 'string' ? info.agent : ''
    storeUpdates.llmFailureInfo = {
      agent,
      model: '',
      llmServer: '',
      error: `Agent "${agent}" is disabled but required by step "${info.node_name}".`,
      nodeName: info.node_name,
      generation: info.generation,
      reason: 'agent_disabled',
    }
  }

  if (
    data.runtime_restore_status === 'complete' ||
    data.runtime_restore_status === 'unavailable' ||
    data.runtime_restore_status === 'intentionally_kept'
  ) {
    storeUpdates.runtimeRestoreStatus = data.runtime_restore_status
  }
  if (data.runtime_restore_warning != null) {
    storeUpdates.runtimeRestoreWarning = data.runtime_restore_warning
  } else if (data.runtime_restore_status) {
    storeUpdates.runtimeRestoreWarning = null
  }

  const parsed = parseSnapshotFailure(data.snapshot_failure)
  if (parsed) {
    storeUpdates.snapshotFailure = parsed
  }

  if ('snapshot_transaction' in data) {
    storeUpdates.snapshotTransaction = data.snapshot_transaction ?? null
  }

  if (data.literature_config && typeof data.literature_config === 'object') {
    const lc = data.literature_config as Record<string, unknown>
    Object.assign(storeUpdates, {
      ...opt('literatureMode', typeof lc.mode === 'string' ? lc.mode : undefined),
      ...opt('deepResearchModel', typeof lc.deep_research_model === 'string' ? lc.deep_research_model : undefined),
      ...opt('deepResearchLlmBaseUrl', typeof lc.deep_research_llm_base_url === 'string' ? lc.deep_research_llm_base_url : undefined),
      ...opt('deepResearchApiKey', typeof lc.deep_research_api_key === 'string' ? lc.deep_research_api_key : undefined),
      ...opt('googleApiKey', typeof lc.google_api_key === 'string' ? lc.google_api_key : undefined),
      ...opt('pubmedEmail', typeof lc.pubmed_email === 'string' ? lc.pubmed_email : undefined),
      ...opt('literatureMaxSearchRounds', typeof lc.max_search_rounds === 'number' ? lc.max_search_rounds : undefined),
    })
  }

  return { sessionUpdates, storeUpdates }
}
