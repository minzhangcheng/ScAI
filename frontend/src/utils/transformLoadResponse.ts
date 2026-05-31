import type {
  SessionLoadResponse,
  RawRoundResult,
  RawTreatmentGroup,
  RawDataFile,
  RawBiochemistryIndicator,
  RawPublicationFigure,
  RawPublicationTable,
} from '../types'
import type { Session, RoundResult, FigureReviewStatus } from '../stores/appStore'

const VALID_STATUSES: readonly Session['status'][] = [
  'pending', 'running', 'paused', 'waiting_input', 'waiting_annotation',
  'waiting_plan_review', 'completed', 'failed', 'generating_report',
]

function opt<K extends string, V>(key: K, value: V | null | undefined): Partial<Record<K, V>> {
  return value != null ? { [key]: value } as Partial<Record<K, V>> : {}
}

export function transformRoundResults(raw: (RawRoundResult | null)[] | undefined): RoundResult[] {
  return (raw || []).map((r) => {
    if (!r) return null
    return {
      round: r.round ?? 0,
      summary: r.pi_summary || r.summary || '',
      pathologyFindings: Array.isArray(r.pathology_findings) ? r.pathology_findings : [],
      bioinformaticsFindings: Array.isArray(r.bioinformatics_findings) ? r.bioinformatics_findings : [],
      biochemistryFindings: Array.isArray(r.biochemistry_findings) ? r.biochemistry_findings : [],
      literatureFindings: Array.isArray(r.literature_findings) ? r.literature_findings : [],
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

function transformPublicationFigures(raw: RawPublicationFigure[] | undefined) {
  return (raw || []).map((f) => ({
    id: f.id,
    name: f.name,
    description: f.description,
    figureType: f.figure_type || f.figureType || '',
    pngPath: f.png_path || f.pngPath || '',
    pdfPath: f.pdf_path || f.pdfPath || '',
    pngBase64: f.png_base64 || f.pngBase64,
    reviewStatus: (f.review_status || f.reviewStatus || 'pending') as FigureReviewStatus,
    piFeedback: f.pi_feedback || f.piFeedback,
    revisionCount: f.revision_count ?? f.revisionCount ?? 0,
    createdAt: f.created_at || f.createdAt || '',
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
    imageBase64: raw.image_base64 || raw.imageBase64 || '',
    textDiagram: raw.text_diagram || raw.textDiagram || undefined,
    isTextMode: raw.is_text_mode ?? raw.isTextMode ?? false,
    createdAt: raw.created_at || raw.createdAt || '',
  }
}

export interface TransformedLoadResponse {
  sessionUpdates: Partial<Session>
  storeUpdates: {
    stateVersion?: number
    snapshotCount?: number
    roundWaitSeconds?: number
  }
}

export function transformLoadResponse(data: SessionLoadResponse): TransformedLoadResponse {
  const status: Session['status'] = VALID_STATUSES.includes(data.status as Session['status'])
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
    ...opt('graphicAbstract', data.graphic_abstract != null ? transformGraphicAbstract(data.graphic_abstract) : undefined),
    ...opt('integratedReport', data.integrated_report ?? undefined),
    ...opt('explorationReportUrl', data.exploration_report_url),
    ...opt('analysisPhase', data.analysis_phase),
  }

  const storeUpdates: TransformedLoadResponse['storeUpdates'] = {
    ...opt('stateVersion', data.state_version),
    ...opt('snapshotCount', data.snapshot_count),
    ...opt('roundWaitSeconds', typeof data.round_wait_seconds === 'number' ? data.round_wait_seconds : undefined),
  }

  return { sessionUpdates, storeUpdates }
}
