import type { RoundSummary } from '../../types'
import type { AnnotationData } from '../../utils/annotationGate'
import type { AnalysisPlan } from '../PlanReviewDialog'
import type { ExplorationReport } from '../ExplorationReportDialog'

export interface PlanReviewData {
  plan?: AnalysisPlan | null
  dialogue_html?: string
  phase?: string
  round?: number
  stage?: string
  wait_seconds?: number
}

export interface ExplorationReportData extends Partial<ExplorationReport> {
  report_a?: ExplorationReport
  round_summary?: { summary?: string }
  html_url?: string
  detailed_report_url?: string
}

export interface RoundSummaryContent {
  report_segment_id?: string | null
  round?: number
  max_rounds?: number
  summary?: string
  findings?: string[]
  data_requests?: { pathology?: string[]; biochemistry?: string[]; other?: string[] }
  status?: string
  pathology_findings?: string[]
  bioinformatics_findings?: string[]
  biochemistry_findings?: string[]
  literature_findings?: string[]
  phase?: string
  integration_candidates?: {
    direction: string
    score?: number
    notes?: string
    [key: string]: unknown
  }[]
  integration_report_summary?: string
  lineage_id?: string
  direction?: string
  direction_key?: string
  result_identity_key?: string
}

export interface AnalysisPanelDialogState {
  annotationData: AnnotationData | null
  feedbackRoundData: RoundSummary | null
  explorationReportData: ExplorationReportData | null
  integrationGateData: any
  planReviewData: PlanReviewData | null
}
