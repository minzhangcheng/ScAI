import type { RawAnnotationState, RawClusterAnnotation } from '../types'
import type { ClusterAnnotation } from '../components/AnnotationConfirmationDialog'

/** Payload the annotation confirmation dialog renders. */
export interface AnnotationData {
  annotations?: ClusterAnnotation[]
  tissue?: string
  organism?: string
  tissue_summary?: string
  report_html_url?: string
  detailed_report_url?: string
  validation_status?: string
  validation_summary?: string
  validation_text_summary?: string
  validation_plan?: any
  validation_result?: { artifacts?: Array<{ type: string; name: string; path: string }> } & Record<string, any>
  reclustering_recommendations?: Array<{
    cluster_id: string
    requires_reclustering: boolean
    reclustering_reason: string
    suggested_reclustering_scope?: string | null
  }>
  proportion_group_labels?: string[]
  proportion_dimension?: string
}

/** Report URLs live on the load response, not inside annotation_state. */
export interface AnnotationReportUrls {
  annotation_report_url?: string
  detailed_report_url?: string
}

function toClusterAnnotation(raw: RawClusterAnnotation): ClusterAnnotation {
  return {
    cluster_id: raw.cluster_id ?? '',
    cell_type: raw.cell_type ?? '',
    subtypes: raw.subtypes ?? [],
    key_markers: raw.key_markers ?? [],
    confidence: raw.confidence ?? 'low',
    evidence: raw.evidence ?? '',
    alternative_cell_types: raw.alternative_cell_types ?? [],
    low_confidence_reason: raw.low_confidence_reason ?? '',
    user_confirmed: raw.user_confirmed ?? false,
    user_override: raw.user_override ?? null,
    ...(raw.group_proportions ? { group_proportions: raw.group_proportions } : {}),
    ...(raw.requires_reclustering != null ? { requires_reclustering: raw.requires_reclustering } : {}),
    ...(raw.reclustering_reason != null ? { reclustering_reason: raw.reclustering_reason } : {}),
    ...(raw.suggested_reclustering_scope !== undefined
      ? { suggested_reclustering_scope: raw.suggested_reclustering_scope }
      : {}),
  }
}

/**
 * Rebuild the annotation dialog payload from the server's authoritative
 * `annotation_state`.
 *
 * The dialog used to be reconstructible only from the transient
 * `annotation_confirmation` realtime message. That message is lossy — /load
 * replays a bounded window and the stream is capped — so once it was gone the
 * gate could not be reopened at all. This lets a reloaded client reach the gate
 * from persisted state instead.
 *
 * Returns null when no gate is pending.
 */
export function buildAnnotationDataFromState(
  state: RawAnnotationState | null | undefined,
  urls: AnnotationReportUrls,
): AnnotationData | null {
  if (!state) return null

  return {
    annotations: (state.cluster_annotations ?? []).map(toClusterAnnotation),
    tissue: state.tissue ?? '',
    organism: state.organism ?? '',
    tissue_summary: state.tissue_summary ?? '',
    proportion_group_labels: state.proportion_group_labels ?? [],
    proportion_dimension: state.proportion_dimension ?? 'group',
    ...(urls.annotation_report_url ? { report_html_url: urls.annotation_report_url } : {}),
    ...(urls.detailed_report_url ? { detailed_report_url: urls.detailed_report_url } : {}),
    ...(state.validation_status != null ? { validation_status: state.validation_status } : {}),
    ...(state.validation_summary != null ? { validation_summary: state.validation_summary } : {}),
    ...(state.validation_text_summary != null
      ? { validation_text_summary: state.validation_text_summary }
      : {}),
    ...(state.validation_plan != null ? { validation_plan: state.validation_plan } : {}),
    ...(state.validation_result != null ? { validation_result: state.validation_result } : {}),
    ...(state.reclustering_recommendations != null
      ? { reclustering_recommendations: state.reclustering_recommendations }
      : {}),
  }
}
