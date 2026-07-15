import { describe, it, expect } from 'vitest'
import { buildAnnotationDataFromState } from './annotationGate'
import type { RawAnnotationState } from '../types'

const rawState: RawAnnotationState = {
  cluster_annotations: [
    {
      cluster_id: '0',
      cell_type: 'Hepatocyte',
      subtypes: ['periportal-like'],
      key_markers: ['Alb', 'Cyp2e1'],
      confidence: 'high',
      evidence: 'Canonical hepatocyte markers',
      alternative_cell_types: [],
      low_confidence_reason: '',
      user_confirmed: false,
      user_override: null,
      group_proportions: { Con: 9.8, DKO: 0.7 },
    },
  ],
  tissue: 'liver',
  organism: 'mouse',
  tissue_summary: 'Liver tissue summary',
  analysis_summary: 'Analysis summary',
  pending: true,
  confirmed: false,
  proportion_group_labels: ['Con', 'DKO'],
  proportion_dimension: 'group',
  validation_status: 'completed',
  validation_summary: 'Validation summary',
  validation_text_summary: 'Validation text',
  reclustering_recommendations: [
    { cluster_id: '3', requires_reclustering: true, reclustering_reason: 'mixed' },
  ],
}

describe('buildAnnotationDataFromState', () => {
  it('rebuilds the dialog payload from the persisted annotation state', () => {
    const data = buildAnnotationDataFromState(rawState, {})

    expect(data).not.toBeNull()
    expect(data!.tissue).toBe('liver')
    expect(data!.organism).toBe('mouse')
    expect(data!.tissue_summary).toBe('Liver tissue summary')
    expect(data!.proportion_group_labels).toEqual(['Con', 'DKO'])
    expect(data!.proportion_dimension).toBe('group')
    expect(data!.validation_status).toBe('completed')
    expect(data!.validation_summary).toBe('Validation summary')
    expect(data!.validation_text_summary).toBe('Validation text')
    expect(data!.reclustering_recommendations).toHaveLength(1)
  })

  it('maps cluster_annotations onto the annotations the dialog renders', () => {
    const data = buildAnnotationDataFromState(rawState, {})

    expect(data!.annotations).toHaveLength(1)
    expect(data!.annotations![0].cluster_id).toBe('0')
    expect(data!.annotations![0].cell_type).toBe('Hepatocyte')
    expect(data!.annotations![0].key_markers).toEqual(['Alb', 'Cyp2e1'])
    expect(data!.annotations![0].group_proportions).toEqual({ Con: 9.8, DKO: 0.7 })
  })

  it('carries report URLs through, since they live outside annotation_state', () => {
    const data = buildAnnotationDataFromState(rawState, {
      annotation_report_url: '/api/session/s1/annotation-report/html',
      detailed_report_url: '/api/session/s1/detailed-report/html',
    })

    expect(data!.report_html_url).toBe('/api/session/s1/annotation-report/html')
    expect(data!.detailed_report_url).toBe('/api/session/s1/detailed-report/html')
  })

  it('returns null when no annotation gate is pending', () => {
    expect(buildAnnotationDataFromState(null, {})).toBeNull()
    expect(buildAnnotationDataFromState(undefined, {})).toBeNull()
  })

  it('tolerates a state with no clusters rather than throwing', () => {
    const data = buildAnnotationDataFromState({ tissue: 'liver' } as RawAnnotationState, {})

    expect(data!.annotations).toEqual([])
    expect(data!.tissue).toBe('liver')
  })
})
