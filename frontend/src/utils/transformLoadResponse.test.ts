import { describe, it, expect } from 'vitest'
import { transformLoadResponse } from './transformLoadResponse'
import type { SessionLoadResponse } from '../types'

const MINIMAL_RESPONSE: SessionLoadResponse = {
  session_id: 'test-123',
  name: 'Test Session',
  status: 'running',
  current_round: 2,
  created_at: '2026-01-01T00:00:00Z',
  initial_instruction: 'Analyze cells',
  treatment_groups: [],
  data_files: [],
  biochemistry_indicators: [],
  round_results: [],
  publication_figures: [],
  publication_tables: [],
}

describe('transformLoadResponse', () => {
  it('transforms minimal response with present fields', () => {
    const result = transformLoadResponse(MINIMAL_RESPONSE)
    expect(result.sessionUpdates.status).toBe('running')
    expect(result.sessionUpdates.currentRound).toBe(2)
    expect(result.sessionUpdates.roundResults).toEqual([])
    expect(result.sessionUpdates.treatmentGroups).toEqual([])
    expect(result.sessionUpdates.dataFiles).toEqual([])
    expect(result.sessionUpdates.biochemistryIndicators).toEqual([])
    expect(result.sessionUpdates.publicationFigures).toEqual([])
    expect(result.sessionUpdates.publicationTables).toEqual([])
  })

  it('does not include keys for missing fields', () => {
    const { round_results: _rr, data_files: _df, publication_figures: _pf, publication_tables: _pt, graphic_abstract: _ga, final_summary: _fs, integrated_report: _ir, ...rest } = MINIMAL_RESPONSE
    const result = transformLoadResponse(rest)
    expect(result.sessionUpdates.roundResults).toBeUndefined()
    expect(result.sessionUpdates.dataFiles).toBeUndefined()
    expect(result.sessionUpdates.publicationFigures).toBeUndefined()
    expect(result.sessionUpdates.publicationTables).toBeUndefined()
    expect(result.sessionUpdates.graphicAbstract).toBeUndefined()
    expect(result.sessionUpdates.finalSummary).toBeUndefined()
    expect(result.sessionUpdates.integratedReport).toBeUndefined()
  })

  it('filters null entries from round_results', () => {
    const data: SessionLoadResponse = {
      ...MINIMAL_RESPONSE,
      round_results: [
        { round: 1, pi_summary: 'Round 1 summary', pathology_findings: ['finding a'], bioinformatics_findings: [], biochemistry_findings: [], literature_findings: [] },
        null as any,
        { round: 3, summary: 'Round 3 summary', bioinformatics_findings: ['finding b'], pathology_findings: [], biochemistry_findings: [], literature_findings: [] },
      ],
    }
    const result = transformLoadResponse(data)
    expect(result.sessionUpdates.roundResults).toHaveLength(2)
    expect(result.sessionUpdates.roundResults![0].round).toBe(1)
    expect(result.sessionUpdates.roundResults![0].summary).toBe('Round 1 summary')
    expect(result.sessionUpdates.roundResults![0].pathologyFindings).toEqual(['finding a'])
    expect(result.sessionUpdates.roundResults![1].summary).toBe('Round 3 summary')
    expect(result.sessionUpdates.roundResults![1].bioinformaticsFindings).toEqual(['finding b'])
  })

  it('falls back to unknown status as pending', () => {
    const data: SessionLoadResponse = { ...MINIMAL_RESPONSE, status: 'gibberish' }
    const result = transformLoadResponse(data)
    expect(result.sessionUpdates.status).toBe('pending')
  })

  it('transforms publication_figures and publication_tables', () => {
    const data: SessionLoadResponse = {
      ...MINIMAL_RESPONSE,
      publication_figures: [{
        id: 'f1', name: 'Fig 1', description: 'A figure',
        figure_type: 'umap', png_path: '/fig.png', pdf_path: '/fig.pdf',
        review_status: 'approved', pi_feedback: 'good',
        revision_count: 1, created_at: '2026-01-01',
      }],
      publication_tables: [{
        id: 't1', name: 'Table 1', description: 'A table',
        csv_path: '/table.csv', columns: ['col1', 'col2'],
        preview_data: [{ col1: 'a' }], review_status: 'pending',
        created_at: '2026-01-01',
      }],
    }
    const result = transformLoadResponse(data)
    expect(result.sessionUpdates.publicationFigures).toHaveLength(1)
    expect(result.sessionUpdates.publicationFigures![0].figureType).toBe('umap')
    expect(result.sessionUpdates.publicationFigures![0].reviewStatus).toBe('approved')
    expect(result.sessionUpdates.publicationTables).toHaveLength(1)
    expect(result.sessionUpdates.publicationTables![0].columns).toEqual(['col1', 'col2'])
  })

  it('transforms graphic_abstract', () => {
    const data: SessionLoadResponse = {
      ...MINIMAL_RESPONSE,
      graphic_abstract: {
        id: 'ga1', prompt: 'test prompt',
        image_base64: 'base64data', is_text_mode: false, created_at: '2026-01-01',
      },
    }
    const result = transformLoadResponse(data)
    expect(result.sessionUpdates.graphicAbstract).toBeDefined()
    expect(result.sessionUpdates.graphicAbstract!.imageBase64).toBe('base64data')
    expect(result.sessionUpdates.graphicAbstract!.isTextMode).toBe(false)
  })

  it('extracts store-level fields when present', () => {
    const data: SessionLoadResponse = {
      ...MINIMAL_RESPONSE,
      state_version: 42,
      snapshot_count: 7,
      round_wait_seconds: 60,
    }
    const result = transformLoadResponse(data)
    expect(result.storeUpdates.stateVersion).toBe(42)
    expect(result.storeUpdates.snapshotCount).toBe(7)
    expect(result.storeUpdates.roundWaitSeconds).toBe(60)
  })

  it('omits storeUpdates keys when not provided', () => {
    const { state_version, snapshot_count, round_wait_seconds, ...rest } = MINIMAL_RESPONSE
    const result = transformLoadResponse(rest)
    expect(result.storeUpdates.stateVersion).toBeUndefined()
    expect(result.storeUpdates.snapshotCount).toBeUndefined()
    expect(result.storeUpdates.roundWaitSeconds).toBeUndefined()
  })

  it('transforms data_files with both snake_case and camelCase keys', () => {
    const data: SessionLoadResponse = {
      ...MINIMAL_RESPONSE,
      data_files: [{
        id: 'df1', filename: 'data.h5ad', section: 'sequencing' as const,
        file_path: '/path/to/data', data_type: 'sn_rnaseq', group_id: 'g1',
        sample_number: 1, description: 'test', timestamp: '2026-01-01',
      }],
    }
    const result = transformLoadResponse(data)
    expect(result.sessionUpdates.dataFiles![0].filePath).toBe('/path/to/data')
    expect(result.sessionUpdates.dataFiles![0].dataType).toBe('sn_rnaseq')
    expect(result.sessionUpdates.dataFiles![0].groupId).toBe('g1')
  })

  it('transforms treatment_groups with both key formats', () => {
    const data: SessionLoadResponse = {
      ...MINIMAL_RESPONSE,
      treatment_groups: [{
        id: 'tg1', name: 'Control', description: 'Control group',
        sample_count: 3, order: 0,
      }],
    }
    const result = transformLoadResponse(data)
    expect(result.sessionUpdates.treatmentGroups![0].sampleCount).toBe(3)
  })

  it('transforms biochemistry_indicators', () => {
    const data: SessionLoadResponse = {
      ...MINIMAL_RESPONSE,
      biochemistry_indicators: [{
        id: 'bi1', indicator: 'ALT', value: 45.0, unit: 'U/L',
        reference_range: '0-40', interpretation: 'elevated',
        group_id: 'g1', sample_number: 1,
      }],
    }
    const result = transformLoadResponse(data)
    expect(result.sessionUpdates.biochemistryIndicators![0].indicator).toBe('ALT')
    expect(result.sessionUpdates.biochemistryIndicators![0].referenceRange).toBe('0-40')
  })

  it('does not overwrite name with empty string when name is missing', () => {
    const { name, ...rest } = MINIMAL_RESPONSE
    const result = transformLoadResponse(rest)
    expect(result.sessionUpdates.name).toBeUndefined()
  })
})
