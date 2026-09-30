import { describe, it, expect } from 'vitest'
import { transformLoadResponse, transformRoundResults } from './transformLoadResponse'
import { applyStoreUpdates } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import type { SessionLoadResponse } from '../types'

import { resultIdentityKey } from './roundKey'

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

  it('loads the versioned integrated report body and frozen figure catalogue', () => {
    const result = transformLoadResponse({
      ...MINIMAL_RESPONSE,
      integrated_report: 'legacy body',
      integrated_report_state: {
        published: {
          generation_id: 'generation-2',
          publication_revision: 2,
          lineage_id: 'main',
          markdown: '# Versioned body',
          catalog_fingerprint: 'f'.repeat(64),
          published_at: '2026-09-19T00:00:00Z',
          figure_refs: [{
            number: 1,
            figure_id: 'figure-1',
            name: 'Frozen panel',
            render_sha256: 'a'.repeat(64),
            pdf_sha256: 'b'.repeat(64),
            review_status: 'approved',
          }],
        },
      },
    })

    expect(result.sessionUpdates.integratedReport).toBe('# Versioned body')
    expect(result.sessionUpdates.integratedReportPublication).toMatchObject({
      generationId: 'generation-2',
      publicationRevision: 2,
      figureRefs: [{ number: 1, figureId: 'figure-1', pdfSha256: 'b'.repeat(64) }],
    })
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

  it('transforms publication_figures and publication_tables', () => {
    const data: SessionLoadResponse = {
      ...MINIMAL_RESPONSE,
      publication_figures: [{
        id: 'f1', name: 'Fig 1', description: 'A figure',
        figure_type: 'umap', png_path: '/fig.png', pdf_path: '/fig.pdf',
        review_status: 'pending', review_unavailable: true, pi_feedback: 'unavailable',
        revision_count: 1, plan_step_id: 4, plan_id: 'plan-v2', purpose: 'result',
        created_at: '2026-01-01',
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
    expect(result.sessionUpdates.publicationFigures![0].reviewStatus).toBe('pending')
    expect(result.sessionUpdates.publicationFigures![0].reviewUnavailable).toBe(true)
    expect(result.sessionUpdates.publicationFigures![0].planStepId).toBe(4)
    expect(result.sessionUpdates.publicationFigures![0].planId).toBe('plan-v2')
    expect(result.sessionUpdates.publicationFigures![0].purpose).toBe('result')
    expect(result.sessionUpdates.publicationTables).toHaveLength(1)
    expect(result.sessionUpdates.publicationTables![0].columns).toEqual(['col1', 'col2'])
  })

  it('drops persisted image sentinels so authenticated URL fallbacks are used', () => {
    const result = transformLoadResponse({
      ...MINIMAL_RESPONSE,
      publication_figures: [{
        id: 'f1', name: 'Fig 1', description: 'A figure',
        figure_type: 'umap', png_path: '/fig.jpg', pdf_path: '/fig.pdf',
        png_base64: '[image]', review_status: 'approved',
        revision_count: 0, created_at: '2026-01-01',
      }],
      graphic_abstract: {
        id: 'ga1', prompt: 'test prompt', image_base64: '[image]',
        is_text_mode: false, created_at: '2026-01-01',
      },
    })

    expect(result.sessionUpdates.publicationFigures![0].pngBase64).toBeUndefined()
    expect(result.sessionUpdates.graphicAbstract!.imageBase64).toBeUndefined()
  })

})

describe('transformRoundResults integration fields', () => {

  it('preserves evidence, coverage, and restore lineage identically on reload', () => {
    const [round] = transformRoundResults([{
      round: 6,
      phase: 'innovation',
      round_status: 'partial',
      evidence_status: 'substantive',
      direction_coverage_eligible: false,
      bioinformatics_evidence_verified: false,
      evidence_execution_ids: ['exec-6'],
      evidence_notes: ['Mandatory gate was force-released.'],
      evidence_gate_findings: [{ gate: 'analysis-unit', forced_release: true }],
      lineage_id: 'restore-snapshot-2',
      branch_id: 'restore-snapshot-2',
      branch_label: 'Alternative branch',
      restored_from_snapshot_id: 'snapshot-2',
      direction_key: 'candidate-axis',
      result_identity_key: 'restore-snapshot-2:innovation:6:candidate-axis',
    }])

    expect(round).toMatchObject({
      roundStatus: 'partial',
      evidenceStatus: 'substantive',
      directionCoverageEligible: false,
      bioinformaticsEvidenceVerified: false,
      evidenceExecutionIds: ['exec-6'],
      evidenceNotes: ['Mandatory gate was force-released.'],
      evidenceGateFindings: [{ gate: 'analysis-unit', forced_release: true }],
      lineageId: 'restore-snapshot-2',
      branchId: 'restore-snapshot-2',
      branchLabel: 'Alternative branch',
      restoredFromSnapshotId: 'snapshot-2',
      directionKey: 'candidate-axis',
      resultIdentityKey: 'restore-snapshot-2:innovation:6:candidate-axis',
    })
  })

  it('keeps main and restored-branch results distinct without explicit identity keys', () => {
    const rounds = transformRoundResults([
      {
        round: 6,
        phase: 'innovation',
        direction: '  Candidate Axis  ',
        pi_summary: 'Main result',
      },
      {
        round: 6,
        phase: 'innovation',
        // Backend normalize_direction_text uses lowercase, collapsed spaces;
        // the fallback must share that contract so raw `direction` and an
        // explicit `direction_key` identify the same biological direction.
        direction_key: 'candidate axis',
        lineage_id: 'restore-snapshot-6',
        branch_id: 'restore-snapshot-6',
        pi_summary: 'Branch result',
      },
    ])

    expect(rounds.map(resultIdentityKey)).toEqual([
      'main:innovation:6:candidate axis',
      'restore-snapshot-6:innovation:6:candidate axis',
    ])
    expect(new Set(rounds.map(resultIdentityKey)).size).toBe(2)
  })
})

describe('transformLoadResponse — annotation gate', () => {
  it('rebuilds annotationGateData from annotation_state so a reloaded client can reopen the gate', () => {
    const result = transformLoadResponse({
      ...MINIMAL_RESPONSE,
      status: 'waiting_annotation',
      annotation_report_url: '/api/session/test-123/annotation-report/html',
      annotation_state: {
        cluster_annotations: [
          { cluster_id: '0', cell_type: 'Hepatocyte', key_markers: ['Alb'] },
        ],
        tissue: 'liver',
        organism: 'mouse',
        pending: true,
      },
    })

    expect(result.storeUpdates.annotationGateData).not.toBeNull()
    expect(result.storeUpdates.annotationGateData!.tissue).toBe('liver')
    expect(result.storeUpdates.annotationGateData!.organism).toBe('mouse')
    expect(result.storeUpdates.annotationGateData!.annotations).toHaveLength(1)
    expect(result.storeUpdates.annotationGateData!.report_html_url).toBe(
      '/api/session/test-123/annotation-report/html'
    )
  })

  it('clears annotationGateData when the session has no pending gate', () => {
    const result = transformLoadResponse(MINIMAL_RESPONSE)
    expect(result.storeUpdates.annotationGateData).toBeNull()
  })
})

describe('transformLoadResponse — LLM intervention gate', () => {

  it('rebuilds a disabled-agent dialog from the authoritative load response', () => {
    const result = transformLoadResponse({
      ...MINIMAL_RESPONSE,
      status: 'waiting_input',
      wait_input_reason: 'agent_disabled',
      agent_disabled_info: {
        agent: 'literature',
        node_name: 'literature_research',
        generation: 2,
      },
    })

    expect(result.storeUpdates.llmFailureInfo).toMatchObject({
      agent: 'literature',
      nodeName: 'literature_research',
      generation: 2,
      reason: 'agent_disabled',
    })
  })

  it.each([undefined, 'annotation_tool_failure'])('retains failure kind %s on load and clears the released gate', (failureKind) => {
    const pending = transformLoadResponse({
      ...MINIMAL_RESPONSE,
      status: 'waiting_input',
      wait_input_reason: 'llm_failure',
      llm_failure_info: {
        agent: 'bioinformatics',
        node_name: 'bioinformatics_analysis',
        generation: 4,
        failure_kind: failureKind,
      },
    })

    applyStoreUpdates(pending.storeUpdates)
    expect(useAppStore.getState().llmFailureInfo?.nodeName).toBe('bioinformatics_analysis')
    expect(useAppStore.getState().llmFailureInfo?.failureKind).toBe(failureKind)
    expect(useAppStore.getState().showLlmTimeoutDialog).toBe(true)

    applyStoreUpdates(transformLoadResponse(MINIMAL_RESPONSE).storeUpdates)
    expect(useAppStore.getState().llmFailureInfo).toBeNull()
    expect(useAppStore.getState().showLlmTimeoutDialog).toBe(false)
  })
})

describe('round report download identity', () => {
  it('retains the saved segment on reload and websocket transformation', () => {
    const [round] = transformRoundResults([{ round: 2, phase: 'innovation', report_segment_id: 'saved-round' }])
    expect(round.reportSegmentId).toBe('saved-round')
  })
})


describe('stage review report reload URLs', () => {
  it('clears a stale exploration download when the latest stage report failed', () => {
    const result = transformLoadResponse({ ...MINIMAL_RESPONSE, exploration_report_url: null })
    expect(result.sessionUpdates).toHaveProperty('explorationReportUrl', '')
  })
})
