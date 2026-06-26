import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ExplorationReportDialog } from './ExplorationReportDialog'

describe('ExplorationReportDialog evidence-gated directions', () => {
  it('shows compact fields and expandable evidence details', async () => {
    render(
      <ExplorationReportDialog
        report={{
          cluster_overview: '',
          data_quality_warnings: [],
          strongest_heterogeneity: [],
          key_pathway_changes: [],
          abnormal_markers: [],
          low_confidence_annotations: [],
          pi_direction_assessment: '',
          research_direction_candidates: [{
            direction_id: 'dir_001',
            direction: 'IFN network',
            rationale: 'R',
            overall_priority: 'high',
            data_reliability: 'high',
            literature_novelty: 'high',
            functional_impact: 'high',
            evidence_ref: 'dir_001',
            risk_flags: ['requires_new_experiment'],
          }],
          direction_literature_evidence: [{ direction_id: 'dir_001', direction: 'IFN network', status: 'completed', search_tier: 'deep', confidence: 'medium', negative_validation_confidence: 'medium', species_tissue_qualification: 'cross_tissue_only', direct_evidence_count: 0, analogous_evidence_count: 1, contradictory_evidence_count: 0, methodological_precedent_count: 1, direct_evidence: [], analogous_evidence: [{ pmid: '123', title: 'Trace metal immunity', relevance: '', supports: '', why_it_matters: 'Analogous mechanism' }], queries: [{ category: 'analogous_mechanism', query: 'trace metal immunity' }], limitations: ['No direct evidence'] }],
        }}
        roundSummary="summary"
        language="en"
        countdownSeconds={-1}
        onChooseDirections={vi.fn()}
        onLetPiDecide={vi.fn()}
        onRequestSupplementary={vi.fn()}
        onClose={vi.fn()}
      />
    )
    expect(screen.getByText('IFN network')).toBeInTheDocument()
    expect(screen.getByText(/Priority/i)).toBeInTheDocument()
    expect(screen.getByText(/Requires new experiment/i)).toBeInTheDocument()
    fireEvent.click(screen.getByText(/View evidence/i))
    expect(screen.getByText(/PMID: 123/i)).toBeInTheDocument()
    expect(screen.getAllByText(/trace metal immunity/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/Cross-tissue evidence only/i)).toBeInTheDocument()
    expect(screen.getByText(/No direct evidence/i)).toBeInTheDocument()
  })

  it('shows search timed out when partial evidence has zero retained references', () => {
    render(<ExplorationReportDialog
      report={{
        cluster_overview: '',
        data_quality_warnings: [],
        strongest_heterogeneity: [],
        key_pathway_changes: [],
        abnormal_markers: [],
        low_confidence_annotations: [],
        pi_direction_assessment: '',
        research_direction_candidates: [{
          direction_id: 'dir_001',
          direction: 'Resolve copper redistribution',
          rationale: 'Copper data are contradictory.',
          overall_priority: 'high',
          data_reliability: 'medium',
          literature_novelty: 'unknown',
          functional_impact: 'high',
          evidence_ref: 'dir_001',
        }],
        direction_literature_evidence: [{
          direction_id: 'dir_001',
          direction: 'Resolve copper redistribution',
          status: 'partial',
          timed_out: true,
          timeout_seconds: 180,
          direct_evidence_count: 0,
          analogous_evidence_count: 0,
          contradictory_evidence_count: 0,
          methodological_precedent_count: 0,
          direct_evidence: [],
          analogous_evidence: [],
          contradictory_evidence: [],
          methodological_precedent: [],
          limitations: ['Direction-level timeout exceeded'],
        }],
      }}
      roundSummary="Round 1 complete"
      language="en"
      countdownSeconds={-1}
      onChooseDirections={vi.fn()}
      onLetPiDecide={vi.fn()}
      onRequestSupplementary={vi.fn()}
      onClose={vi.fn()}
    />)

    expect(screen.getByText(/Search timed out/i)).toBeInTheDocument()
    fireEvent.click(screen.getByText(/View evidence/i))
    expect(screen.getByText(/timed out before references could be retained/i)).toBeInTheDocument()
    expect(screen.getByText(/Direct: 0/i)).toBeInTheDocument()
  })

  it('shows partial evidence with retained PMID when timed out after collecting references', () => {
    render(<ExplorationReportDialog
      report={{
        cluster_overview: '',
        data_quality_warnings: [],
        strongest_heterogeneity: [],
        key_pathway_changes: [],
        abnormal_markers: [],
        low_confidence_annotations: [],
        pi_direction_assessment: '',
        research_direction_candidates: [{
          direction_id: 'dir_003',
          direction: 'Investigate metabolic collapse',
          rationale: 'Metabolic pathways are downregulated.',
          overall_priority: 'high',
          data_reliability: 'high',
          literature_novelty: 'medium',
          functional_impact: 'high',
          evidence_ref: 'dir_003',
        }],
        direction_literature_evidence: [{
          direction_id: 'dir_003',
          direction: 'Investigate metabolic collapse',
          status: 'partial',
          timed_out: true,
          timeout_seconds: 180,
          direct_evidence_count: 1,
          analogous_evidence_count: 0,
          contradictory_evidence_count: 0,
          methodological_precedent_count: 0,
          direct_evidence: [{
            pmid: '12345678',
            title: 'Mouse liver metabolism and copper stress',
            relevance: 'high',
            supports: 'direct',
            why_it_matters: 'Retained before timeout.',
          }],
          analogous_evidence: [],
          contradictory_evidence: [],
          methodological_precedent: [],
          limitations: ['Direction-level timeout exceeded'],
        }],
      }}
      roundSummary="Round 1 complete"
      language="en"
      countdownSeconds={-1}
      onChooseDirections={vi.fn()}
      onLetPiDecide={vi.fn()}
      onRequestSupplementary={vi.fn()}
      onClose={vi.fn()}
    />)

    expect(screen.getByText(/Partial evidence/i)).toBeInTheDocument()
    fireEvent.click(screen.getByText(/View evidence/i))
    expect(screen.getByText(/references below are partial/i)).toBeInTheDocument()
    expect(screen.getByText(/PMID: 12345678/i)).toBeInTheDocument()
  })
})
