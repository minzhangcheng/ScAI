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

  it('renders enhanced direction fields and premise conflict badge', () => {
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
          direction: 'Resolve premise conflict',
          rationale: 'Rationale',
          overall_priority: 'high',
          data_reliability: 'medium',
          literature_novelty: 'unknown',
          functional_impact: 'high',
          recommendation: 'needs_more_data',
          biological_plausibility: 'medium',
          artifact_risk: 'high',
          core_observation: 'Cluster 7 hepatocytes show Mt2 log2FC=1.3.',
          main_hypothesis: 'Copper redistribution drives stress.',
          alternative_hypotheses: ['Zinc response', 'Annotation artifact'],
          artifact_risks: ['sample confounding'],
          supporting_data: [{ source: 'bioinformatics', evidence: 'Cluster 7 hepatocytes show Mt2 log2FC=1.3.', strength: 'high' }],
          contradictory_or_missing_evidence: ['No direct matched model evidence'],
          contradicts_stated_premise: true,
          validation_plan: {
            next_analysis: ['Sample-level pseudobulk'],
            wet_lab_validation: ['ICP-MS'],
            decision_criteria: ['Copper redistribution supports premise conflict'],
          },
          pi_review_summary: 'Premise conflict needs validation first.',
        }],
      }}
      roundSummary="summary"
      language="en"
      countdownSeconds={-1}
      onChooseDirections={vi.fn()}
      onLetPiDecide={vi.fn()}
      onRequestSupplementary={vi.fn()}
      onClose={vi.fn()}
    />)

    expect(screen.getByText('Premise conflict')).toBeInTheDocument()
    expect(screen.getByText('Needs more data')).toBeInTheDocument()
    expect(screen.getByText('Artifact risk:')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Resolve premise conflict'))
    expect(screen.getByText('Cluster 7 hepatocytes show Mt2 log2FC=1.3.')).toBeInTheDocument()
    expect(screen.getByText('Copper redistribution drives stress.')).toBeInTheDocument()
    expect(screen.getByText('Zinc response')).toBeInTheDocument()
    expect(screen.getByText('[bioinformatics] Cluster 7 hepatocytes show Mt2 log2FC=1.3. (strength: high)')).toBeInTheDocument()
    expect(screen.getByText('ICP-MS')).toBeInTheDocument()
    expect(screen.getByText('Consider requesting supplementary analysis before choosing this direction for Stage 2.')).toBeInTheDocument()
  })

  it('renders enhanced literature evidence metadata', () => {
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
          direction: 'Evidence rich direction',
          rationale: 'Rationale',
          overall_priority: 'high',
          data_reliability: 'high',
        }],
        direction_literature_evidence: [{
          direction_id: 'dir_001',
          direction: 'Evidence rich direction',
          status: 'partial',
          search_tier: 'standard',
          confidence: 'medium',
          support_level: 'weak',
          contradiction_level: 'moderate',
          prior_art_level: 'low',
          novelty_status: 'unknown_due_to_timeout',
          knowledge_gap_summary: 'Matched model evidence is missing.',
          fallback_tier: 'tier3_broad',
          scheduler_limitation: 'minimum attempt only',
          direct_evidence_count: 0,
          analogous_evidence_count: 1,
          contradictory_evidence_count: 1,
          methodological_precedent_count: 0,
          analogous_evidence: [],
          contradictory_evidence: [],
          queries: [],
          limitations: ['Direction-level timeout exceeded'],
        }],
      }}
      roundSummary="summary"
      language="en"
      countdownSeconds={-1}
      onChooseDirections={vi.fn()}
      onLetPiDecide={vi.fn()}
      onRequestSupplementary={vi.fn()}
      onClose={vi.fn()}
    />)

    fireEvent.click(screen.getByText('View evidence'))
    expect(screen.getByText('Support level: weak')).toBeInTheDocument()
    expect(screen.getByText('Contradiction level: moderate')).toBeInTheDocument()
    expect(screen.getByText('Prior art: low')).toBeInTheDocument()
    expect(screen.getByText('Novelty status: unknown_due_to_timeout')).toBeInTheDocument()
    expect(screen.getByText('Matched model evidence is missing.')).toBeInTheDocument()
    expect(screen.getByText('Fallback tier: tier3_broad')).toBeInTheDocument()
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
