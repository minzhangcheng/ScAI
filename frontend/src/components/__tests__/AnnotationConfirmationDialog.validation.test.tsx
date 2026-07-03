import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { AnnotationConfirmationDialog } from '../AnnotationConfirmationDialog';

const baseProps: any = {
  annotations: [{ cluster_id: '7', cell_type: 'Technical/Contamination', confidence: 'low',
                  key_markers: ['Nrxn1'], subtypes: [], evidence: '', n_cells: 100,
                  alternative_cell_types: [], low_confidence_reason: '', user_confirmed: false,
                  user_override: null, requires_reclustering: true,
                  reclustering_reason: 'bimodal lineage', suggested_reclustering_scope: 'targeted_subcluster' }],
  tissue: 'liver', organism: 'mouse', tissueSummary: '',
  language: 'en', countdownSeconds: 0,
  validationStatus: 'completed',
  validationSummary: 'Cluster 7 reclassified as technical contamination.',
  validationTextSummary: '- Cluster 7: contradicted; action=relabel; qc=[high doublet]',
  validationResult: { artifacts: [{ type: 'figure', name: 'marker_val',
                      path: '/data/sessions/abc/workspace/annotation_validation/m.jpg' }] },
  reclusteringRecommendations: [{ cluster_id: '7', requires_reclustering: true,
                                  reclustering_reason: 'bimodal lineage', suggested_reclustering_scope: 'targeted_subcluster' }],
  onConfirm: async () => {}, onRequestSupplementary: async () => {}, onClose: () => {},
};

describe('AnnotationConfirmationDialog validation', () => {
  it('renders summary, reclustering rec, and artifact link', () => {
    render(<AnnotationConfirmationDialog {...baseProps} />);
    expect(screen.getByText(/reclassified as technical contamination/i)).toBeInTheDocument();
    expect(screen.getByText(/bimodal lineage/i)).toBeInTheDocument();
    expect(screen.getByText(/marker_val/i)).toBeInTheDocument();
  });

  it('omits the validation section when no validation data is present', () => {
    const props = { ...baseProps, validationStatus: undefined, validationSummary: undefined,
                    validationTextSummary: undefined, validationResult: undefined,
                    reclusteringRecommendations: undefined };
    render(<AnnotationConfirmationDialog {...props} />);
    expect(screen.queryByText(/reclassified as technical contamination/i)).not.toBeInTheDocument();
  });

  it('refreshes the table when a revision changes cell_type at the same confidence', () => {
    const a0 = { cluster_id: '7', cell_type: 'Hepatic neurons', confidence: 'low',
                 key_markers: ['Nrxn1'], subtypes: [], evidence: '', n_cells: 100,
                 alternative_cell_types: [], low_confidence_reason: '', user_confirmed: false,
                 user_override: null };
    const props1: any = { ...baseProps, annotations: [a0] };
    const { rerender } = render(<AnnotationConfirmationDialog {...props1} />);
    expect(screen.getByText(/Hepatic neurons/i)).toBeInTheDocument();

    const a1 = { ...a0, cell_type: 'Technical/Contamination' };
    rerender(<AnnotationConfirmationDialog {...{ ...baseProps, annotations: [a1] }} />);
    expect(screen.getByText(/Technical\/Contamination/i)).toBeInTheDocument();
    expect(screen.queryByText(/Hepatic neurons/i)).not.toBeInTheDocument();
  });

  it('renders group proportion header and values in label order', () => {
    const props = {
      ...baseProps,
      annotations: [{
        ...baseProps.annotations[0],
        group_proportions: { NASH: 8, Control: 12.5 },
      }],
      proportionGroupLabels: ['Control', 'NASH'],
      proportionDimension: 'group',
    };
    render(<AnnotationConfirmationDialog {...props} />);

    expect(screen.getByText('Cell Proportion (Control/NASH)')).toBeInTheDocument();
    expect(screen.getByText('12.5% / 8.0%')).toBeInTheDocument();
  });

  it('renders missing proportion values as dash and handles overall dimension', () => {
    const props = {
      ...baseProps,
      annotations: [{
        ...baseProps.annotations[0],
        group_proportions: {},
      }],
      proportionGroupLabels: ['整体'],
      proportionDimension: 'overall',
    };
    render(<AnnotationConfirmationDialog {...props} />);

    expect(screen.getByText('Cell Proportion (Overall)')).toBeInTheDocument();
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('renders proportion column without empty parentheses when labels are unavailable', () => {
    const props = {
      ...baseProps,
      annotations: [{
        ...baseProps.annotations[0],
        group_proportions: {},
      }],
      proportionGroupLabels: [],
      proportionDimension: 'group',
    };
    render(<AnnotationConfirmationDialog {...props} />);

    expect(screen.getByText('Cell Proportion')).toBeInTheDocument();
    expect(screen.queryByText('Cell Proportion ()')).not.toBeInTheDocument();
  });

  it('refreshes the table when only group proportions change', () => {
    const a0 = {
      ...baseProps.annotations[0],
      cell_type: 'Hepatocyte',
      group_proportions: { Control: 12.5, NASH: 8 },
    };
    const props1: any = {
      ...baseProps,
      annotations: [a0],
      proportionGroupLabels: ['Control', 'NASH'],
      proportionDimension: 'group',
    };
    const { rerender } = render(<AnnotationConfirmationDialog {...props1} />);
    expect(screen.getByText('12.5% / 8.0%')).toBeInTheDocument();

    const a1 = { ...a0, group_proportions: { Control: 20, NASH: 5 } };
    rerender(<AnnotationConfirmationDialog {...{ ...props1, annotations: [a1] }} />);
    expect(screen.getByText('20.0% / 5.0%')).toBeInTheDocument();
    expect(screen.queryByText('12.5% / 8.0%')).not.toBeInTheDocument();
  });
});
