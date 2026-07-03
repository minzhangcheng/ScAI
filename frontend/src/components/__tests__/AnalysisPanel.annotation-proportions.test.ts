import { describe, expect, it } from 'vitest';
import AnalysisPanelSrc from '../AnalysisPanel.tsx?raw';

describe('AnalysisPanel annotation proportion payload wiring', () => {
  it('accepts annotation proportion payload fields and passes them to the dialog', () => {
    const src = AnalysisPanelSrc as string;

    expect(src).toContain('proportion_group_labels?: string[]');
    expect(src).toContain('proportion_dimension?: string');
    expect(src).toContain('proportionGroupLabels={annotationData.proportion_group_labels');
    expect(src).toContain('proportionDimension={annotationData.proportion_dimension');
  });
});
