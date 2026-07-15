import { describe, expect, it } from 'vitest';
import AnalysisPanelSrc from '../AnalysisPanel.tsx?raw';
import annotationGateSrc from '../../utils/annotationGate.ts?raw';

describe('AnalysisPanel annotation proportion payload wiring', () => {
  it('declares annotation proportion payload fields', () => {
    // AnnotationData moved to utils/annotationGate so the dialog payload can be
    // rebuilt from the server's annotation_state, not only from the message.
    const src = annotationGateSrc as string;

    expect(src).toContain('proportion_group_labels?: string[]');
    expect(src).toContain('proportion_dimension?: string');
  });

  it('passes the proportion payload through to the dialog', () => {
    const src = AnalysisPanelSrc as string;

    expect(src).toContain('proportionGroupLabels={annotationData.proportion_group_labels');
    expect(src).toContain('proportionDimension={annotationData.proportion_dimension');
  });
});
