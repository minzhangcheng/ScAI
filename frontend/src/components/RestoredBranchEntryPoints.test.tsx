import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ResultsView } from './results/ResultsView'
import { AnalysisDialogs } from './analysisPanel/AnalysisDialogs'

vi.mock('./AnnotationConfirmationDialog', () => ({
  AnnotationConfirmationDialog: ({ reportHtmlUrl, detailedReportUrl }: {
    reportHtmlUrl?: string
    detailedReportUrl?: string
  }) => <div data-testid="annotation-urls">{reportHtmlUrl}|{detailedReportUrl}</div>,
}))

describe('restored report entry points', () => {
  it('keeps round status but does not render a bioinformatics verification badge', () => {
    render(<ResultsView controller={{
      currentSession: {
        id: 'session-evidence', name: 'Results', status: 'completed', currentRound: 1,
        createdAt: '2026-08-29T00:00:00Z', instruction: 'test', treatmentGroups: [],
        dataFiles: [], biochemistryIndicators: [], restoreBranchId: 'main',
        roundResults: [{
          round: 1, phase: 'exploration', roundStatus: 'partial',
          evidenceStatus: 'substantive', bioinformaticsEvidenceVerified: false,
          directionCoverageEligible: true, evidenceNotes: [], lineageId: 'main',
          pathologyFindings: [], bioinformaticsFindings: [], biochemistryFindings: [],
          literatureFindings: [], resultIdentityKey: 'main:1',
        }],
      },
      curatedCount: 0, downloadReport: vi.fn(), expandedRounds: new Set(),
      graphicAbstract: null, integratedReport: null, language: 'en', loading: false,
      orderedFigures: [], publicationFigures: [], publicationTables: [],
      t: (key: string) => key, toggleRound: vi.fn(),
    } as never} />)

    expect(screen.getByText('roundStatusPartial')).toBeInTheDocument()
    expect(screen.queryByText('evidenceUnverified')).not.toBeInTheDocument()
    expect(screen.queryByText('evidenceVerified')).not.toBeInTheDocument()
  })

  it('scopes the Results full-report link to the restored branch', () => {
    render(<ResultsView controller={{
      currentSession: {
        id: 'session-results', name: 'Results', status: 'completed', currentRound: 1,
        createdAt: '2026-08-29T00:00:00Z', instruction: 'test', treatmentGroups: [],
        dataFiles: [], biochemistryIndicators: [], roundResults: [],
        restoreBranchId: 'restore/a b',
      },
      curatedCount: 0,
      downloadReport: vi.fn(),
      expandedRounds: new Set(),
      graphicAbstract: null,
      integratedReport: null,
      language: 'en',
      loading: false,
      orderedFigures: [],
      publicationFigures: [],
      publicationTables: [],
      t: (key: string) => key,
      toggleRound: vi.fn(),
    } as never} />)

    expect(screen.getByRole('link', { name: 'View Full Report' })).toHaveAttribute(
      'href',
      '/api/session/session-results/report/html?branch_id=restore%2Fa%20b',
    )
  })

  it('scopes annotation report URLs without duplicating existing branch queries', () => {
    render(<AnalysisDialogs controller={{
      annotationData: {
        annotations: [],
        report_html_url: '/api/session/s/annotation-report/html?view=full',
        detailed_report_url: '/api/session/s/detailed-report/html?branch_id=existing',
      },
      currentSession: { id: 's', currentRound: 1, restoreBranchId: 'restore/a b' },
      language: 'en',
      roundWaitSeconds: 30,
      showAnnotationDialog: true,
      setShowAnnotationDialog: vi.fn(),
      handleConfirmAnnotations: vi.fn(),
      handleSupplementaryAnalysis: vi.fn(),
    } as never} />)

    expect(screen.getByTestId('annotation-urls')).toHaveTextContent(
      '/api/session/s/annotation-report/html?view=full&branch_id=restore%2Fa%20b|'
      + '/api/session/s/detailed-report/html?branch_id=existing',
    )
  })
})
