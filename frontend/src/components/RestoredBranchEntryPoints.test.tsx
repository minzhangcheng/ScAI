import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ResultsView } from './results/ResultsView'


vi.mock('./AnnotationConfirmationDialog', () => ({
  AnnotationConfirmationDialog: ({ reportHtmlUrl, detailedReportUrl }: {
    reportHtmlUrl?: string
    detailedReportUrl?: string
  }) => <div data-testid="annotation-urls">{reportHtmlUrl}|{detailedReportUrl}</div>,
}))

describe('restored report entry points', () => {

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

  it('scopes loaded exploration and integration report links to the restored branch', () => {
    render(<ResultsView controller={{
      currentSession: {
        id: 'session-results-links', name: 'Results', status: 'completed', currentRound: 1,
        createdAt: '2026-08-29T00:00:00Z', instruction: 'test', treatmentGroups: [],
        dataFiles: [], biochemistryIndicators: [], roundResults: [],
        restoreBranchId: 'restore/a b',
        explorationReportUrl: '/api/session/session-results-links/exploration-report/html',
        integrationReportUrl: '/api/session/session-results-links/integration-report/html?view=full',
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

    expect(screen.getByRole('link', { name: 'View Exploration Report' })).toHaveAttribute(
      'href',
      '/api/session/session-results-links/exploration-report/html?branch_id=restore%2Fa%20b',
    )
    expect(screen.getByRole('link', { name: 'View Integration Report' })).toHaveAttribute(
      'href',
      '/api/session/session-results-links/integration-report/html?view=full&branch_id=restore%2Fa%20b',
    )
  })
})
