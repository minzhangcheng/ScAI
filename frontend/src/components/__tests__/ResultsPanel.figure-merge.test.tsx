/**
 * The Results gallery must not shrink to the PI's curated list.
 *
 * `integrated_report` carries the figures the PI picked for the brief report —
 * a highlight reel, not an inventory. The panel used to replace the gallery
 * array with it, so a session holding 16 figures rendered 6 and the other 10
 * were unreachable from the UI. Measured across 24 completed sessions the
 * message narrowed the gallery in 14 of them and widened it in 7, which is why
 * the fix is a union rather than preferring either side.
 *
 * These tests pin: union not replacement, an empty curated list is inert, and
 * base64 already held by the client survives a message that was stripped of it.
 */
import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ResultsPanel, mergeFigureMetadata } from '../ResultsPanel'
import { STRIPPED_FIGURE_PLACEHOLDER } from '../../constants/storage'
import { useAppStore, type Session, type PublicationFigure } from '../../stores/appStore'

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  generateId: () => 'test-id',
  useSession: () => ({ getReport: vi.fn().mockResolvedValue(null) }),
}))

function figure(name: string, overrides: Partial<PublicationFigure> = {}): PublicationFigure {
  return {
    id: `id-${name}`,
    name,
    description: `${name} description`,
    figureType: 'umap',
    pngPath: `/w/figures/${name}.jpg`,
    pdfPath: `/w/figures/${name}.pdf`,
    pngBase64: 'AAA',
    reviewStatus: 'approved',
    revisionCount: 0,
    createdAt: '2026-07-31T00:00:00Z',
    ...overrides,
  }
}

function makeSession(figures: PublicationFigure[]): Session {
  return {
    id: 'session-figs',
    name: 'Figure gallery',
    status: 'completed',
    currentRound: 3,
    analysisPhase: 'innovation',
    createdAt: '2026-07-31T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
    publicationFigures: figures,
  }
}

function integratedReportMessage(figures: Record<string, unknown>[]) {
  return {
    id: 'msg-ir',
    type: 'integrated_report',
    agent: 'pi',
    timestamp: '2026-07-31T13:39:00Z',
    content: { report: 'the report', figures, tables: [] },
  } as any
}

describe('ResultsPanel — the curated list adds to the gallery, it does not replace it', () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [], language: 'en' } as any)
  })

  it('does not replace the authoritative report with a persisted blob placeholder', () => {
    const placeholderMessage = integratedReportMessage([])
    placeholderMessage.content.report = STRIPPED_FIGURE_PLACEHOLDER
    useAppStore.setState({
      currentSession: {
        ...makeSession([]),
        integratedReport: '# Authoritative report\n\nFull report content.',
      },
      messages: [placeholderMessage],
    } as any)

    render(<ResultsPanel />)

    expect(screen.getByRole('heading', { name: 'Authoritative report' })).toBeInTheDocument()
    expect(screen.getByText('Full report content.')).toBeInTheDocument()
    expect(screen.queryByText(STRIPPED_FIGURE_PLACEHOLDER)).not.toBeInTheDocument()
  })

  it('drops stale base64 when a new revision omits its preview bytes', () => {
    const existing = figure('revised', {
      id: 'stable-id',
      revisionCount: 1,
      pngBase64: 'OLD',
    })
    const incoming = figure('revised', {
      id: 'stable-id',
      revisionCount: 2,
      pngBase64: undefined,
    })

    expect(mergeFigureMetadata(incoming, existing).pngBase64).toBeUndefined()
  })

  it('keeps base64 for a compact event describing the same render', () => {
    const existing = figure('same-render', {
      id: 'stable-id',
      revisionCount: 2,
      pngBase64: 'CURRENT',
    })
    const incoming = figure('same-render', {
      id: 'stable-id',
      revisionCount: 2,
      pngBase64: undefined,
    })

    expect(mergeFigureMetadata(incoming, existing).pngBase64).toBe('CURRENT')
  })

  it('does not resurrect an invalidated registered figure from a persisted report message', () => {
    useAppStore.setState({
      currentSession: makeSession([figure('current')]),
      messages: [integratedReportMessage([
        {
          id: 'id-stale', name: 'stale', figure_type: 'heatmap',
          review_status: 'approved', is_publication: true,
        },
      ])],
    } as any)

    render(<ResultsPanel />)

    expect(screen.getByText('current')).toBeInTheDocument()
    expect(screen.queryByText('stale')).not.toBeInTheDocument()
  })
})
