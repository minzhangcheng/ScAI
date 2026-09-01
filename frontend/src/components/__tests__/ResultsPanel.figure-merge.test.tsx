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

  it('keeps every session figure when the PI curates a subset', () => {
    const all = ['a', 'b', 'c', 'd'].map(n => figure(n))
    useAppStore.setState({
      currentSession: makeSession(all),
      messages: [integratedReportMessage([
        { id: 'id-a', name: 'a', figure_type: 'umap', review_status: 'approved' },
      ])],
    } as any)

    render(<ResultsPanel />)

    expect(screen.getByText('a')).toBeInTheDocument()
    expect(screen.getByText('b')).toBeInTheDocument()
    expect(screen.getByText('c')).toBeInTheDocument()
    expect(screen.getByText('d')).toBeInTheDocument()
  })

  it('admits a curated figure the session payload never carried', () => {
    useAppStore.setState({
      currentSession: makeSession([figure('from_session')]),
      // Workspace-scanned figures reach the client only through this message.
      messages: [integratedReportMessage([
        {
          id: 'id-scanned', name: 'from_workspace', figure_type: 'heatmap',
          review_status: 'pending', is_publication: false,
        },
      ])],
    } as any)

    render(<ResultsPanel />)

    expect(screen.getByText('from_workspace')).toBeInTheDocument()
    expect(screen.getByText('from_session')).toBeInTheDocument()
  })

  it('ignores an integrated_report that curated nothing', () => {
    // `content.figures` is truthy when empty, so this used to empty the gallery.
    useAppStore.setState({
      currentSession: makeSession([figure('kept')]),
      messages: [integratedReportMessage([])],
    } as any)

    render(<ResultsPanel />)

    expect(screen.getByText('kept')).toBeInTheDocument()
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

  it('does not duplicate a figure when the report is regenerated', () => {
    // Workspace figures are given a fresh uuid on every assembly, so identity
    // has to come from the name.
    useAppStore.setState({
      currentSession: makeSession([figure('shared')]),
      messages: [
        integratedReportMessage([{ id: 'uuid-1', name: 'shared', figure_type: 'umap', review_status: 'approved' }]),
        integratedReportMessage([{ id: 'uuid-2', name: 'shared', figure_type: 'umap', review_status: 'approved' }]),
      ],
    } as any)

    render(<ResultsPanel />)

    expect(screen.getAllByText('shared')).toHaveLength(1)
  })

  it('keeps plan provenance when a compact report payload updates a figure', () => {
    const existing = figure('planned', { planStepId: 7, planId: 'plan-v2', purpose: 'result' })
    const compact = figure('planned', {
      pngBase64: undefined,
      planStepId: undefined,
      planId: undefined,
      purpose: undefined,
    })

    expect(mergeFigureMetadata(compact, existing)).toMatchObject({
      pngBase64: 'AAA',
      planStepId: 7,
      planId: 'plan-v2',
      purpose: 'result',
    })
  })

  it('drops stale base64 when a same-name event replaces the figure id', () => {
    const existing = figure('replaced', { id: 'old-id', pngBase64: 'OLD' })
    const incoming = figure('replaced', {
      id: 'new-id',
      pngBase64: undefined,
    })

    expect(mergeFigureMetadata(incoming, existing).pngBase64).toBeUndefined()
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

  it('adds a figure immediately from a revision event', () => {
    useAppStore.setState({
      currentSession: makeSession([]),
      messages: [{
        id: 'msg-revision',
        type: 'figure_revision',
        agent: 'pi',
        timestamp: '2026-07-31T13:39:00Z',
        content: {
          figure: {
            id: 'id-live',
            name: 'live_revision',
            description: 'Needs another pass',
            figure_type: 'heatmap',
            png_path: '/w/figures/live_revision.jpg',
            pdf_path: '/w/figures/live_revision.pdf',
            review_status: 'revision_requested',
            revision_count: 1,
            plan_step_id: 8,
            plan_id: 'plan-v3',
            purpose: 'result',
          },
        },
      }],
    } as any)

    render(<ResultsPanel />)

    expect(screen.getByText('live_revision')).toBeInTheDocument()
    expect(screen.getByText('Revision Requested')).toBeInTheDocument()
    expect(screen.getByText('Plan step 8')).toBeInTheDocument()
    expect(screen.queryByText('Final')).not.toBeInTheDocument()
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
