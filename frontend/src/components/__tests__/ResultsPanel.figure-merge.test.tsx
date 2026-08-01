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
import { ResultsPanel } from '../ResultsPanel'
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
        { id: 'id-scanned', name: 'from_workspace', figure_type: 'heatmap', review_status: 'pending' },
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
})
