import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  useAppStore,
  type IntegratedReportPublication, type Message, type PublicationFigure, type Session,
} from '../../stores/appStore'
import {
  mergeIntegratedReportFigures, reportBoundFigures, useResultsController,
} from './useResultsController'

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  generateId: () => 'test-id',
  useSession: () => ({ getReport: vi.fn().mockResolvedValue(null) }),
}))

function figure(id: string, name = 'Shared panel'): PublicationFigure {
  return {
    id,
    name,
    description: id,
    figureType: 'heatmap',
    pngPath: `/${id}.jpg`,
    pdfPath: '',
    reviewStatus: 'approved',
    revisionCount: 0,
    createdAt: '2026-09-19T00:00:00Z',
  }
}

function publication(figureId: string): IntegratedReportPublication {
  return {
    generationId: 'generation-2',
    publicationRevision: 2,
    lineageId: 'main',
    markdown: '# Report',
    figureRefs: [{
      number: 1,
      figureId,
      name: 'Shared panel',
      description: 'Frozen version',
      figureType: 'heatmap',
      renderSha256: 'a'.repeat(64),
      reviewStatus: 'approved',
      reviewUnavailable: false,
      revisionCount: 0,
      diagnostic: false,
      isPublication: true,
    }],
    reportingLimitations: [],
    catalogFingerprint: 'f'.repeat(64),
    publishedAt: '2026-09-19T00:00:00Z',
  }
}

function session(overrides: Partial<Session> = {}): Session {
  return {
    id: 'report-session',
    name: 'Report session',
    status: 'waiting_input',
    currentRound: 1,
    analysisPhase: 'innovation',
    createdAt: '2026-09-19T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
    ...overrides,
  }
}

function message(type: Message['type'], content: Message['content']): Message {
  return {
    id: `message-${type}`,
    type,
    agent: 'pi',
    timestamp: '2026-09-19T00:01:00Z',
    content,
  }
}

describe('Integrated Report figure binding', () => {
  beforeEach(() => {
    useAppStore.setState({ currentSession: null, messages: [], language: 'en' })
  })

  it('keeps a frozen id that is absent from the mutable inventory', () => {
    const session = {
      publicationFigures: [figure('current-id')],
      integratedReportPublication: publication('frozen-id'),
    } as Session

    const bound = reportBoundFigures(session)

    expect(bound.figures.map(item => item.id)).toEqual([
      'frozen-id', 'current-id',
    ])
    expect(bound.figures[0]).toMatchObject({
      reportNumber: 1,
      reportGenerationId: 'generation-2',
    })
    expect(bound.curatedKeys).toEqual(new Set(['frozen-id']))
  })

  it('clears old numbers and binds same-name revisions by exact id', () => {
    const old = {
      ...figure('old-id'), reportNumber: 1, reportGenerationId: 'generation-1',
    }
    const current = figure('current-id')

    const merged = mergeIntegratedReportFigures([old, current], [{
      id: 'current-id',
      name: 'Shared panel',
      description: 'Current frozen version',
      figure_type: 'heatmap',
      review_status: 'approved',
      revision_count: 1,
      report_number: 1,
      report_generation_id: 'generation-2',
      is_publication: true,
    }])

    expect(merged.figures).toHaveLength(2)
    expect(merged.figures.find(item => item.id === 'current-id')).toMatchObject({
      reportNumber: 1,
      reportGenerationId: 'generation-2',
    })
    expect(merged.figures.find(item => item.id === 'old-id')?.reportNumber)
      .toBeUndefined()
    expect(merged.curatedKeys).toEqual(new Set(['current-id']))
  })

  it.each(['figure_review', 'figure_approved', 'figure_revision'] as const)(
    'preserves the loaded publication when %s history arrives without a report event',
    type => {
      const published = publication('frozen-id')
      published.figureRefs[0].piReportNote = 'Retain the stated limitation.'
      useAppStore.setState({
        currentSession: session({
          integratedReport: published.markdown,
          integratedReportPublication: published,
          publicationFigures: [figure('frozen-id')],
        }),
      })
      const { result } = renderHook(() => useResultsController())

      act(() => {
        useAppStore.getState().addMessage(message(type, {
          figure: {
            id: 'frozen-id',
            name: 'Shared panel',
            description: 'Mutable review description',
            figure_type: 'heatmap',
            review_status: 'pending',
            revision_count: 3,
            png_base64: 'REVIEW_PREVIEW',
          },
        }))
      })

      expect(result.current.orderedFigures).toHaveLength(1)
      expect(result.current.orderedFigures[0]).toMatchObject({
        id: 'frozen-id',
        description: 'Frozen version',
        reviewStatus: 'approved',
        revisionCount: 0,
        reportNumber: 1,
        reportGenerationId: 'generation-2',
        piReportNote: 'Retain the stated limitation.',
      })
      expect(result.current.orderedFigures[0].pngBase64).toBeUndefined()
      expect(result.current.curatedCount).toBe(1)
    },
  )

  it('keeps a different review artifact beside the frozen same-name card', () => {
    useAppStore.setState({
      currentSession: session({
        integratedReportPublication: publication('frozen-id'),
      }),
      messages: [message('figure_approved', {
        figure: { ...figure('new-id'), description: 'New render' },
      })],
    })

    const { result } = renderHook(() => useResultsController())

    expect(result.current.orderedFigures.map(item => item.id)).toEqual([
      'frozen-id', 'new-id',
    ])
    expect(result.current.orderedFigures[0].reportNumber).toBe(1)
    expect(result.current.orderedFigures[1].reportNumber).toBeUndefined()
    expect(result.current.orderedFigures[1].description).toBe('New render')
  })

  it('does not replace a current artifact with a legacy same-name explicit id', () => {
    useAppStore.setState({
      currentSession: session({ publicationFigures: [figure('current-id')] }),
      messages: [message('integrated_report', {
        report: '# Legacy report',
        figures: [{ ...figure('old-id'), is_publication: true }],
      })],
    })

    const { result } = renderHook(() => useResultsController())

    expect(result.current.orderedFigures.map(item => item.id)).toEqual(['current-id'])
    expect(result.current.orderedFigures[0].description).toBe('current-id')
  })

  it('resolves an id-less legacy figure to the current inventory id', () => {
    useAppStore.setState({
      currentSession: session({ publicationFigures: [figure('current-id')] }),
      messages: [message('integrated_report', {
        report: '# Legacy report',
        figures: [{
          name: 'Shared panel',
          description: 'Legacy selected panel',
          figure_type: 'heatmap',
          is_publication: true,
        }],
      })],
    })

    const { result } = renderHook(() => useResultsController())

    expect(result.current.orderedFigures).toHaveLength(1)
    expect(result.current.orderedFigures[0]).toMatchObject({
      id: 'current-id', description: 'Legacy selected panel',
    })
    expect(result.current.curatedCount).toBe(1)
  })
})
