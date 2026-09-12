/**
 * Result artifacts belong to the exact `<session-id>,<branch-id>` pair.
 *
 * The defect: component-local result state kept a source session's figures
 * while the store already named the restored child. Figure URLs are built from
 * `currentSession.id`, so those source records became child URL 404s. A render
 * must expose no old-owner artifact even before a synchronization effect runs.
 */
import { renderHook, act } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useResultsController } from '../results/useResultsController'
import { useAppStore, type Session, type PublicationFigure } from '../../stores/appStore'

const mocks = vi.hoisted(() => ({
  getReport: vi.fn(),
}))

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  generateId: () => 'test-id',
  useSession: () => ({ getReport: mocks.getReport }),
}))

function figure(name: string): PublicationFigure {
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
    createdAt: '2026-09-01T00:00:00Z',
  }
}

function session(
  id: string,
  overrides: Partial<Session> = {},
): Session {
  return {
    id,
    name: id,
    status: 'waiting_input',
    currentRound: 1,
    analysisPhase: 'innovation',
    createdAt: '2026-09-01T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
    ...overrides,
  }
}

describe('Results artifacts are owner-isolated', () => {
  beforeEach(() => {
    mocks.getReport.mockReset()
    mocks.getReport.mockResolvedValue(null)
    useAppStore.setState({ messages: [], language: 'en' } as any)
  })

  it('shows only the new owner artifacts immediately after a session switch', () => {
    useAppStore.setState({
      currentSession: session('source-session', {
        publicationFigures: [figure('source-figure')],
      }),
      messages: [],
    } as any)

    const { result } = renderHook(() => useResultsController())
    expect(result.current.publicationFigures.map(item => item.name)).toEqual([
      'source-figure',
    ])

    act(() => {
      useAppStore.setState({
        currentSession: session('child-session'),
        messages: [],
      } as any)
    })

    expect(result.current.currentSession?.id).toBe('child-session')
    expect(result.current.publicationFigures).toEqual([])
    expect(result.current.publicationTables).toEqual([])
    expect(result.current.integratedReport).toBeNull()
    expect(result.current.graphicAbstract).toBeNull()
    expect(result.current.curatedCount).toBe(0)
  })

  it('isolates a branch-only owner change', () => {
    useAppStore.setState({
      currentSession: session('same-session', {
        restoreBranchId: 'main',
        publicationFigures: [figure('source-figure')],
      }),
      messages: [],
    } as any)

    const { result, rerender } = renderHook(() => useResultsController())
    expect(result.current.publicationFigures).toHaveLength(1)

    act(() => {
      useAppStore.getState().updateSession({ restoreBranchId: 'restore-r001' })
    })
    rerender()

    expect(result.current.publicationFigures).toEqual([])
    expect(result.current.report).toBeNull()
  })

  it('ignores a late source report response after the owner changed', async () => {
    let resolveReport: ((value: unknown) => void) | undefined
    mocks.getReport.mockImplementation(() => new Promise(resolve => {
      resolveReport = resolve
    }))

    useAppStore.setState({
      currentSession: session('source-session', { status: 'completed' }),
      messages: [],
    } as any)
    const { result } = renderHook(() => useResultsController())

    act(() => {
      useAppStore.setState({
        currentSession: session('child-session', { status: 'waiting_input' }),
        messages: [],
      } as any)
    })

    await act(async () => {
      resolveReport?.({ final_summary: 'SOURCE-ONLY', generated_at: 'x' })
      await Promise.resolve()
    })

    expect(result.current.report).toBeNull()
    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('clears messages in the same store transaction as an owner change', () => {
    const source = session('source-session')
    const chatController = new AbortController()
    useAppStore.getState().setCurrentSession(source)
    useAppStore.setState({
      messages: [{
        id: 'old-message',
        type: 'result',
        content: 'source',
        timestamp: '2026-09-01T00:00:00Z',
      }] as any,
      chatMessages: [{
        id: 'old-chat', role: 'assistant', content: 'source chat',
        timestamp: '2026-09-01T00:00:00Z',
      }],
      chatLoading: true,
      chatDelegationActive: true,
      chatDelegationPlan: 'source plan',
      chatAbortController: chatController,
    })

    act(() => {
      useAppStore.getState().setCurrentSession(session('child-session'))
    })

    expect(useAppStore.getState().currentSession?.id).toBe('child-session')
    expect(useAppStore.getState().messages).toEqual([])
    expect(useAppStore.getState().chatMessages).toEqual([])
    expect(useAppStore.getState().chatLoading).toBe(false)
    expect(useAppStore.getState().chatDelegationActive).toBe(false)
    expect(useAppStore.getState().chatDelegationPlan).toBeNull()
    expect(useAppStore.getState().chatAbortController).toBeNull()
    expect(chatController.signal.aborted).toBe(true)
  })

  it('clears chat state atomically when only the branch owner changes', () => {
    const chatController = new AbortController()
    useAppStore.getState().setCurrentSession(session('same-session', {
      restoreBranchId: 'source-branch',
    }))
    useAppStore.setState({
      chatMessages: [{
        id: 'source-chat', role: 'assistant', content: 'source branch',
        timestamp: '2026-09-01T00:00:00Z',
      }],
      chatLoading: true,
      chatDelegationActive: true,
      chatDelegationPlan: 'source branch plan',
      chatAbortController: chatController,
    })

    act(() => {
      useAppStore.getState().updateSession({ restoreBranchId: 'child-branch' })
    })

    const state = useAppStore.getState()
    expect(state.currentSession?.restoreBranchId).toBe('child-branch')
    expect(state.chatMessages).toEqual([])
    expect(state.chatLoading).toBe(false)
    expect(state.chatDelegationActive).toBe(false)
    expect(state.chatDelegationPlan).toBeNull()
    expect(state.chatAbortController).toBeNull()
    expect(chatController.signal.aborted).toBe(true)
  })
})
