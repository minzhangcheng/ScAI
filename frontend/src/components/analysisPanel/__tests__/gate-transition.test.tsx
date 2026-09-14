/**
 * Gate continuations are single-flight and owner-keyed.
 *
 * Switching a restored session from its gate into the workflow transiently
 * persists PAUSED. The old UI rendered the generic Resume control in that
 * window, and a click could produce the `/resume` 400 even though the gate
 * action had already converted the graph.
 */
import { renderHook, act } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAnalysisPanelGateActions } from '../actions/useAnalysisPanelGateActions'
import { useAppStore, type Session } from '../../../stores/appStore'

const mocks = vi.hoisted(() => ({
  apiFetch: vi.fn(),
}))

vi.mock('../../../hooks/useApi', () => ({
  apiFetch: mocks.apiFetch,
}))

function session(
  id: string,
  status: Session['status'],
  restoreBranchId?: string,
): Session {
  return {
    id,
    name: id,
    status,
    currentRound: 1,
    analysisPhase: 'innovation',
    createdAt: '2026-09-01T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
    restoreBranchId,
  }
}

function options(overrides: Record<string, unknown> = {}) {
  return {
    currentSession: session('session-a', 'waiting_input'),
    stateVersion: 10,
    wsConnected: true,
    wsConnect: vi.fn(),
    setShowAnnotationDialog: vi.fn(),
    setShowPlanReviewDialog: vi.fn(),
    setShowExplorationDialog: vi.fn(),
    setShowIntegrationGateDialog: vi.fn(),
    setIntegrationGateData: vi.fn(),
    setIsSupplementaryRunning: vi.fn(),
    roundSupplementaryPendingRef: { current: false },
    onGateActionError: vi.fn(),
    ...overrides,
  }
}

describe('Gate continuation transitions', () => {
  beforeEach(() => {
    mocks.apiFetch.mockReset()
    useAppStore.setState({
      currentSession: session('session-a', 'waiting_input'),
      stateVersion: 10,
      wsConnected: true,
    })
  })

  it('submits one direction endpoint and never an automatic /resume', async () => {
    let resolveRequest: ((value: unknown) => void) | undefined
    mocks.apiFetch.mockImplementation(() => new Promise(resolve => {
      resolveRequest = resolve
    }))
    const props = options()
    const { result } = renderHook(() => useAnalysisPanelGateActions(props))

    let request: Promise<unknown> | undefined
    await act(async () => {
      request = result.current.handleChooseDirections([
        { direction: 'Kupffer heterogeneity', notes: '', isCustom: true },
      ])
      resolveRequest?.({})
      await request
    })

    expect(mocks.apiFetch).toHaveBeenCalledTimes(1)
    expect(String(mocks.apiFetch.mock.calls[0][0])).toContain('/choose-direction')
    expect(
      mocks.apiFetch.mock.calls.some(call => String(call[0]).includes('/resume')),
    ).toBe(false)
    expect(props.onGateActionError).not.toHaveBeenCalled()
    expect(result.current.gateContinuationPending?.ownerKey).toContain('session-a')
  })

  it('double-clicking issues one mutation', async () => {
    let resolveRequest: ((value: unknown) => void) | undefined
    mocks.apiFetch.mockImplementation(() => new Promise(resolve => {
      resolveRequest = resolve
    }))
    const props = options()
    const { result } = renderHook(() => useAnalysisPanelGateActions(props))

    let first: Promise<unknown> | undefined
    await act(async () => {
      first = result.current.handleChooseDirections([
        { direction: 'First click', notes: '', isCustom: false },
      ])
      void result.current.handleChooseDirections([
        { direction: 'Second click', notes: '', isCustom: false },
      ])
      resolveRequest?.({})
      await first
    })

    expect(mocks.apiFetch).toHaveBeenCalledTimes(1)
  })

  it('closes annotation and plan dialogs as soon as their requests start', async () => {
    mocks.apiFetch.mockResolvedValue({})
    const props = options({
      currentSession: session('session-a', 'waiting_annotation'),
    })
    useAppStore.setState({
      currentSession: session('session-a', 'waiting_annotation'),
    })
    const { result } = renderHook(() => useAnalysisPanelGateActions(props))

    await act(async () => {
      await result.current.handleConfirmAnnotations([])
    })
    expect(props.setShowAnnotationDialog).toHaveBeenCalledWith(false)

    await act(async () => {
      await result.current.handlePlanModify('Use the paired design')
    })
    expect(props.setShowPlanReviewDialog).toHaveBeenCalledWith(false)
  })

  it('closes integration selection immediately and restores it on failure', async () => {
    mocks.apiFetch.mockRejectedValue(new Error('selection rejected'))
    const setShowIntegrationGateDialog = vi.fn()
    const props = options({ setShowIntegrationGateDialog })
    const { result } = renderHook(() => useAnalysisPanelGateActions(props))

    await act(async () => {
      await result.current.submitIntegrationDirections([
        { direction: 'Validate Kupffer-cell origin', notes: '' },
      ])
    })

    expect(setShowIntegrationGateDialog.mock.calls).toEqual([[false], [true]])
  })

  it('holds pending through PAUSED and clears on a strictly newer authoritative state', async () => {
    mocks.apiFetch.mockResolvedValue({})
    let props = options()
    const { result, rerender } = renderHook(
      () => useAnalysisPanelGateActions(props),
    )

    await act(async () => {
      await result.current.handleChooseDirections([
        { direction: 'Kupffer heterogeneity', notes: '', isCustom: false },
      ])
    })
    expect(result.current.gateContinuationPending).not.toBeNull()

    // The redrive parks transiently in PAUSED. Equal version does not settle.
    props = options({
      stateVersion: 10,
      currentSession: session('session-a', 'paused'),
    })
    rerender()
    expect(result.current.gateContinuationPending).not.toBeNull()

    // PAUSED never clears, even with a newer version.
    props = options({
      stateVersion: 11,
      currentSession: session('session-a', 'paused'),
    })
    rerender()
    expect(result.current.gateContinuationPending).not.toBeNull()

    // Equal-version RUNNING is still not authoritative settlement.
    props = options({
      stateVersion: 10,
      currentSession: session('session-a', 'running'),
    })
    rerender()
    expect(result.current.gateContinuationPending).not.toBeNull()

    // Missing versions do not clear either.
    props = options({
      stateVersion: undefined,
      currentSession: session('session-a', 'running'),
    })
    rerender()
    expect(result.current.gateContinuationPending).not.toBeNull()

    // A strictly newer authoritative post-redrive state settles once.
    props = options({
      stateVersion: 12,
      currentSession: session('session-a', 'running'),
    })
    rerender()
    expect(result.current.gateContinuationPending).toBeNull()
  })

  it('runs same-owner dialog cleanup when newer state arrives before HTTP success', async () => {
    let resolveRequest: ((value: unknown) => void) | undefined
    mocks.apiFetch.mockImplementation(() => new Promise(resolve => {
      resolveRequest = resolve
    }))
    const setShowExplorationDialog = vi.fn()
    let props = options({ setShowExplorationDialog })
    const { result, rerender } = renderHook(
      () => useAnalysisPanelGateActions(props),
    )

    let request!: Promise<unknown>
    act(() => {
      request = result.current.handleChooseDirections([
        { direction: 'Kupffer heterogeneity', notes: '', isCustom: false },
      ])
    })
    expect(setShowExplorationDialog).toHaveBeenCalledWith(false)
    props = options({
      setShowExplorationDialog,
      stateVersion: 11,
      currentSession: session('session-a', 'running'),
    })
    act(() => {
      useAppStore.getState().updateSession({ status: 'running' })
      useAppStore.getState().setStateVersion(11)
    })
    rerender()
    expect(result.current.gateContinuationPending).toBeNull()

    await act(async () => {
      resolveRequest?.({})
      await request
    })

    expect(setShowExplorationDialog).toHaveBeenCalledWith(false)
    expect(mocks.apiFetch).toHaveBeenCalledTimes(1)
  })

  it('shows a concrete error and restores the action on failure', async () => {
    mocks.apiFetch.mockRejectedValue(new Error('upstream unavailable'))
    const setShowExplorationDialog = vi.fn()
    const props = options({ setShowExplorationDialog })
    const { result } = renderHook(() => useAnalysisPanelGateActions(props))

    await act(async () => {
      await result.current.handleChooseDirections([
        { direction: 'Kupffer heterogeneity', notes: '', isCustom: false },
      ])
    })

    expect(result.current.gateContinuationPending).toBeNull()
    expect(props.onGateActionError).toHaveBeenCalledWith(
      expect.any(Error), 'direction_selection',
    )
    expect(setShowExplorationDialog.mock.calls).toEqual([[false], [true]])
  })

  it('owner switching clears the pending indicator without touching the new owner', async () => {
    mocks.apiFetch.mockResolvedValue({})
    let props = options()
    const { result, rerender } = renderHook(
      () => useAnalysisPanelGateActions(props),
    )
    await act(async () => {
      await result.current.handleLetPiDecide()
    })
    expect(result.current.gateContinuationPending).not.toBeNull()

    props = options({
      currentSession: session('session-b', 'paused', 'restore-child'),
    })
    act(() => {
      useAppStore.getState().setCurrentSession(props.currentSession)
    })
    rerender()
    expect(result.current.gateContinuationPending).toBeNull()
  })
})
