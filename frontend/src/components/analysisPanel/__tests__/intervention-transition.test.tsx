import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore, type Session } from '../../../stores/appStore'
import { useAnalysisPanelInterventionActions } from '../actions/useAnalysisPanelInterventionActions'

const mocks = vi.hoisted(() => ({
  apiFetch: vi.fn(),
}))

vi.mock('../../../hooks/useApi', () => ({
  apiFetch: mocks.apiFetch,
  generateId: () => 'error-id',
}))

const activeSession: Session = {
  id: 'session-a',
  name: 'session-a',
  status: 'waiting_species_input',
  currentRound: 1,
  analysisPhase: 'exploration',
  createdAt: '2026-09-01T00:00:00Z',
  instruction: 'test',
  treatmentGroups: [],
  dataFiles: [],
  biochemistryIndicators: [],
  roundResults: [],
}

function options() {
  return {
    currentSession: activeSession,
    clearExplorationFailedState: vi.fn(),
    clearSpeciesInputState: vi.fn(),
    setShowSpeciesInputDialog: vi.fn(),
    setShowLlmTimeoutDialog: vi.fn(),
    setShowStopDialog: vi.fn(),
    handleStop: vi.fn(),
    annotationData: null,
    annotationGateData: null,
    setAnnotationData: vi.fn(),
    setShowAnnotationDialog: vi.fn(),
    integrationGateData: null,
    setShowIntegrationGateDialog: vi.fn(),
    explorationReportData: null,
    setShowExplorationDialog: vi.fn(),
    feedbackRoundData: null,
    setShowFeedbackDialog: vi.fn(),
    setShowPlanReviewDialog: vi.fn(),
    wsConnect: vi.fn(),
    maxExplorationRound: 1,
    addMessage: vi.fn(),
  }
}

describe('Intervention dialog transitions', () => {
  beforeEach(() => {
    mocks.apiFetch.mockReset()
    useAppStore.setState({
      currentSession: activeSession,
      llmFailureInfo: {
        agent: 'pi', model: 'test', llmServer: 'test', error: 'timeout',
        nodeName: 'pi_planning', generation: 2,
      },
    })
  })

  it('hides species input immediately and restores it with an actionable error', async () => {
    mocks.apiFetch.mockRejectedValue(new Error('species rejected'))
    const props = options()
    const { result } = renderHook(() => useAnalysisPanelInterventionActions(props))

    await act(async () => {
      await expect(result.current.handleSpeciesInput('Mus musculus')).rejects.toThrow(
        'species rejected',
      )
    })

    expect(props.setShowSpeciesInputDialog.mock.calls).toEqual([[false], [true]])
    expect(props.clearSpeciesInputState).not.toHaveBeenCalled()
    expect(props.addMessage).toHaveBeenCalledWith(expect.objectContaining({
      type: 'error', content: expect.stringContaining('species rejected'),
    }))
  })

  it('hides an LLM intervention immediately and restores it on rejection', async () => {
    mocks.apiFetch.mockRejectedValue(new Error('stale generation'))
    const props = options()
    const { result } = renderHook(() => useAnalysisPanelInterventionActions(props))

    await act(async () => {
      await result.current.handleLlmRetry()
    })

    expect(props.setShowLlmTimeoutDialog.mock.calls).toEqual([[false], [true]])
    expect(props.addMessage).toHaveBeenCalledWith(expect.objectContaining({
      type: 'error', content: expect.stringContaining('stale generation'),
    }))
  })
})
