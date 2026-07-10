import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisPanel } from '../AnalysisPanel'
import { useAppStore, type Message, type Session } from '../../stores/appStore'

const sessionMocks = vi.hoisted(() => ({
  startAnalysis: vi.fn(),
  resumeAnalysis: vi.fn(),
  continueRound: vi.fn().mockResolvedValue(undefined),
  pauseAnalysis: vi.fn(),
  stopAnalysis: vi.fn(),
}))

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  generateId: () => 'test-id',
  useConfig: () => ({
    checkStatus: vi.fn(),
    validateApiKey: vi.fn(),
    validateSavedConfig: vi.fn(),
    saveConfig: vi.fn(),
  }),
  useSession: () => sessionMocks,
  useUserConfig: () => ({
    getUserConfigStatus: vi.fn(),
    validateSavedConfig: vi.fn(),
  }),
}))

function makeSession(status: Session['status'], analysisPhase: Session['analysisPhase'] = 'integration'): Session {
  return {
    id: 'session-1',
    name: 'Integration gate regression',
    status,
    currentRound: 9,
    analysisPhase,
    createdAt: '2026-07-08T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }
}

const integrationGateMessage: Message = {
  id: 'gate-1',
  type: 'integration_gate',
  timestamp: '2026-07-08T00:00:00Z',
  content: {
    cycle: 2,
    candidates: [{ direction: 'Stale integration direction' }],
    html_url: '/report',
  },
}

const innovationWaitingSummary: Message = {
  id: 'round-9',
  type: 'round_summary',
  agent: 'pi',
  timestamp: '2026-07-08T01:00:00Z',
  content: {
    round: 9,
    phase: 'innovation',
    status: 'waiting_input',
    summary: 'Innovation round is waiting for ordinary feedback.',
    findings: [],
    data_requests: {},
  },
}

describe('AnalysisPanel integration gate lifecycle', () => {
  beforeEach(() => {
    sessionMocks.continueRound.mockClear()
    sessionMocks.continueRound.mockResolvedValue(undefined)
    useAppStore.setState({
      currentSession: null,
      messages: [],
      language: 'en',
      wsConnected: true,
      wsConnect: undefined,
      readMode: false,
      agentErrors: [],
      authUser: { id: 'user-1', username: 'tester', role: 'admin', status: 'active' },
      roundWaitSeconds: -1,
    })
  })

  it('hides stale integration gate content after the session resumes running', async () => {
    useAppStore.setState({
      currentSession: makeSession('waiting_input'),
      messages: [integrationGateMessage],
    })

    const { rerender } = render(<AnalysisPanel />)

    expect(await screen.findByText(/Stale integration direction/i)).toBeInTheDocument()

    useAppStore.setState({ currentSession: makeSession('running') })
    rerender(<AnalysisPanel />)

    await waitFor(() => {
      expect(screen.queryByText(/Stale integration direction/i)).not.toBeInTheDocument()
    })
  })

  it('restores the ordinary feedback dialog and Continue action after stale gate history replay', async () => {
    useAppStore.setState({
      currentSession: makeSession('waiting_input', 'innovation'),
      messages: [integrationGateMessage, innovationWaitingSummary],
      roundWaitSeconds: -1,
    })

    render(<AnalysisPanel />)

    expect(await screen.findByRole('heading', { name: /Round 9 Complete/i })).toBeInTheDocument()
    expect(screen.queryByText(/Stale integration direction/i)).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /^Continue$/i }))

    await waitFor(() => {
      expect(sessionMocks.continueRound).toHaveBeenCalledTimes(1)
    })
  })
})
