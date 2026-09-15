import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisPanel } from '../AnalysisPanel'
import { STRIPPED_FIGURE_PLACEHOLDER } from '../../constants/storage'
import { useAppStore, type Message, type RoundResult, type Session } from '../../stores/appStore'

const sessionMocks = vi.hoisted(() => ({
  startAnalysis: vi.fn(),
  resumeAnalysis: vi.fn(),
  continueRound: vi.fn().mockResolvedValue(undefined),
  pauseAnalysis: vi.fn(),
  stopAnalysis: vi.fn(),
  apiFetch: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('../../hooks/useApi', () => ({
  apiFetch: sessionMocks.apiFetch,
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

const authoritativeIntegrationRound: RoundResult = {
  round: 4,
  summary: 'Authoritative integration round summary.',
  phase: 'integration',
  pathologyFindings: [],
  bioinformaticsFindings: [],
  biochemistryFindings: [],
  literatureFindings: [],
  crossModalCandidates: [],
  integrationCandidates: [{ direction: 'Authoritative candidate direction' }],
  integrationReportSummary: 'Authoritative full integration summary.',
  roundStatus: 'completed',
  evidenceStatus: 'substantive',
  evidenceExecutionIds: [],
  evidenceNotes: [],
  evidenceGateFindings: [],
  lineageId: 'main',
  directionKey: '',
  resultIdentityKey: 'main:integration:4:',
}

const strippedIntegrationRoundMessage: Message = {
  id: 'round-4-integration',
  type: 'round_summary',
  agent: 'pi',
  timestamp: '2026-09-15T00:52:27Z',
  content: {
    round: 4,
    max_rounds: 3,
    phase: 'integration',
    status: 'completed',
    summary: STRIPPED_FIGURE_PLACEHOLDER,
    integration_report_summary: STRIPPED_FIGURE_PLACEHOLDER,
    integration_candidates: [{ direction: 'Authoritative candidate direction' }],
    findings: [],
    data_requests: {},
    lineage_id: 'main',
    direction_key: '',
    result_identity_key: 'main:integration:4:',
  },
}

describe('AnalysisPanel integration gate lifecycle', () => {
  beforeEach(() => {
    sessionMocks.continueRound.mockClear()
    sessionMocks.continueRound.mockResolvedValue(undefined)
    sessionMocks.apiFetch.mockClear()
    sessionMocks.apiFetch.mockResolvedValue(undefined)
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

  afterEach(() => {
    vi.useRealTimers()
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

  it('uses authoritative round text when persisted realtime history contains image sentinels', async () => {
    useAppStore.setState({
      currentSession: {
        ...makeSession('running', 'innovation'),
        currentRound: 4,
        roundResults: [authoritativeIntegrationRound],
      },
      messages: [strippedIntegrationRoundMessage],
    })

    render(<AnalysisPanel />)

    expect(await screen.findByText(authoritativeIntegrationRound.summary)).toBeInTheDocument()
    expect(screen.getByText(authoritativeIntegrationRound.integrationReportSummary!)).toBeInTheDocument()
    expect(screen.queryByText(STRIPPED_FIGURE_PLACEHOLDER)).not.toBeInTheDocument()
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

  it('opens the integration modal automatically and reopens it from the waiting badge', async () => {
    useAppStore.setState({
      currentSession: makeSession('waiting_input'),
      messages: [integrationGateMessage],
    })

    render(<AnalysisPanel />)

    expect(await screen.findByRole('dialog', { name: /Integration Analysis/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: /Integration Analysis/i })).not.toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Waiting for Input/i }))
    expect(await screen.findByRole('dialog', { name: /Integration Analysis/i })).toBeInTheDocument()
  })

  it('reconnects when the waiting badge has no locally recoverable gate payload', () => {
    const reconnect = vi.fn()
    useAppStore.setState({
      currentSession: makeSession('waiting_input', 'innovation'),
      messages: [],
      wsConnect: reconnect,
    })

    render(<AnalysisPanel />)
    fireEvent.click(screen.getByRole('button', { name: /Waiting for Input/i }))

    expect(reconnect).toHaveBeenCalledTimes(1)
  })

  it('keeps the original countdown deadline while the integration modal is closed', async () => {
    vi.useFakeTimers()
    useAppStore.setState({
      currentSession: makeSession('waiting_input'),
      messages: [integrationGateMessage],
      roundWaitSeconds: 2,
    })

    render(<AnalysisPanel />)
    await act(async () => {})
    expect(screen.getByRole('dialog', { name: /Integration Analysis/i })).toBeInTheDocument()

    await act(async () => { await vi.advanceTimersByTimeAsync(1000) })
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(screen.queryByRole('dialog', { name: /Integration Analysis/i })).not.toBeInTheDocument()

    await act(async () => { await vi.advanceTimersByTimeAsync(500) })
    fireEvent.click(screen.getByRole('button', { name: /Waiting for Input/i }))
    expect(screen.getByRole('dialog', { name: /Integration Analysis/i })).toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(600) })

    expect(sessionMocks.apiFetch).toHaveBeenCalledTimes(1)
    expect(sessionMocks.apiFetch).toHaveBeenCalledWith(
      '/session/session-1/choose-direction',
      expect.objectContaining({ body: JSON.stringify({ let_pi_decide: true }) }),
    )
  })

  it('does not repeat a no-wait PI decision after close and reopen', async () => {
    sessionMocks.apiFetch.mockRejectedValue(new Error('upstream unavailable'))
    useAppStore.setState({
      currentSession: makeSession('waiting_input'),
      messages: [integrationGateMessage],
      roundWaitSeconds: 0,
    })

    render(<AnalysisPanel />)
    await waitFor(() => expect(sessionMocks.apiFetch).toHaveBeenCalledTimes(1))

    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    fireEvent.click(screen.getByRole('button', { name: /Waiting for Input/i }))

    await waitFor(() => expect(screen.getByRole('dialog', { name: /Integration Analysis/i })).toBeInTheDocument())
    expect(sessionMocks.apiFetch).toHaveBeenCalledTimes(1)
  })

  it('restores focus when authoritative session state directly unmounts the gate', async () => {
    useAppStore.setState({
      currentSession: makeSession('waiting_input'),
      messages: [],
    })

    const view = (
      <>
        <button type="button">Stable outside control</button>
        <AnalysisPanel />
      </>
    )
    const { rerender } = render(view)
    const outside = screen.getByRole('button', { name: /stable outside control/i })
    outside.focus()

    await act(async () => {
      useAppStore.setState({ messages: [integrationGateMessage] })
      rerender(view)
    })
    await waitFor(() => expect(screen.getByRole('button', { name: /close/i })).toHaveFocus())

    await act(async () => {
      useAppStore.setState({ currentSession: makeSession('running') })
      rerender(view)
    })
    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: /Integration Analysis/i })).not.toBeInTheDocument()
    })
    expect(outside).toHaveFocus()
  })
})
