import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useWebSocket } from '../useApi'
import { subscribeSnapshotListingChanged } from '../api/snapshots'
import { useAppStore } from '../../stores/appStore'

describe('useWebSocket', () => {
  const closeMock = vi.fn()
  const wsInstances: Array<{
    url: string
    close: ReturnType<typeof vi.fn>
    onopen: ((event: Event) => void) | null
    onmessage: ((event: MessageEvent) => void) | null
    onclose: ((event: CloseEvent) => void) | null
    onerror: ((event: Event) => void) | null
  }> = []

  class MockWebSocket {
    public readyState: number = WebSocket.OPEN
    public close = closeMock
    public url: string
    private _onopen: ((event: Event) => void) | null = null
    public onmessage: ((event: MessageEvent) => void) | null = null
    public onclose: ((event: CloseEvent) => void) | null = null
    public onerror: ((event: Event) => void) | null = null

    constructor(url: string) {
      this.url = url
      wsInstances.push(this)
    }

    set onopen(handler: ((event: Event) => void) | null) {
      this._onopen = handler
      // Fire synchronously so connectingRef.current gets reset before connect() returns
      if (handler) {
        handler(new Event('open'))
      }
    }

    get onopen(): ((event: Event) => void) | null {
      return this._onopen
    }
  }

  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket as any)
    closeMock.mockClear()
    wsInstances.length = 0
    useAppStore.setState({
      currentSession: null,
      messages: [],
      wsConnected: false,
      stateVersion: 0,
    })
  })

  it('old socket onclose does not disturb the new socket after session switch', async () => {
    const { result } = renderHook(() => useWebSocket())

    const sessionA = {
      id: 'session-A',
      name: 'Session A',
      status: 'completed' as const,
      currentRound: 0,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }

    const sessionB = {
      id: 'session-B',
      name: 'Session B',
      status: 'completed' as const,
      currentRound: 0,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }

    act(() => {
      useAppStore.getState().setCurrentSession(sessionA)
    })

    await act(async () => {
      result.current.connect()
    })

    expect(wsInstances.length).toBe(1)
    expect(result.current.connected).toBe(true)

    act(() => {
      useAppStore.getState().setCurrentSession(sessionB)
    })

    await act(async () => {
      result.current.connect()
    })

    expect(wsInstances.length).toBe(2)
    expect(closeMock).toHaveBeenCalledTimes(1)

    const A = wsInstances[0]
    act(() => {
      A.onclose?.({ code: 1005 } as CloseEvent)
    })

    expect(result.current.connected).toBe(true)
    expect(wsInstances.length).toBe(2)
  })

  it('routes snapshot invalidation without adding an internal message', async () => {
    const session = {
      id: 'session-snapshot',
      name: 'Snapshot session',
      status: 'running' as const,
      currentRound: 1,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }
    useAppStore.getState().setCurrentSession(session)
    const invalidated = vi.fn()
    const unsubscribe = subscribeSnapshotListingChanged(invalidated)
    const { result } = renderHook(() => useWebSocket())
    await act(async () => result.current.connect())

    act(() => {
      wsInstances[0].onmessage?.({
        data: JSON.stringify({ type: 'snapshot_listing_changed' }),
      } as MessageEvent)
    })

    expect(invalidated).toHaveBeenCalledWith(session.id)
    expect(useAppStore.getState().messages).toEqual([])
    unsubscribe()
  })

  it('closes old socket and opens new on session switch', async () => {
    const { result } = renderHook(() => useWebSocket())

    expect(result.current.connected).toBe(false)

    const sessionA = {
      id: 'session-A',
      name: 'Session A',
      status: 'completed' as const,
      currentRound: 0,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }

    const sessionB = {
      id: 'session-B',
      name: 'Session B',
      status: 'completed' as const,
      currentRound: 0,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }

    // Connect to session A
    act(() => {
      useAppStore.getState().setCurrentSession(sessionA)
    })

    await act(async () => {
      result.current.connect()
    })

    expect(wsInstances.length).toBe(1)
    expect(wsInstances[0].url).toContain('session-A')
    expect(closeMock).not.toHaveBeenCalled()

    // Switch to session B and connect
    act(() => {
      useAppStore.getState().setCurrentSession(sessionB)
    })

    await act(async () => {
      result.current.connect()
    })

    // Old socket should have been closed before opening new one
    expect(closeMock).toHaveBeenCalledTimes(1)
    // A new WebSocket should have been created for session B
    expect(wsInstances.length).toBe(2)
    expect(wsInstances[1].url).toContain('session-B')
  })

  it('keeps integration round_summary separate from same-number innovation round', async () => {
    const { result } = renderHook(() => useWebSocket())

    act(() => {
      useAppStore.getState().setCurrentSession({
        id: 'session-integration',
        name: 'Integration session',
        status: 'running' as const,
        currentRound: 3,
        createdAt: new Date().toISOString(),
        instruction: '',
        treatmentGroups: [],
        dataFiles: [],
        biochemistryIndicators: [],
        roundResults: [
          {
            round: 3,
            summary: 'Innovation round',
            phase: 'innovation',
            pathologyFindings: [],
            bioinformaticsFindings: ['innovation finding'],
            biochemistryFindings: [],
            literatureFindings: [],
            crossModalCandidates: [],
            integrationCandidates: [],
            roundStatus: 'completed',
            evidenceStatus: 'substantive',
            evidenceExecutionIds: ['exec-innovation'],
            evidenceNotes: [],
            evidenceGateFindings: [],
            lineageId: 'main',
            directionKey: 'innovation-axis',
            resultIdentityKey: 'main:innovation:3:innovation-axis',
          },
        ],
      })
    })

    await act(async () => {
      result.current.connect()
    })

    act(() => {
      wsInstances[wsInstances.length - 1].onmessage?.({
        data: JSON.stringify({
          type: 'round_summary',
          agent: 'pi',
          content: {
            round: 3,
            phase: 'integration',
            summary: 'Integration round',
            integration_report_summary: 'Shared mechanism summary',
            integration_candidates: [{ direction: 'New axis X', score: 0.85 }],
            round_status: 'partial',
            evidence_status: 'substantive',
            direction_coverage_eligible: false,
            bioinformatics_evidence_verified: false,
            evidence_execution_ids: ['exec-integration'],
            evidence_notes: ['A mandatory gate was force-released.'],
            evidence_gate_findings: [{ gate: 'analysis-unit', forced_release: true }],
            lineage_id: 'restore-snapshot-4',
            branch_label: 'Restored branch',
            restored_from_snapshot_id: 'snapshot-4',
            result_identity_key: 'restore-snapshot-4:integration:3:',
          },
        }),
      } as MessageEvent)
    })

    const rounds = useAppStore.getState().currentSession?.roundResults || []
    expect(rounds).toHaveLength(2)
    expect(rounds.map((round) => round.phase)).toEqual(['innovation', 'integration'])
    expect(rounds[1].integrationReportSummary).toBe('Shared mechanism summary')
    expect(rounds[1].integrationCandidates[0].score).toBe(0.85)
    expect(rounds[1]).toMatchObject({
      roundStatus: 'partial',
      evidenceStatus: 'substantive',
      directionCoverageEligible: false,
      bioinformaticsEvidenceVerified: false,
      evidenceExecutionIds: ['exec-integration'],
      evidenceNotes: ['A mandatory gate was force-released.'],
      evidenceGateFindings: [{ gate: 'analysis-unit', forced_release: true }],
      lineageId: 'restore-snapshot-4',
      branchLabel: 'Restored branch',
      restoredFromSnapshotId: 'snapshot-4',
      resultIdentityKey: 'restore-snapshot-4:integration:3:',
    })
  })

  it('does not overwrite a main result with a same-round restored-branch result', async () => {
    const { result } = renderHook(() => useWebSocket())

    act(() => {
      useAppStore.getState().setCurrentSession({
        id: 'session-branch-identity',
        name: 'Branch identity session',
        status: 'running' as const,
        currentRound: 4,
        createdAt: new Date().toISOString(),
        instruction: '',
        treatmentGroups: [],
        dataFiles: [],
        biochemistryIndicators: [],
        roundResults: [{
          round: 4,
          summary: 'Main result',
          phase: 'innovation',
          pathologyFindings: [],
          bioinformaticsFindings: ['main finding'],
          biochemistryFindings: [],
          literatureFindings: [],
          crossModalCandidates: [],
          integrationCandidates: [],
          roundStatus: 'completed',
          evidenceStatus: 'substantive',
          evidenceExecutionIds: [],
          evidenceNotes: [],
          evidenceGateFindings: [],
          lineageId: 'main',
          directionKey: 'same-axis',
          resultIdentityKey: 'main:innovation:4:same-axis',
        }],
      })
    })

    await act(async () => {
      result.current.connect()
    })
    act(() => {
      wsInstances[wsInstances.length - 1].onmessage?.({
        data: JSON.stringify({
          type: 'round_summary',
          agent: 'pi',
          content: {
            round: 4,
            phase: 'innovation',
            direction_key: 'same-axis',
            lineage_id: 'restore-snapshot-4',
            branch_id: 'restore-snapshot-4',
            summary: 'Branch result',
            bioinformatics_findings: ['branch finding'],
          },
        }),
      } as MessageEvent)
    })

    const rounds = useAppStore.getState().currentSession?.roundResults || []
    expect(rounds).toHaveLength(2)
    expect(rounds.map((round) => round.resultIdentityKey)).toEqual([
      'main:innovation:4:same-axis',
      'restore-snapshot-4:innovation:4:same-axis',
    ])
    expect(rounds.map((round) => round.summary)).toEqual([
      'Main result',
      'Branch result',
    ])
  })

  it('uses lifecycle-aware reload for state-version conflicts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: { code: 'SESSION_GONE', message: 'deleted' },
    }), {
      status: 410,
      headers: { 'Content-Type': 'application/json' },
    })))
    useAppStore.getState().setCurrentSession({
      id: 'session-gone',
      name: 'Gone session',
      status: 'running',
      currentRound: 2,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    })
    useAppStore.getState().setStateVersion(5)
    const { result } = renderHook(() => useWebSocket())

    await act(async () => result.current.connect())
    act(() => {
      wsInstances[0].onmessage?.({
        data: JSON.stringify({ type: 'state_sync', version: 4 }),
      } as MessageEvent)
    })

    await vi.waitFor(() => {
      expect(useAppStore.getState().currentSession).toBeNull()
    })
  })

  it('drops a delayed conflict reload after the active session changes', async () => {
    let resolveFetch!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(new Promise<Response>((resolve) => {
      resolveFetch = resolve
    })))
    const sessionA = {
      id: 'session-reload-A',
      name: 'Session A',
      status: 'running' as const,
      currentRound: 4,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }
    useAppStore.getState().setCurrentSession(sessionA)
    useAppStore.getState().setStateVersion(10)
    const { result } = renderHook(() => useWebSocket())
    await act(async () => result.current.connect())

    act(() => {
      wsInstances[0].onmessage?.({
        data: JSON.stringify({ type: 'state_sync', version: 9 }),
      } as MessageEvent)
      useAppStore.getState().setCurrentSession({ ...sessionA, id: 'session-reload-B', name: 'Session B' })
      resolveFetch(new Response(JSON.stringify({
        id: 'session-reload-A',
        name: 'Stale A response',
        status: 'completed',
        current_round: 99,
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }))
    })

    await vi.waitFor(() => {
      expect(useAppStore.getState().currentSession?.id).toBe('session-reload-B')
      expect(useAppStore.getState().currentSession?.currentRound).toBe(4)
    })
  })
})
