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
})
