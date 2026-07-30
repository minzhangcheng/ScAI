import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisPanel } from '../AnalysisPanel'
import { SnapshotPanel } from '../SnapshotPanel'
import { useWebSocket } from '../../hooks/api/useWebSocket'
import { useAppStore, type Session } from '../../stores/appStore'
import { normalizeSnapshotRecordV2 } from '../../utils/snapshotV2'

const mockApiFetch = vi.fn()
const mockGetSnapshotTransaction = vi.fn()
const mockListSnapshotsV2 = vi.fn()
const mockDeleteSnapshotV2 = vi.fn()
const mockRestoreSnapshotV2 = vi.fn()
const mockCancelSnapshotTransaction = vi.fn()

// `useWebSocket` is imported from its own module rather than the `hooks/useApi`
// facade on purpose: the facade is mocked here for AnalysisPanel/SnapshotPanel,
// and the WebSocket case needs the real message dispatcher.
vi.mock('../../hooks/useApi', () => ({
  apiFetch: (...args: unknown[]) => mockApiFetch(...args),
  getSnapshotTransaction: (...args: unknown[]) => mockGetSnapshotTransaction(...args),
  listSnapshotsV2: (...args: unknown[]) => mockListSnapshotsV2(...args),
  deleteSnapshotV2: (...args: unknown[]) => mockDeleteSnapshotV2(...args),
  restoreSnapshotV2: (...args: unknown[]) => mockRestoreSnapshotV2(...args),
  cancelSnapshotTransaction: (...args: unknown[]) => mockCancelSnapshotTransaction(...args),
  generateId: () => 'test-id',
  useConfig: () => ({
    checkStatus: vi.fn(),
    validateApiKey: vi.fn(),
    validateSavedConfig: vi.fn(),
    saveConfig: vi.fn(),
  }),
  useSession: () => ({
    startAnalysis: vi.fn(),
    resumeAnalysis: vi.fn(),
    continueRound: vi.fn(),
    pauseAnalysis: vi.fn(),
    stopAnalysis: vi.fn(),
  }),
  useUserConfig: () => ({
    getUserConfigStatus: vi.fn(),
    validateSavedConfig: vi.fn(),
  }),
}))

const SESSION_ID = 'sess-snap-edge2'
const EMPTY_LIST = {
  snapshots: [],
  store_status: null,
}

function makeSession(id: string = SESSION_ID): Session {
  return {
    id,
    name: 'Snapshot edge cases',
    status: 'waiting_input',
    currentRound: 1,
    analysisPhase: 'exploration',
    createdAt: '2026-07-29T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }
}

function seedStore(overrides: Record<string, unknown> = {}) {
  useAppStore.setState({
    currentSession: makeSession(),
    messages: [],
    language: 'en',
    wsConnected: true,
    snapshotFailure: null,
    snapshotV2Enabled: true,
    snapshotTransaction: null,
    runtimeRestoreStatus: 'complete',
    runtimeRestoreWarning: null,
    ...overrides,
  })
}

beforeEach(() => {
  mockApiFetch.mockReset()
  mockGetSnapshotTransaction.mockReset()
  mockListSnapshotsV2.mockReset()
  mockDeleteSnapshotV2.mockReset()
  mockRestoreSnapshotV2.mockReset()
  mockCancelSnapshotTransaction.mockReset()
  mockListSnapshotsV2.mockResolvedValue(EMPTY_LIST)
  seedStore()
})

afterEach(() => {
  vi.useRealTimers()
})

/**
 * F23 — the retry monitor's poll cap must actually be a cap.
 *
 * The monitor writes the transaction it is polling back into the store on every
 * tick, and the give-up branch writes it once more; if the effect that starts
 * the monitor depends on that record it re-runs, finds a nulled poll ref, and
 * restarts the loop with the attempt counter back at zero.
 */
describe('AnalysisPanel retry monitor poll cap', () => {
  it('stops polling permanently once the poll cap is reached', async () => {
    vi.useFakeTimers()
    const stalled = {
      transaction_id: 'txn-cap',
      session_id: SESSION_ID,
      snapshot_id: 'snap-1',
      status: 'waiting_safe_point' as const,
      recovery_mode: 'rollback' as const,
      failure: null,
    }
    mockGetSnapshotTransaction.mockResolvedValue(stalled)
    seedStore({ snapshotTransaction: stalled })

    render(<AnalysisPanel />)
    for (let i = 0; i < 200; i += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000)
      })
    }

    const settled = mockGetSnapshotTransaction.mock.calls.length
    expect(settled).toBeLessThanOrEqual(151)

    for (let i = 0; i < 20; i += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000)
      })
    }
    expect(mockGetSnapshotTransaction.mock.calls.length).toBe(settled)
  }, 30000)
})

/**
 * F25 — the retry auto-open must be edge-triggered.
 *
 * `setSnapshotTransaction` allocates a new object on every poll, so an
 * auto-open effect keyed on the record re-asserts `showSnapshotPanel` twice a
 * second and the panel's full-screen modal cannot be dismissed for the whole
 * restore.
 */
describe('AnalysisPanel retry auto-open', () => {
  it('keeps the Snapshot panel closed after the user closes it mid-restore', async () => {
    vi.useFakeTimers()
    const retrying = {
      transaction_id: 'txn-retry-open',
      session_id: SESSION_ID,
      snapshot_id: 'snap-1',
      status: 'waiting_safe_point' as const,
      recovery_mode: 'rollback' as const,
      failure: null,
      retry_of_transaction_id: 'txn-original',
    }
    mockGetSnapshotTransaction.mockResolvedValue(retrying)
    seedStore({ snapshotTransaction: retrying })

    render(<AnalysisPanel />)
    await act(async () => {
      await Promise.resolve()
    })
    expect(screen.getByRole('dialog', { name: /analysis snapshots/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /^close$/i }))
    expect(screen.queryByRole('dialog', { name: /analysis snapshots/i })).toBeNull()

    for (let i = 0; i < 3; i += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000)
      })
    }
    expect(screen.queryByRole('dialog', { name: /analysis snapshots/i })).toBeNull()
  }, 15000)
})

/**
 * F24 — a restore that commits while nothing is mounted to notice.
 *
 * The handler used to gate its refresh on `info.reconnect`, a field no backend
 * producer emits, so the client stayed on pre-restore state after the server
 * had already rewritten the session.
 */
describe('state_sync for a committed restore transaction', () => {
  const wsInstances: Array<{ onmessage: ((event: MessageEvent) => void) | null }> = []
  const mockFetch = vi.fn()

  class MockWebSocket {
    public readyState: number = WebSocket.OPEN
    public close = vi.fn()
    public url: string
    private _onopen: ((event: Event) => void) | null = null
    public onmessage: ((event: MessageEvent) => void) | null = null
    public onclose: ((event: CloseEvent) => void) | null = null
    public onerror: ((event: Event) => void) | null = null

    constructor(url: string) {
      this.url = url
      wsInstances.push(this as never)
    }

    set onopen(handler: ((event: Event) => void) | null) {
      this._onopen = handler
      if (handler) handler(new Event('open'))
    }

    get onopen(): ((event: Event) => void) | null {
      return this._onopen
    }
  }

  const committed = {
    transaction_id: 'restore-job-committed',
    session_id: SESSION_ID,
    snapshot_id: 'snap-1',
    status: 'committed',
    recovery_mode: 'rollback',
  }

  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket as never)
    mockFetch.mockReset()
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'completed', current_round: 4, state_version: 12 }),
    })
    vi.stubGlobal('fetch', mockFetch)
    wsInstances.length = 0
    seedStore({ stateVersion: 7 })
  })

  async function sendCommitted() {
    const ws = wsInstances[wsInstances.length - 1]
    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'state_sync',
          change_type: 'snapshot_transaction',
          snapshot_transaction: committed,
        }),
      } as MessageEvent)
    })
  }

  it('refetches session state when a restore commits', async () => {
    const { result } = renderHook(() => useWebSocket())
    await act(async () => {
      result.current.connect()
    })

    await sendCommitted()

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        `/api/session/${SESSION_ID}/load`,
        expect.anything(),
      )
    })
    await waitFor(() => {
      expect(useAppStore.getState().currentSession?.status).toBe('completed')
    })
  })

  it('does not refetch again when the same commit is replayed on reconnect', async () => {
    const { result } = renderHook(() => useWebSocket())
    await act(async () => {
      result.current.connect()
    })

    await sendCommitted()
    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(1))

    await sendCommitted()
    await act(async () => {
      await Promise.resolve()
    })
    expect(mockFetch).toHaveBeenCalledTimes(1)
  })
})

/**
 * F26 — the backend's integrity diagnostics must reach the user.
 *
 * `list_snapshots` deliberately keeps a manifest that fails verification in the
 * listing, and the normalizer parks its diagnostics on `record.failure` — which
 * nothing rendered, so a corrupt snapshot looked exactly like a healthy idle one
 * and was still offered for restore.
 */
describe('SnapshotPanel corrupt snapshot entry', () => {
  const corrupt = normalizeSnapshotRecordV2({
    snapshot_id: 'snap-bad',
    status: 'corrupt',
    integrity_status: 'corrupt',
    error_code: 'MANIFEST_HMAC_MISMATCH',
    error_message: 'HMAC signature mismatch for snapshot snap-bad',
    manifest: null,
  })

  it('marks the entry corrupt instead of rendering a synthetic idle snapshot', async () => {
    mockListSnapshotsV2.mockResolvedValue({ snapshots: [corrupt], store_status: null })

    render(
      <SnapshotPanel
        sessionId={SESSION_ID}
        language="en"
        onClose={vi.fn()}
        onRestore={vi.fn()}
      />,
    )

    expect(await screen.findByText(/MANIFEST_HMAC_MISMATCH/)).toBeInTheDocument()
    expect(screen.getAllByText(/HMAC signature mismatch/).length).toBeGreaterThan(0)
    expect(screen.queryByText('Idle')).toBeNull()
    expect(screen.queryByText('Invalid Date')).toBeNull()
  })

  it('refuses to offer Restore for a corrupt entry', async () => {
    mockListSnapshotsV2.mockResolvedValue({ snapshots: [corrupt], store_status: null })

    render(
      <SnapshotPanel
        sessionId={SESSION_ID}
        language="en"
        onClose={vi.fn()}
        onRestore={vi.fn()}
      />,
    )

    expect(await screen.findByTitle('Restore')).toBeDisabled()
  })
})

/**
 * F27 — a refused cancel must say so.
 *
 * `handleCancelTxn` swallowed every rejection into the console, so the button
 * simply re-enabled and the user had no way to learn the request failed.
 */
describe('SnapshotPanel restore cancel failure', () => {
  it('surfaces the reason a cancel was refused', async () => {
    mockGetSnapshotTransaction.mockResolvedValue({
      transaction_id: 'txn-cancel',
      session_id: SESSION_ID,
      snapshot_id: 'snap-1',
      status: 'waiting_safe_point',
      recovery_mode: 'rollback',
      failure: null,
    })
    mockCancelSnapshotTransaction.mockRejectedValue(
      Object.assign(new Error('Restore transaction not found'), {
        code: 'TRANSACTION_NOT_FOUND',
        status: 404,
      }),
    )

    render(
      <SnapshotPanel
        sessionId={SESSION_ID}
        language="en"
        onClose={vi.fn()}
        onRestore={vi.fn()}
        activeTransactionId="txn-cancel"
      />,
    )

    expect(await screen.findByRole('dialog', { name: /restore progress/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /^cancel$/i }))

    await waitFor(() => {
      expect(screen.getByText(/Restore transaction not found/)).toBeInTheDocument()
    })
    expect(screen.getByRole('button', { name: /^cancel$/i })).not.toBeDisabled()
  })
})
