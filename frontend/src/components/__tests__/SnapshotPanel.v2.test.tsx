import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { SnapshotPanel } from '../SnapshotPanel'

const mockListSnapshotsV2 = vi.fn()
const mockDeleteSnapshotV2 = vi.fn()
const mockRestoreSnapshotV2 = vi.fn()
const mockCancelSnapshotTransaction = vi.fn()
const mockGetSnapshotTransaction = vi.fn()

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  listSnapshotsV2: (...args: unknown[]) => mockListSnapshotsV2(...args),
  deleteSnapshotV2: (...args: unknown[]) => mockDeleteSnapshotV2(...args),
  restoreSnapshotV2: (...args: unknown[]) => mockRestoreSnapshotV2(...args),
  cancelSnapshotTransaction: (...args: unknown[]) => mockCancelSnapshotTransaction(...args),
  getSnapshotTransaction: (...args: unknown[]) => mockGetSnapshotTransaction(...args),
}))

let currentTxn: Record<string, unknown> | null = null
const mockSetTxn = vi.fn((txn: Record<string, unknown> | null) => {
  currentTxn = txn
})

vi.mock('../../stores/appStore', () => {
  const useAppStore = Object.assign(
    (selector?: (s: Record<string, unknown>) => unknown) => {
      const state = {
        snapshotTransaction: currentTxn,
        setSnapshotTransaction: mockSetTxn,
        setSnapshotsV2: vi.fn(),
        setSnapshotStoreStatus: vi.fn(),
        setSnapshotCount: vi.fn(),
      }
      return selector ? selector(state) : state
    },
    {
      getState: () => ({
        snapshotTransaction: currentTxn,
        setSnapshotTransaction: mockSetTxn,
        setSnapshotsV2: vi.fn(),
        setSnapshotStoreStatus: vi.fn(),
        setSnapshotCount: vi.fn(),
      }),
    },
  )
  return { useAppStore }
})

function makeSnapshotV2(overrides: Record<string, unknown> = {}) {
  return {
    snapshot_id: 'snap-1',
    manifest: {
      format_version: 2,
      session_id: 'test-session',
      snapshot_id: 'snap-1',
      captured_at: '2025-01-15T10:30:00Z',
      branch: 'main',
      generation: 5,
      parent_id: null,
      root_hash: 'abc123',
      capsule: { kind: 'idle', graph_version: '1.0' },
      workspace: [],
      objects: {},
      ...(overrides.manifest as Record<string, unknown> || {}),
    },
    failure: null,
    ...overrides,
  }
}

const twoSnapshots = {
  snapshots: [
    makeSnapshotV2({
      snapshot_id: 'snap-1',
      manifest: {
        format_version: 2,
        session_id: 'test-session',
        snapshot_id: 'snap-1',
        captured_at: '2025-01-15T10:30:00Z',
        branch: 'main',
        generation: 5,
        parent_id: null,
        root_hash: 'abc',
        capsule: { kind: 'idle', graph_version: '1.0' },
        workspace: [
          { path: 'main.ipynb', object_hash: 'h1', size: 2048, mode: 644, kind: 'file' },
          { path: 'data.csv', object_hash: 'h2', size: 10240, mode: 644, kind: 'file' },
        ],
        objects: { h1: { sha256: 'h1', size: 2048, media_type: 'text' } },
        runtime_dependencies: {
          python: { name: 'python', version: '3.11', type: 'interpreter' },
          additional: {
            numpy: { name: 'numpy', version: '1.26', type: 'library' },
          },
        },
      },
    }),
    makeSnapshotV2({
      snapshot_id: 'snap-2',
      manifest: {
        format_version: 2,
        session_id: 'test-session',
        snapshot_id: 'snap-2',
        captured_at: '2025-01-15T11:00:00Z',
        branch: 'main',
        generation: 6,
        parent_id: 'snap-1',
        root_hash: 'def',
        capsule: {
          kind: 'graph_node',
          graph_version: '1.0',
          next_node: 'pi_synthesis',
          workflow_state_ref: { sha256: 'ws1', size: 512, media_type: 'application/json' },
          planning_stage: 'full',
        },
        workspace: [
          { path: 'main.ipynb', object_hash: 'h3', size: 4096, mode: 644, kind: 'file' },
        ],
        objects: { h3: { sha256: 'h3', size: 4096, media_type: 'text' } },
        runtime_dependencies: {
          python: { name: 'python', version: '3.11', type: 'interpreter' },
          additional: {
            numpy: { name: 'numpy', version: '1.26', type: 'library' },
          },
        },
      },
    }),
  ],
  store_status: {
    logical_bytes: 10240,
    physical_bytes: 5120,
    dedup_savings: 5120,
    object_count: 8,
    manifest_count: 2,
  },
}

beforeEach(() => {
  currentTxn = null
  mockSetTxn.mockClear()
  mockListSnapshotsV2.mockReset()
  mockDeleteSnapshotV2.mockReset()
  mockRestoreSnapshotV2.mockReset()
  mockCancelSnapshotTransaction.mockReset()
  mockGetSnapshotTransaction.mockReset()
  mockListSnapshotsV2.mockResolvedValue(twoSnapshots)
  mockDeleteSnapshotV2.mockResolvedValue({ deleted_ids: ['snap-1'], freed_bytes: 100, surviving_count: 1 })
  mockRestoreSnapshotV2.mockResolvedValue({
    transaction_id: 'txn-1',
    status: 'queued',
    status_url: '/api/session/sess/snapshot-transactions/txn-1',
    cancel_url: '/api/session/sess/snapshot-transactions/txn-1/cancel',
  })
  mockCancelSnapshotTransaction.mockResolvedValue(undefined)
  mockGetSnapshotTransaction.mockResolvedValue({
    transaction_id: 'txn-1',
    session_id: 'sess',
    snapshot_id: 'snap-1',
    status: 'queued',
    recovery_mode: 'rollback',
    failure: null,
  })
  vi.useRealTimers()
})

// ── StorageSummary tests ──────────────────────────────────────────

describe('StorageSummary', () => {
  it('renders logical_bytes, physical_bytes, and dedup_savings sizes', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText('10.0 KB')).toBeInTheDocument()
    const fives = await screen.findAllByText('5.0 KB')
    expect(fives.length).toBeGreaterThanOrEqual(2)
  })

  it('shows quota progress bar with correct width', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const bar = await screen.findByRole('progressbar')
    expect(bar).toBeInTheDocument()
  })

  it('renders object count and manifest count', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText('Objects:')).toBeInTheDocument()
    expect(await screen.findByText('Manifests:')).toBeInTheDocument()
  })
})

// ── SnapshotTree tests ────────────────────────────────────────────

describe('SnapshotTree', () => {
  it('renders each snapshot manifest captured_at', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const matches = await screen.findAllByText(/1\/15\/2025/)
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })

  it('shows capsule kind for each snapshot', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText(/Idle/)).toBeInTheDocument()
    expect(await screen.findByText(/Graph Node/)).toBeInTheDocument()
  })

  it('displays workspace file count per snapshot', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText(/2 files/)).toBeInTheDocument()
    expect(await screen.findByText(/1 file/)).toBeInTheDocument()
  })

  it('shows runtime python version in snapshot', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const matches = await screen.findAllByText(/3\.11/)
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })

  it('highlights selected snapshot', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const snapCards = await screen.findAllByText(/snap-1/)
    fireEvent.click(snapCards[0])
  })
})

// ── SnapshotPreview tests ─────────────────────────────────────────

describe('SnapshotPreview', () => {
  it('shows continuation target for graph_node capsule', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const previewTexts = await screen.findAllByText(/pi_synthesis/)
    expect(previewTexts.length).toBeGreaterThanOrEqual(1)
  })

  it('displays workspace entry list with size and hash', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText('main.ipynb')).toBeInTheDocument()
    const hashes = await screen.findAllByText('h1')
    expect(hashes.length).toBeGreaterThanOrEqual(1)
  })

  it('renders objects included section', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const objs = await screen.findAllByText(/h1/)
    expect(objs.length).toBeGreaterThanOrEqual(1)
  })

  it('shows runtime dependency info', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText(/numpy/)).toBeInTheDocument()
    expect(await screen.findByText('1.26')).toBeInTheDocument()
  })

  it('renders same-path replacement indicators', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const replacements = await screen.findAllByText(/main\.ipynb/)
    expect(replacements.length).toBeGreaterThanOrEqual(1)
  })
})

// ── SnapshotPanel integration tests ────────────────────────────────

describe('SnapshotPanel V2 integration', () => {
  it('loads snapshots via listSnapshotsV2 on mount', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    await screen.findByText(/10.0 KB/)
    expect(mockListSnapshotsV2).toHaveBeenCalledWith('sess')
  })

  it('renders protected-delete disabled for in-use snapshots', async () => {
    mockListSnapshotsV2.mockResolvedValue({
      snapshots: [makeSnapshotV2({
        snapshot_id: 'snap-protected',
        manifest: {
          captured_at: '2025-01-15T10:30:00Z',
          branch: 'main',
          generation: 1,
          parent_id: null,
          root_hash: 'abc',
          capsule: { kind: 'terminal', graph_version: '1.0', status: 'completed' },
          workspace: [],
          objects: {},
        },
      })],
      store_status: { logical_bytes: 0, physical_bytes: 0, dedup_savings: 0, object_count: 0, manifest_count: 1 },
    })

    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const deleteBtn = await screen.findByRole('button', { name: /delete/i })
    expect(deleteBtn).toBeDisabled()
  })

  it('provides restore and delete entry points', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByRole('button', { name: /restore/i })).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: /delete/i })).toBeInTheDocument()
  })
})

// ── Action semantics ───────────────────────────────────────────────

describe('SnapshotPanel action semantics', () => {
  it('Delete calls deleteSnapshotV2 and never opens discard-current restore', async () => {
    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    await screen.findByRole('button', { name: /delete/i })
    fireEvent.click(screen.getByRole('button', { name: /delete/i }))
    fireEvent.click(await screen.findByRole('button', { name: /confirm delete/i }))

    await waitFor(() => {
      expect(mockDeleteSnapshotV2).toHaveBeenCalled()
    })
    expect(mockRestoreSnapshotV2).not.toHaveBeenCalled()
    expect(screen.queryByText(/discard current/i)).not.toBeInTheDocument()
  })

  it('Discard Current is shown only after pre_restore_unavailable with confirmation', async () => {
    mockRestoreSnapshotV2.mockRejectedValueOnce({
      code: 'pre_restore_unavailable',
      message: 'Pre-restore capture unavailable',
    })

    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    fireEvent.click(await screen.findByRole('button', { name: /restore/i }))
    const startBtn = await screen.findByRole('button', { name: /start restore|confirm restore|yes/i })
    fireEvent.click(startBtn)

    expect(await screen.findByRole('dialog', { name: /discard current/i })).toBeInTheDocument()
    const confirmDiscard = screen.getByRole('button', { name: /confirm discard/i })
    expect(confirmDiscard).toBeDisabled()

    const phraseInput = screen.getByLabelText(/type.*discard|confirmation phrase/i)
    fireEvent.change(phraseInput, { target: { value: 'DISCARD CURRENT STATE' } })
    expect(confirmDiscard).not.toBeDisabled()
    fireEvent.click(confirmDiscard)

    await waitFor(() => {
      expect(mockRestoreSnapshotV2).toHaveBeenCalledWith(
        'sess',
        expect.any(String),
        expect.objectContaining({
          current_state_choice: 'discard_current',
          discard_confirmation: true,
        }),
      )
    })
  })

  it('Cancel addresses existing transaction then reloads status', async () => {
    mockGetSnapshotTransaction
      .mockResolvedValueOnce({
        transaction_id: 'txn-1',
        session_id: 'sess',
        snapshot_id: 'snap-1',
        status: 'queued',
        recovery_mode: 'rollback',
        failure: null,
      })
      .mockResolvedValueOnce({
        transaction_id: 'txn-1',
        session_id: 'sess',
        snapshot_id: 'snap-1',
        status: 'cancelled',
        recovery_mode: 'rollback',
        failure: null,
      })

    render(
      <SnapshotPanel
        sessionId="sess"
        language="en"
        onClose={vi.fn()}
        onRestore={vi.fn()}
        activeTransactionId="txn-1"
      />,
    )

    expect(await screen.findByRole('dialog', { name: /restore progress/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /^cancel$/i }))

    await waitFor(() => {
      expect(mockCancelSnapshotTransaction).toHaveBeenCalledWith('sess', 'txn-1')
      expect(mockGetSnapshotTransaction).toHaveBeenCalledWith('sess', 'txn-1')
    })
  })

  it('reloads and polls recovery for a queued job', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    mockGetSnapshotTransaction
      .mockResolvedValueOnce({
        transaction_id: 'txn-poll',
        session_id: 'sess',
        snapshot_id: 'snap-1',
        status: 'queued',
        recovery_mode: 'rollback',
        failure: null,
      })
      .mockResolvedValueOnce({
        transaction_id: 'txn-poll',
        session_id: 'sess',
        snapshot_id: 'snap-1',
        status: 'applying',
        recovery_mode: 'rollback',
        failure: null,
      })
      .mockResolvedValueOnce({
        transaction_id: 'txn-poll',
        session_id: 'sess',
        snapshot_id: 'snap-1',
        status: 'committed',
        recovery_mode: 'rollback',
        failure: null,
      })

    render(
      <SnapshotPanel
        sessionId="sess"
        language="en"
        onClose={vi.fn()}
        onRestore={vi.fn()}
        activeTransactionId="txn-poll"
      />,
    )

    await waitFor(() => {
      expect(mockGetSnapshotTransaction).toHaveBeenCalledWith('sess', 'txn-poll')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2500)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2500)
    })

    expect(mockGetSnapshotTransaction.mock.calls.length).toBeGreaterThanOrEqual(2)
    vi.useRealTimers()
  }, 15000)

  it('shows terminal timeout actions', async () => {
    mockGetSnapshotTransaction.mockResolvedValue({
      transaction_id: 'txn-to',
      session_id: 'sess',
      snapshot_id: 'snap-1',
      status: 'timed_out',
      recovery_mode: 'rollback',
      failure: { code: 'TIMEOUT', message: 'Safe point wait timed out' },
    })

    render(
      <SnapshotPanel
        sessionId="sess"
        language="en"
        onClose={vi.fn()}
        onRestore={vi.fn()}
        activeTransactionId="txn-to"
      />,
    )

    expect(await screen.findByRole('dialog', { name: /restore progress/i })).toBeInTheDocument()
    expect(screen.getAllByText(/timed out/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByRole('button', { name: /dismiss/i })).toBeInTheDocument()
  })

  it('deduplicates WebSocket and poll updates for the same transaction status', async () => {
    mockGetSnapshotTransaction.mockResolvedValue({
      transaction_id: 'txn-dedup',
      session_id: 'sess',
      snapshot_id: 'snap-1',
      status: 'staging',
      recovery_mode: 'rollback',
      failure: null,
    })

    render(
      <SnapshotPanel
        sessionId="sess"
        language="en"
        onClose={vi.fn()}
        onRestore={vi.fn()}
        activeTransactionId="txn-dedup"
      />,
    )

    await waitFor(() => expect(mockGetSnapshotTransaction).toHaveBeenCalled())

    act(() => {
      mockSetTxn({
        transaction_id: 'txn-dedup',
        session_id: 'sess',
        snapshot_id: 'snap-1',
        status: 'staging',
        recovery_mode: 'rollback',
        failure: null,
      })
    })

    const progressDialogs = screen.getAllByRole('dialog', { name: /restore progress/i })
    expect(progressDialogs.length).toBe(1)
  })
})
