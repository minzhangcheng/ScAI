/**
 * The snapshot panel advises a cleanup before the quota refuses a capture.
 *
 * The threshold is the backend's (`quota_warning`), so these cases pin that the
 * panel acts on that flag rather than re-deriving one, and that a response
 * without the quota trio renders nothing rather than "NaN%".
 */
import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { SnapshotPanel } from '../SnapshotPanel'

const mockListSnapshotsV2 = vi.fn()

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  listSnapshotsV2: (...args: unknown[]) => mockListSnapshotsV2(...args),
  deleteSnapshotV2: vi.fn(),
  restoreSnapshotV2: vi.fn(),
  cancelSnapshotTransaction: vi.fn(),
  getSnapshotTransaction: vi.fn(),
}))

vi.mock('../../stores/appStore', () => {
  const state = {
    snapshotTransaction: null,
    setSnapshotTransaction: vi.fn(),
    setSnapshotsV2: vi.fn(),
    setSnapshotStoreStatus: vi.fn(),
    setSnapshotCount: vi.fn(),
  }
  const useAppStore = Object.assign(
    (selector?: (s: Record<string, unknown>) => unknown) =>
      (selector ? selector(state) : state),
    { getState: () => state },
  )
  return { useAppStore }
})

const GIB = 1024 ** 3

function oneSnapshot(storeStatus: Record<string, unknown>) {
  return {
    snapshots: [
      {
        snapshot_id: 'snap-1',
        manifest: {
          format_version: 2,
          session_id: 'sess',
          snapshot_id: 'snap-1',
          captured_at: '2026-07-30T00:00:00Z',
          branch: 'main',
          generation: 1,
          parent_id: null,
          root_hash: 'abc',
          capsule: { kind: 'idle', graph_version: '2.0' },
          workspace: [],
          objects: {},
          runtime: { status: 'complete', adata_refs: {}, residual_ref: null },
        },
        failure: null,
      },
    ],
    store_status: storeStatus,
  }
}

function baseStatus(over: Record<string, unknown> = {}) {
  return {
    logical_bytes: 10 * GIB,
    physical_bytes: 10 * GIB,
    dedup_savings: 0,
    object_count: 1,
    manifest_count: 1,
    ...over,
  }
}

beforeEach(() => {
  mockListSnapshotsV2.mockReset()
})

describe('snapshot quota advisory', () => {
  it('advises a cleanup when the backend flags the quota', async () => {
    mockListSnapshotsV2.mockResolvedValue(oneSnapshot(baseStatus({
      physical_bytes: 460 * GIB,
      quota_bytes: 500 * GIB,
      quota_usage_ratio: 0.92,
      quota_warning: true,
    })))

    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('92%')
    expect(alert.textContent).toContain('Delete snapshots you no longer need')
  })

  it('advises in Chinese when the panel is in Chinese', async () => {
    mockListSnapshotsV2.mockResolvedValue(oneSnapshot(baseStatus({
      physical_bytes: 460 * GIB,
      quota_bytes: 500 * GIB,
      quota_usage_ratio: 0.92,
      quota_warning: true,
    })))

    render(<SnapshotPanel sessionId="sess" language="zh" onClose={vi.fn()} onRestore={vi.fn()} />)

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('92%')
    expect(alert.textContent).toContain('请删除不再需要的快照')
  })

  it('stays quiet below the threshold', async () => {
    mockListSnapshotsV2.mockResolvedValue(oneSnapshot(baseStatus({
      physical_bytes: 100 * GIB,
      quota_bytes: 500 * GIB,
      quota_usage_ratio: 0.2,
      quota_warning: false,
    })))

    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    await waitFor(() => expect(mockListSnapshotsV2).toHaveBeenCalled())
    await screen.findByText('Objects:')
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('renders nothing when the response carries no quota fields', async () => {
    // An older backend, or a fixture predating the trio: the panel must not
    // invent a percentage out of undefined.
    mockListSnapshotsV2.mockResolvedValue(oneSnapshot(baseStatus()))

    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    await waitFor(() => expect(mockListSnapshotsV2).toHaveBeenCalled())
    await screen.findByText('Objects:')
    expect(screen.queryByRole('alert')).toBeNull()
    expect(document.body.textContent).not.toContain('NaN')
  })
})

describe('experimental marking', () => {
  // The panel is only reachable on an instance that opted in, and the feature is
  // deliberately absent from the user guide — so the screen itself has to say it
  // is experimental. These pin that it cannot be dropped silently.
  it('marks the panel experimental in English', async () => {
    mockListSnapshotsV2.mockResolvedValue(oneSnapshot(baseStatus()))

    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText('Experimental')).toBeInTheDocument()
    expect(document.body.textContent).toContain('in development and not yet stable')
  })

  it('marks the panel experimental in Chinese', async () => {
    mockListSnapshotsV2.mockResolvedValue(oneSnapshot(baseStatus()))

    render(<SnapshotPanel sessionId="sess" language="zh" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText('实验性')).toBeInTheDocument()
    expect(document.body.textContent).toContain('尚不稳定')
  })

  it('marks it even before the listing resolves', async () => {
    // The caveat sits outside the loading branch on purpose: a slow list must
    // not leave the panel looking like a settled feature.
    mockListSnapshotsV2.mockReturnValue(new Promise(() => {}))

    render(<SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />)

    expect(await screen.findByText('Experimental')).toBeInTheDocument()
  })
})
