import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SnapshotList } from '../runtime/SnapshotList'

/**
 * The decision-snapshot panel. Part B spec 4.9.
 *
 * What it has to get right is mostly what it does *not* show: nothing on an
 * installation that takes no snapshots, details before the user expands the
 * summary, and no way at all to create one.
 *
 * Mocked at the module rather than spied on the export: the component binds
 * its imports at load, so a spy on the namespace would leave it calling the
 * real thing.
 */

const listSnapshots = vi.fn()
const restoreSnapshot = vi.fn()
const deleteSnapshot = vi.fn()
const getRestoreStatus = vi.fn()
const snapshotListingChangedListeners = new Set<(sessionId: string) => void>()

function notifySnapshotListingChanged(sessionId: string) {
  for (const listener of snapshotListingChangedListeners) listener(sessionId)
}

vi.mock('../../hooks/api/snapshots', () => ({
  listSnapshots: (...args: unknown[]) => listSnapshots(...args),
  restoreSnapshot: (...args: unknown[]) => restoreSnapshot(...args),
  deleteSnapshot: (...args: unknown[]) => deleteSnapshot(...args),
  getRestoreStatus: (...args: unknown[]) => getRestoreStatus(...args),
  subscribeSnapshotListingChanged: (listener: (sessionId: string) => void) => {
    snapshotListingChangedListeners.add(listener)
    return () => snapshotListingChangedListeners.delete(listener)
  },
}))

const ROW = {
  snapshot_id: 'r001-annotation-g57',
  round: 1,
  gate: 'annotation_confirmation',
  runtime_revision: 57,
  created_at: '2026-09-08T10:00:00+00:00',
  bytes_total: 2_147_483_648,
  capture_seconds: 41.2,
  retention_rank: 1,
  deletion_order: 2,
  superseded: false,
  saved_names: 271,
  skipped_count: 1,
  executor_checkpoint_present: true,
  scai_version: '0.55.9',
}

function listing(overrides: Record<string, unknown> = {}) {
  return {
    enabled: true,
    snapshots: [ROW],
    failed: [],
    restored_sessions: {},
    bytes_total: 2_147_483_648,
    ...overrides,
  }
}

async function expandSnapshots() {
  fireEvent.click(await screen.findByRole('button', { name: /Decision snapshots/ }))
}

beforeEach(() => {
  listSnapshots.mockReset()
  restoreSnapshot.mockReset()
  deleteSnapshot.mockReset()
  getRestoreStatus.mockReset()
  snapshotListingChangedListeners.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('SnapshotList', () => {
  it('renders nothing when the installation takes no snapshots', async () => {
    const interval = vi.spyOn(window, 'setInterval')
    listSnapshots.mockResolvedValue(
      listing({ enabled: false, snapshots: [], failed: [] }),
    )
    const { container } = render(<SnapshotList sessionId="s1" />)
    await act(async () => {
      await Promise.resolve()
    })
    expect(listSnapshots).toHaveBeenCalledTimes(1)
    expect(container).toBeEmptyDOMElement()
    expect(interval).not.toHaveBeenCalled()
  })

  it('keeps the zero summary expandable before a session has reached a decision', async () => {
    listSnapshots.mockResolvedValue(listing({ snapshots: [], failed: [], bytes_total: 0 }))
    render(<SnapshotList sessionId="s1" />)
    await waitFor(() => expect(listSnapshots).toHaveBeenCalled())
    expect(await screen.findByText('0 snapshots with 0 GB on disk')).toBeInTheDocument()
    await expandSnapshots()
    expect(screen.getByText('No decision snapshots have been captured yet.')).toBeInTheDocument()
  })

  it('names the decision each snapshot precedes, not its identifier', async () => {
    listSnapshots.mockResolvedValue(listing())
    render(<SnapshotList sessionId="s1" />)
    await expandSnapshots()
    expect(
      await screen.findByText('before confirming the cell annotation'),
    ).toBeInTheDocument()
    expect(screen.getByText('Round 1')).toBeInTheDocument()
    expect(screen.queryByText(/value(?:s)? not saved/)).not.toBeInTheDocument()
    // Exactly the row's own size. The header carries a session total that
    // reads the same at this scale, and matching both would not say which one
    // the row shows.
    expect(screen.getByText('2.0 GB')).toBeInTheDocument()
  })

  it('toggles the details from anywhere on the summary row', async () => {
    listSnapshots.mockResolvedValue(listing())
    render(<SnapshotList sessionId="s1" />)

    const summary = await screen.findByRole('button', { name: /Decision snapshots/ })
    expect(summary).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('Round 1')).not.toBeInTheDocument()

    fireEvent.click(summary)
    expect(summary).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('Round 1')).toBeInTheDocument()

    fireEvent.click(summary)
    expect(summary).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('Round 1')).not.toBeInTheDocument()
  })

  it('reports what the snapshots occupy, counting a shared dump once', async () => {
    listSnapshots.mockResolvedValue(listing({
      snapshots: [ROW, { ...ROW, snapshot_id: 'r003-round-g99', round: 3 }],
      // Two rows declaring 2 GB each, occupying 2 GB between them: the dump is
      // one file, shared by a hardlink.
      bytes_total: 2_147_483_648,
    }))
    render(<SnapshotList sessionId="s1" />)
    expect(await screen.findByText('2 snapshots with 2.0 GB on disk')).toBeInTheDocument()
  })

  it('shows the snapshot importance rank without deletion-order wording', async () => {
    listSnapshots.mockResolvedValue(listing())
    const { rerender } = render(<SnapshotList sessionId="s1" />)
    await expandSnapshots()
    expect(screen.getByText('Importance rank: 1/4 (1 is highest)')).toBeInTheDocument()
    expect(screen.queryByText(/space runs short/)).not.toBeInTheDocument()

    rerender(<SnapshotList sessionId="s1" isZh />)
    expect(screen.getByText('重要性排序：1/4（1 为最高）')).toBeInTheDocument()
  })

  it('refreshes after capture invalidation without polling the listing', async () => {
    const interval = vi.spyOn(window, 'setInterval')
    listSnapshots
      .mockResolvedValueOnce(listing({ snapshots: [], failed: [], bytes_total: 0 }))
      .mockResolvedValue(listing())

    render(<SnapshotList sessionId="s1" />)
    await act(async () => {
      await Promise.resolve()
    })
    expect(screen.getByText('0 snapshots with 0 GB on disk')).toBeInTheDocument()

    await act(async () => {
      notifySnapshotListingChanged('s1')
      await Promise.resolve()
    })

    expect(listSnapshots).toHaveBeenCalledTimes(2)
    expect(screen.getByText('1 snapshot with 2.0 GB on disk')).toBeInTheDocument()
    expect(interval).not.toHaveBeenCalled()
  })

  it('keeps the last listing and operation message when a refresh fails', async () => {
    listSnapshots
      .mockResolvedValueOnce(listing())
      .mockRejectedValueOnce(new Error('snapshot volume is temporarily unavailable'))
    restoreSnapshot.mockReturnValueOnce(new Promise(() => {}))

    render(<SnapshotList sessionId="s1" />)
    expect(await screen.findByText('1 snapshot with 2.0 GB on disk')).toBeInTheDocument()
    await expandSnapshots()
    fireEvent.click(screen.getByRole('button', {
      name: /Restore into a new session/,
    }))
    expect(await screen.findByText('Creating the new session…')).toBeInTheDocument()

    act(() => {
      notifySnapshotListingChanged('s1')
    })

    await waitFor(() => expect(listSnapshots).toHaveBeenCalledTimes(2))
    expect(screen.getByText('1 snapshot with 2.0 GB on disk')).toBeInTheDocument()
    expect(screen.getByText('Creating the new session…')).toBeInTheDocument()
  })

  it('offers no way to create a snapshot', async () => {
    listSnapshots.mockResolvedValue(listing())
    render(<SnapshotList sessionId="s1" />)
    await expandSnapshots()
    await screen.findByText('Round 1')
    for (const button of screen.getAllByRole('button')) {
      expect(button.textContent ?? '').not.toMatch(/create|new snapshot|新建|创建/i)
    }
  })

  it('says restoring leaves this session untouched', async () => {
    listSnapshots.mockResolvedValue(listing())
    render(<SnapshotList sessionId="s1" />)
    expect(await screen.findByText(/leaves this one untouched/)).toBeInTheDocument()
    await expandSnapshots()
    expect(
      screen.getByText(/carries no reports, and may be missing values/),
    ).toBeInTheDocument()
  })

  it('opens the session a restore produced', async () => {
    listSnapshots.mockResolvedValue(listing())
    restoreSnapshot.mockResolvedValue({ job_id: 'job-1' })
    getRestoreStatus.mockResolvedValue({
      status: 'completed', new_session_id: 'fork-1',
    })
    const onOpenSession = vi.fn()
    render(<SnapshotList sessionId="s1" onOpenSession={onOpenSession} />)

    await expandSnapshots()
    fireEvent.click(await screen.findByRole('button', {
      name: /Restore into a new session/,
    }))
    await waitFor(() => expect(onOpenSession).toHaveBeenCalledWith('fork-1'))
    expect(restoreSnapshot).toHaveBeenCalledWith('s1', ROW.snapshot_id)
  })

  it('reports a refused restore instead of silently doing nothing', async () => {
    listSnapshots.mockResolvedValue(listing())
    restoreSnapshot.mockResolvedValue({ job_id: 'job-1' })
    getRestoreStatus.mockResolvedValue({
      status: 'failed', error: 'the bundle does not match its manifest',
    })
    render(<SnapshotList sessionId="s1" />)

    await expandSnapshots()
    fireEvent.click(await screen.findByRole('button', {
      name: /Restore into a new session/,
    }))
    expect(
      await screen.findByText(/the bundle does not match its manifest/),
    ).toBeInTheDocument()
  })

  it('shows a failed capture even when there are no snapshots', async () => {
    listSnapshots.mockResolvedValue(listing({
      snapshots: [],
      bytes_total: 0,
      failed: [{
        snapshot_id: 'r002-round-g300', round: 2, gate: 'round_wait',
        reason: 'stall', at: '2026-09-08T11:00:00+00:00',
      }],
    }))
    render(<SnapshotList sessionId="s1" />)
    expect(
      await screen.findByText('0 snapshots with 0 GB on disk'),
    ).toBeInTheDocument()
    await expandSnapshots()
    expect(
      screen.getByText(/Round 2: this decision has no snapshot \(stall\)/),
    ).toBeInTheDocument()
  })

  it('marks a superseded capture without hiding it', async () => {
    listSnapshots.mockResolvedValue(listing({
      snapshots: [{ ...ROW, superseded: true }],
    }))
    render(<SnapshotList sessionId="s1" />)
    await expandSnapshots()
    expect(await screen.findByText('superseded')).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: /Restore into a new session/ }),
    ).toBeEnabled()
  })

  it('collapses stale details after deleting the last restore point', async () => {
    listSnapshots
      .mockResolvedValueOnce(listing())
      .mockResolvedValueOnce(listing({ snapshots: [], failed: [], bytes_total: 0 }))
    deleteSnapshot.mockResolvedValue(undefined)
    render(<SnapshotList sessionId="s1" />)

    await expandSnapshots()
    fireEvent.click(await screen.findByRole('button', { name: /Delete/ }))
    await waitFor(() =>
      expect(deleteSnapshot).toHaveBeenCalledWith('s1', ROW.snapshot_id),
    )
    await waitFor(() => expect(listSnapshots).toHaveBeenCalledTimes(2))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Decision snapshots/ })).toHaveAttribute(
        'aria-expanded',
        'false',
      )
    })
    expect(screen.queryByText(/carries no reports/)).not.toBeInTheDocument()

    await expandSnapshots()
    expect(screen.getByText('No decision snapshots have been captured yet.')).toBeInTheDocument()
  })
})
