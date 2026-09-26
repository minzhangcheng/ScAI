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
})
