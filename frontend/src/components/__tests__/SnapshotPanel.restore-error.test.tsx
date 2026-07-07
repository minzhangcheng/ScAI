import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach, type Mock } from 'vitest'
import { SnapshotPanel } from '../SnapshotPanel'

vi.mock('../../hooks/useApi', () => ({ apiFetch: vi.fn() }))
import { apiFetch } from '../../hooks/useApi'

const mockFetch = apiFetch as unknown as Mock

const oneSnapshot = {
  success: true,
  data: {
    snapshots: [{
      id: 's1', name: 'S1', description: '', phase: 'pre_round_wait',
      round_number: 1, timestamp: new Date().toISOString(), size_bytes: 1,
      parent_id: null, branch_label: null, notebook_file: '', workspace_manifest: {},
    }],
  },
}

beforeEach(() => {
  mockFetch.mockReset()
  mockFetch.mockImplementation((url: string) => {
    if (url.endsWith('/snapshots')) return Promise.resolve(oneSnapshot)
    if (url.includes('/rollback-preview')) {
      return Promise.resolve({ success: true, data: {
        files_to_delete_count: 0, notebook_cells_removed: 0, descendant_snapshots: [],
      } })
    }
    if (url.includes('/restore')) {
      return Promise.reject(new Error('Rollback failed: current_round type mismatch'))
    }
    return Promise.resolve({ success: true, data: {} })
  })
})

describe('SnapshotPanel restore error', () => {
  it('shows an inline error when restore fails', async () => {
    render(
      <SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />
    )
    // Wait for the snapshot list to render.
    await screen.findByTitle('Restore')
    fireEvent.click(screen.getByTitle('Restore'))          // open confirm
    fireEvent.click(await screen.findByText('Yes'))        // confirm restore
    expect(await screen.findByText(/Rollback failed/)).toBeInTheDocument()
  })
})
