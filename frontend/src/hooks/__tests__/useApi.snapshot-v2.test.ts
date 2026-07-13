import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useAppStore } from '../../stores/appStore'

// Mock fetch globally
const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

// Helper to get auth token mock
vi.mock('../../utils/authStorage', () => ({
  getToken: vi.fn(() => 'test-token'),
  clearToken: vi.fn(),
}))

describe('Snapshot V2 API methods', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({
      currentSession: {
        id: 'test-session',
        name: 'Test Session',
        status: 'running',
        currentRound: 1,
        createdAt: new Date().toISOString(),
        instruction: '',
        treatmentGroups: [],
        dataFiles: [],
        biochemistryIndicators: [],
        roundResults: [],
      },
    })
  })

  describe('listSnapshotsV2', () => {
    it('builds correct GET URL and parses response', async () => {
      const responseData = {
        success: true,
        data: {
          snapshots: [
            {
              snapshot_id: 'snap-1',
              manifest: {
                format_version: 2,
                session_id: 'test-session',
                snapshot_id: 'snap-1',
                captured_at: '2025-01-01T00:00:00Z',
                branch: 'main',
                generation: 1,
                parent_id: null,
                root_hash: '',
                capsule: { kind: 'idle', graph_version: '1.0' },
                workspace: [],
                objects: {},
              },
              failure: null,
            },
          ],
          store_status: {
            logical_bytes: 1000,
            physical_bytes: 500,
            dedup_savings: 500,
            object_count: 3,
            manifest_count: 1,
          },
        },
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      // Dynamically import the method from useApi (will be available after implementation)
      const { listSnapshotsV2 } = await import('../useApi')
      const result = await listSnapshotsV2('test-session')

      expect(mockFetch).toHaveBeenCalledTimes(1)
      const callUrl = mockFetch.mock.calls[0][0]
      expect(callUrl).toContain('/api/session/test-session/snapshots-v2')
      expect(mockFetch.mock.calls[0][1]?.method).toBeUndefined()

      expect(result.snapshots).toHaveLength(1)
      expect(result.snapshots[0].snapshot_id).toBe('snap-1')
      expect(result.snapshots[0].manifest.capsule.kind).toBe('idle')
      expect(result.store_status.manifest_count).toBe(1)
    })

    it('normalizes the real backend manifest shape (flat, continuation/created_at/branch_id)', async () => {
      // The backend serves each snapshot as a *flat* manifest dump using the
      // authoritative field names — no `manifest` envelope, `continuation`
      // instead of `capsule`, `created_at` instead of `captured_at`,
      // `branch_id` instead of `branch`. The panel consumes the wrapped shape.
      const responseData = {
        success: true,
        data: {
          snapshots: [
            {
              format_version: 2,
              snapshot_id: 'snap-real',
              session_id: 'test-session',
              name: 'snap-real',
              description: '',
              phase: 'pre_execution_stage1',
              created_at: '2026-07-13T12:30:35Z',
              round_number: 1,
              parent_id: null,
              branch_id: 'main',
              branch_label: 'main',
              root_hash: '752bee46',
              continuation: {
                graph_version: '2.0',
                kind: 'graph_node',
                next_node: 'bioinformatics_analysis',
                planning_stage: 'stage1',
                workflow_state_ref: { sha256: 'ws', size: 512, media_type: 'application/json' },
              },
              workspace: [],
              objects: {},
              status: 'ok',
              integrity_status: 'ok',
            },
          ],
          store_status: {
            logical_bytes: 1000,
            physical_bytes: 500,
            dedup_savings: 500,
            object_count: 3,
            manifest_count: 1,
          },
        },
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      const { listSnapshotsV2 } = await import('../useApi')
      const result = await listSnapshotsV2('test-session')

      const rec = result.snapshots[0]
      expect(rec.snapshot_id).toBe('snap-real')
      // Wrapped envelope + old field names the panel/tree/preview read.
      expect(rec.manifest).toBeDefined()
      expect(rec.manifest.capsule.kind).toBe('graph_node')
      expect(rec.manifest.captured_at).toBe('2026-07-13T12:30:35Z')
      expect(rec.manifest.branch).toBe('main')
      expect(rec.failure).toBeNull()
    })

    it('maps a corrupt backend entry to a failure without crashing manifest access', async () => {
      const responseData = {
        success: true,
        data: {
          snapshots: [
            {
              snapshot_id: 'snap-bad',
              status: 'corrupt',
              integrity_status: 'corrupt',
              error_code: 'HASH_MISMATCH',
              error_message: 'root hash mismatch',
              manifest: null,
            },
          ],
          store_status: {
            logical_bytes: 0,
            physical_bytes: 0,
            dedup_savings: 0,
            object_count: 0,
            manifest_count: 0,
          },
        },
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      const { listSnapshotsV2 } = await import('../useApi')
      const result = await listSnapshotsV2('test-session')

      const rec = result.snapshots[0]
      expect(rec.snapshot_id).toBe('snap-bad')
      expect(rec.failure?.code).toBe('HASH_MISMATCH')
      // Panel/tree/preview deref manifest fields unconditionally; must be safe.
      expect(rec.manifest).toBeDefined()
      expect(rec.manifest.capsule.kind).toBeDefined()
      expect(() => new Date(rec.manifest.captured_at).getTime()).not.toThrow()
    })
  })

  describe('previewSnapshotV2', () => {
    it('builds correct GET URL for preview', async () => {
      const responseData = {
        success: true,
        data: {
          manifest: {
            format_version: 2,
            session_id: 'test-session',
            snapshot_id: 'snap-1',
            captured_at: '2025-01-01T00:00:00Z',
            branch: 'main',
            generation: 1,
            parent_id: null,
            root_hash: '',
            capsule: { kind: 'idle', graph_version: '1.0' },
          },
        },
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      const { previewSnapshotV2 } = await import('../useApi')
      const result = await previewSnapshotV2('test-session', 'snap-1')

      expect(mockFetch.mock.calls[0][0]).toContain('/api/session/test-session/snapshots-v2/snap-1')
      expect(result.manifest.snapshot_id).toBe('snap-1')
    })
  })

  describe('restoreSnapshotV2', () => {
    it('sends correct request body and returns transaction response with 202', async () => {
      const responseData = {
        transaction_id: 'txn-1',
        status: 'queued',
        status_url: '/api/session/test-session/snapshot-transactions/txn-1',
        cancel_url: '/api/session/test-session/snapshot-transactions/txn-1/cancel',
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 202,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      const { restoreSnapshotV2 } = await import('../useApi')
      const result = await restoreSnapshotV2('test-session', 'snap-1', {
        snapshot_id: 'snap-1',
        runtime_choice: 'keep',
        current_state_choice: 'abort',
        discard_confirmation: false,
      })

      expect(mockFetch).toHaveBeenCalledTimes(1)
      const callUrl = mockFetch.mock.calls[0][0]
      const callOptions = mockFetch.mock.calls[0][1]
      expect(callUrl).toContain('/api/session/test-session/snapshots/snap-1/restore-v2')
      expect(callOptions.method).toBe('POST')
      expect(callOptions.body).toBe(JSON.stringify({
        snapshot_id: 'snap-1',
        runtime_choice: 'keep',
        current_state_choice: 'abort',
        discard_confirmation: false,
      }))
      expect(result.transaction_id).toBe('txn-1')
      expect(result.status).toBe('queued')
    })

    it('does NOT change session status to running after restore call', async () => {
      const responseData = {
        transaction_id: 'txn-1',
        status: 'queued',
        status_url: '/api/session/test-session/snapshot-transactions/txn-1',
        cancel_url: '/api/session/test-session/snapshot-transactions/txn-1/cancel',
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 202,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      useAppStore.getState().updateSession({ status: 'paused' })

      const { restoreSnapshotV2 } = await import('../useApi')
      await restoreSnapshotV2('test-session', 'snap-1', {
        snapshot_id: 'snap-1',
        current_state_choice: 'abort',
        discard_confirmation: false,
      })

      expect(useAppStore.getState().currentSession?.status).toBe('paused')
    })
  })

  describe('cancelSnapshotTransaction', () => {
    it('builds correct POST URL for cancel', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve({ success: true }),
      })

      const { cancelSnapshotTransaction } = await import('../useApi')
      await cancelSnapshotTransaction('test-session', 'txn-1')

      expect(mockFetch.mock.calls[0][0]).toContain('/api/session/test-session/snapshot-transactions/txn-1/cancel')
      expect(mockFetch.mock.calls[0][1]?.method).toBe('POST')
    })
  })

  describe('getSnapshotTransaction', () => {
    it('builds correct GET URL and returns transaction record', async () => {
      const responseData = {
        success: true,
        data: {
          transaction_id: 'txn-1',
          session_id: 'test-session',
          snapshot_id: 'snap-1',
          status: 'queued',
          recovery_mode: 'rollback',
          failure: null,
        },
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      const { getSnapshotTransaction } = await import('../useApi')
      const result = await getSnapshotTransaction('test-session', 'txn-1')

      expect(mockFetch.mock.calls[0][0]).toContain('/api/session/test-session/snapshot-transactions/txn-1')
      expect(mockFetch.mock.calls[0][0]).not.toContain('/cancel')
      expect(result.transaction_id).toBe('txn-1')
      expect(result.status).toBe('queued')
    })
  })

  describe('deleteSnapshotV2', () => {
    it('builds correct DELETE URL for leaf deletion', async () => {
      const responseData = {
        success: true,
        data: {
          deleted_ids: ['snap-1'],
          freed_bytes: 500,
          surviving_count: 0,
        },
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      const { deleteSnapshotV2 } = await import('../useApi')
      const result = await deleteSnapshotV2('test-session', 'snap-1')

      const callUrl = mockFetch.mock.calls[0][0]
      expect(callUrl).toContain('/api/session/test-session/snapshots-v2/snap-1')
      expect(callUrl).not.toContain('branch=true')
      expect(mockFetch.mock.calls[0][1]?.method).toBe('DELETE')
      expect(result.deleted_ids).toEqual(['snap-1'])
    })

    it('adds branch=true query param for branch deletion', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve({ success: true, data: { deleted_ids: ['snap-1', 'snap-2'], freed_bytes: 1000, surviving_count: 0 } }),
      })

      const { deleteSnapshotV2 } = await import('../useApi')
      await deleteSnapshotV2('test-session', 'snap-1', true)

      const callUrl = mockFetch.mock.calls[0][0]
      expect(callUrl).toContain('branch=true')
    })
  })

  describe('exportSnapshotsV2', () => {
    it('builds correct GET URL for export', async () => {
      const responseData = {
        status: 'ready',
        download_url: '/api/session/test-session/export/download',
        filename: 'v2-export-test.tar.gz',
        size_bytes: 12345,
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve(responseData),
      })

      const { exportSnapshotsV2 } = await import('../useApi')
      const result = await exportSnapshotsV2('test-session', { include_runtime: true })

      const callUrl = mockFetch.mock.calls[0][0]
      expect(callUrl).toContain('/api/session/test-session/export-v2')
      expect(callUrl).toContain('include_runtime=true')
      expect(result.download_url).toBeTruthy()
    })

    it('includes snapshot_ids query param when provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve({ status: 'ready', download_url: '/download', filename: 'x.tar.gz', size_bytes: 100 }),
      })

      const { exportSnapshotsV2 } = await import('../useApi')
      await exportSnapshotsV2('test-session', { snapshot_ids: ['snap-1', 'snap-2'] })

      const callUrl = mockFetch.mock.calls[0][0]
      expect(callUrl).toContain('snapshot_ids=snap-1%2Csnap-2')
    })
  })

  describe('importSnapshotsV2', () => {
    it('sends file as gzip with x-trust header', async () => {
      const mockResult = {
        success: true,
        dest_session_id: 'sess-2',
        imported_snapshot_ids: ['snap-a'],
        committed_count: 1,
        skipped_runtime_snapshot_ids: [],
        errors: [],
        warnings: [],
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () => Promise.resolve(mockResult),
      })

      const { importSnapshotsV2 } = await import('../useApi')
      const file = new File(['test-data'], 'export.tar.gz', { type: 'application/gzip' })
      const result = await importSnapshotsV2('test-session', file, true)

      expect(mockFetch).toHaveBeenCalledTimes(1)
      const callUrl = mockFetch.mock.calls[0][0]
      const callOptions = mockFetch.mock.calls[0][1]
      expect(callUrl).toContain('/api/session/test-session/import-v2')
      expect(callOptions.method).toBe('POST')
      expect(callOptions.headers['X-Filename']).toBe('export.tar.gz')
      expect(callOptions.headers['X-Trust']).toBe('true')
      expect(callOptions.body).toBe(file)
      expect(result.imported_snapshot_ids).toEqual(['snap-a'])
      expect(result.committed_count).toBe(1)
    })

    it('defaults trust to false when not provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: new Map(),
        json: () =>
          Promise.resolve({
            success: true,
            dest_session_id: 'sess-2',
            imported_snapshot_ids: ['snap-b'],
            committed_count: 1,
            skipped_runtime_snapshot_ids: [],
            errors: [],
            warnings: [],
          }),
      })

      const { importSnapshotsV2 } = await import('../useApi')
      const file = new File(['test-data'], 'export.tar.gz', { type: 'application/gzip' })
      await importSnapshotsV2('test-session', file)

      const callOptions = mockFetch.mock.calls[0][1]
      expect(callOptions.headers['X-Trust']).toBe('false')
    })
  })
})
