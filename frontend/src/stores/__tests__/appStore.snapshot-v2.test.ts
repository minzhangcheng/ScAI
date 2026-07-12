import { describe, it, expect, beforeEach } from 'vitest'
import { useAppStore } from '../appStore'

describe('Snapshot V2 store actions', () => {
  beforeEach(() => {
    useAppStore.setState({
      snapshotsV2: [],
      snapshotStoreStatus: null,
      snapshotTransaction: null,
      snapshotFailure: null,
    })
  })

  describe('setSnapshotsV2', () => {
    it('updates snapshotsV2 state', () => {
      const snapshots = [
        {
          snapshot_id: 'snap-1',
          manifest: {
            format_version: 2 as const,
            session_id: 'test-session',
            snapshot_id: 'snap-1',
            captured_at: '2025-01-01T00:00:00Z',
            branch: 'main',
            generation: 1,
            parent_id: null,
            root_hash: '',
            capsule: { kind: 'idle' as const, graph_version: '1.0' },
            workspace: [],
            objects: {},
          },
          failure: null,
        },
      ]

      useAppStore.getState().setSnapshotsV2(snapshots)
      expect(useAppStore.getState().snapshotsV2).toEqual(snapshots)
    })

    it('replaces previous snapshots list', () => {
      useAppStore.getState().setSnapshotsV2([{ snapshot_id: 'old' } as any])
      useAppStore.getState().setSnapshotsV2([{ snapshot_id: 'new' } as any])
      expect(useAppStore.getState().snapshotsV2).toHaveLength(1)
      expect(useAppStore.getState().snapshotsV2[0].snapshot_id).toBe('new')
    })
  })

  describe('setSnapshotStoreStatus', () => {
    it('updates snapshotStoreStatus state', () => {
      const status = {
        logical_bytes: 1000,
        physical_bytes: 500,
        dedup_savings: 500,
        object_count: 3,
        manifest_count: 1,
      }

      useAppStore.getState().setSnapshotStoreStatus(status)
      expect(useAppStore.getState().snapshotStoreStatus).toEqual(status)
    })
  })

  describe('setSnapshotTransaction', () => {
    it('updates snapshotTransaction state', () => {
      const txn = {
        transaction_id: 'txn-1',
        session_id: 'test-session',
        snapshot_id: 'snap-1',
        status: 'queued' as const,
        recovery_mode: 'rollback' as const,
        failure: null,
      }

      useAppStore.getState().setSnapshotTransaction(txn)
      expect(useAppStore.getState().snapshotTransaction).toEqual(txn)
    })

    it('updates progress for same transaction ID', () => {
      useAppStore.getState().setSnapshotTransaction({
        transaction_id: 'txn-1',
        session_id: 'test-session',
        snapshot_id: 'snap-1',
        status: 'queued',
        recovery_mode: 'rollback',
        failure: null,
      })

      useAppStore.getState().setSnapshotTransaction({
        transaction_id: 'txn-1',
        session_id: 'test-session',
        snapshot_id: 'snap-1',
        status: 'staging',
        recovery_mode: 'rollback',
        failure: null,
      })

      expect(useAppStore.getState().snapshotTransaction?.status).toBe('staging')
    })

    it('persists risk warning across progress updates', () => {
      useAppStore.getState().setSnapshotTransaction({
        transaction_id: 'txn-1',
        session_id: 'test-session',
        snapshot_id: 'snap-1',
        status: 'queued',
        recovery_mode: 'forward_only',
        failure: null,
      })

      useAppStore.getState().setSnapshotTransaction({
        transaction_id: 'txn-1',
        session_id: 'test-session',
        snapshot_id: 'snap-1',
        status: 'staging',
        recovery_mode: 'forward_only',
        failure: null,
      })

      expect(useAppStore.getState().snapshotTransaction?.recovery_mode).toBe('forward_only')
    })
  })

  describe('setSnapshotFailure', () => {
    it('updates snapshotFailure state', () => {
      const failure = {
        failure: {
          snapshot_id: 'snap-1',
          session_id: 'sess-1',
          failed_at: '2026-07-12T00:00:00Z',
          code: 'INTEGRITY_ERROR',
          message: 'Manifest hash mismatch',
          details: { expected: 'abc', actual: 'def' },
        },
        allowed_actions: ['retry', 'terminate'] as const,
      }

      useAppStore.getState().setSnapshotFailure({
        ...failure,
        allowed_actions: [...failure.allowed_actions],
      })
      expect(useAppStore.getState().snapshotFailure).toEqual({
        ...failure,
        allowed_actions: [...failure.allowed_actions],
      })
    })
  })

  describe('clearSnapshotFailure', () => {
    it('clears snapshotFailure to null', () => {
      useAppStore.getState().setSnapshotFailure({
        failure: {
          snapshot_id: 'snap-1',
          session_id: 'sess-1',
          failed_at: '2026-07-12T00:00:00Z',
          code: 'ERR',
          message: 'err',
        },
        allowed_actions: ['retry'],
      })

      useAppStore.getState().clearSnapshotFailure()
      expect(useAppStore.getState().snapshotFailure).toBeNull()
    })
  })

  describe('runtime restore safety', () => {
    beforeEach(() => {
      useAppStore.setState({
        runtimeRestoreStatus: 'complete',
        runtimeRestoreWarning: null,
        snapshotFailure: null,
      })
    })

    it('persists intentionally_kept keep warning across reload hydration', () => {
      useAppStore.getState().setRuntimeRestoreStatus('intentionally_kept')
      useAppStore.getState().setRuntimeRestoreWarning(
        'Live kernel retained; may be inconsistent with restored business state.',
      )

      // Simulate page reload: re-hydrate from backend session load payload
      useAppStore.getState().hydrateRuntimeRestoreFromSession({
        runtime_restore_status: 'intentionally_kept',
        runtime_restore_warning:
          'Live kernel retained; may be inconsistent with restored business state.',
      })

      expect(useAppStore.getState().runtimeRestoreStatus).toBe('intentionally_kept')
      expect(useAppStore.getState().runtimeRestoreWarning).toMatch(/inconsistent/i)
    })

    it('reset restore (unavailable) disables resume but enables rerun', () => {
      useAppStore.getState().setRuntimeRestoreStatus('unavailable')

      expect(useAppStore.getState().isForwardResumeBlocked()).toBe(true)
      expect(useAppStore.getState().isRerunEnabled()).toBe(true)
    })

    it('intentionally_kept does not block resume', () => {
      useAppStore.getState().setRuntimeRestoreStatus('intentionally_kept')

      expect(useAppStore.getState().isForwardResumeBlocked()).toBe(false)
      expect(useAppStore.getState().isRerunEnabled()).toBe(true)
    })

    it('complete status allows resume and does not force rerun-only mode', () => {
      useAppStore.getState().setRuntimeRestoreStatus('complete')

      expect(useAppStore.getState().isForwardResumeBlocked()).toBe(false)
      expect(useAppStore.getState().isRerunEnabled()).toBe(false)
    })

    it('snapshot failure blocks generic resume/start until dedicated action', () => {
      useAppStore.getState().setSnapshotFailure({
        failure: {
          snapshot_id: 'snap-1',
          session_id: 's',
          failed_at: '2025-01-01T00:00:00Z',
          code: 'CAPTURE_FAILED',
          message: 'fail',
        },
        allowed_actions: ['retry', 'manage', 'terminate'],
      })

      expect(useAppStore.getState().isSnapshotFailureBlocked()).toBe(true)
      expect(useAppStore.getState().isForwardResumeBlocked()).toBe(true)
    })
  })
})
