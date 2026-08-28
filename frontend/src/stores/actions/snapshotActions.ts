import type { StoreApi } from 'zustand'

import { resolveSnapshotFailureActions } from '../../utils/snapshotV2'
import type { AppState } from '../appState'

type SetState = StoreApi<AppState>['setState']
type GetState = StoreApi<AppState>['getState']

type SnapshotActionKeys =
  | 'snapshotsV2' | 'setSnapshotsV2' | 'snapshotStoreStatus' | 'setSnapshotStoreStatus'
  | 'snapshotTransaction' | 'setSnapshotTransaction'
  | 'annotationGateData' | 'setAnnotationGateData'
  | 'snapshotFailure' | 'setSnapshotFailure' | 'clearSnapshotFailure'
  | 'runtimeRestoreStatus' | 'setRuntimeRestoreStatus'
  | 'runtimeRestoreWarning' | 'setRuntimeRestoreWarning'
  | 'hydrateRuntimeRestoreFromSession' | 'isForwardResumeBlocked' | 'isRerunEnabled'

export function createSnapshotActions(
  set: SetState,
  get: GetState,
): Pick<AppState, SnapshotActionKeys> {
  return {
    snapshotsV2: [],
    setSnapshotsV2: snapshotsV2 => set({ snapshotsV2 }),
    snapshotStoreStatus: null,
    setSnapshotStoreStatus: snapshotStoreStatus => set({ snapshotStoreStatus }),
    snapshotTransaction: null,
    setSnapshotTransaction: snapshotTransaction => set({
      snapshotTransaction: snapshotTransaction !== null
        ? {
            ...snapshotTransaction,
            recovery_mode: snapshotTransaction.recovery_mode
              ?? (get().snapshotTransaction?.recovery_mode ?? 'rollback'),
          }
        : null,
    }),
    annotationGateData: null,
    setAnnotationGateData: annotationGateData => set({ annotationGateData }),
    snapshotFailure: null,
    setSnapshotFailure: snapshotFailure => set({ snapshotFailure }),
    clearSnapshotFailure: () => set({ snapshotFailure: null }),
    runtimeRestoreStatus: 'complete',
    setRuntimeRestoreStatus: runtimeRestoreStatus => set({ runtimeRestoreStatus }),
    runtimeRestoreWarning: null,
    setRuntimeRestoreWarning: runtimeRestoreWarning => set({ runtimeRestoreWarning }),
    hydrateRuntimeRestoreFromSession: payload => {
      const statusRaw = payload.runtime_restore_status
      const runtimeRestoreStatus = statusRaw === 'unavailable' || statusRaw === 'intentionally_kept'
        ? statusRaw
        : 'complete'
      const runtimeRestoreWarning = typeof payload.runtime_restore_warning === 'string'
        ? payload.runtime_restore_warning
        : null
      const updates: Partial<AppState> = { runtimeRestoreStatus, runtimeRestoreWarning }
      const rawFailure = payload.snapshot_failure
      if (rawFailure === null || rawFailure === undefined) {
        updates.snapshotFailure = null
      } else if (rawFailure && typeof rawFailure === 'object') {
        const failure = rawFailure as Record<string, unknown>
        const code = typeof failure.code === 'string' ? failure.code : ''
        const message = typeof failure.message === 'string' ? failure.message : ''
        if (code || message) {
          updates.snapshotFailure = {
            failure: {
              snapshot_id: typeof failure.snapshot_id === 'string' ? failure.snapshot_id : '',
              session_id: typeof failure.session_id === 'string' ? failure.session_id : '',
              failed_at: typeof failure.failed_at === 'string' ? failure.failed_at : '',
              code,
              message,
              details: failure.details && typeof failure.details === 'object'
                ? failure.details as Record<string, unknown>
                : undefined,
            },
            allowed_actions: resolveSnapshotFailureActions(
              failure,
              payload.snapshot_failure_allowed_actions,
            ),
          }
        }
      }
      set(updates)
    },
    isForwardResumeBlocked: () => {
      const state = get()
      if (state.snapshotFailure) return true
      return state.runtimeRestoreStatus === 'unavailable'
    },
    isRerunEnabled: () => {
      const state = get()
      return state.runtimeRestoreStatus === 'unavailable'
        || state.runtimeRestoreStatus === 'intentionally_kept'
    },
  }
}
