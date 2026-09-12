import type { StoreApi } from 'zustand'

import type { AppState } from '../appState'
import { chatOwnerKey } from '../../utils/chatOwnership'

type SetState = StoreApi<AppState>['setState']
type GetState = StoreApi<AppState>['getState']

type RuntimeActionKeys =
  | 'annotationGateData' | 'annotationGateOwnerKey' | 'setAnnotationGateData'
  | 'runtimeRestoreStatus' | 'setRuntimeRestoreStatus'
  | 'runtimeRestoreWarning' | 'setRuntimeRestoreWarning'
  | 'hydrateRuntimeRestoreFromSession' | 'isForwardResumeBlocked' | 'isRerunEnabled'

/**
 * Runtime availability, and the annotation gate that reloads beside it.
 *
 * Split out of `snapshotActions.ts` when the snapshot browser was removed: the
 * snapshot half of that slice had nothing left to talk to, but this half is the
 * disclosure spec 5.1 requires. A session imported without its runtime, or one
 * whose runtime could not be rehydrated, arrives here as `unavailable` and the
 * UI offers a re-run instead of a resume.
 */
export function createRuntimeActions(
  set: SetState,
  get: GetState,
): Pick<AppState, RuntimeActionKeys> {
  return {
    annotationGateData: null,
    annotationGateOwnerKey: null,
    setAnnotationGateData: annotationGateData => set(state => ({
      annotationGateData,
      annotationGateOwnerKey: annotationGateData
        ? chatOwnerKey(
            state.currentSession?.id,
            state.currentSession?.restoreBranchId,
          )
        : null,
    })),
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
      set({ runtimeRestoreStatus, runtimeRestoreWarning })
    },
    isForwardResumeBlocked: () => get().runtimeRestoreStatus === 'unavailable',
    isRerunEnabled: () => {
      const state = get()
      return state.runtimeRestoreStatus === 'unavailable'
        || state.runtimeRestoreStatus === 'intentionally_kept'
    },
  }
}
