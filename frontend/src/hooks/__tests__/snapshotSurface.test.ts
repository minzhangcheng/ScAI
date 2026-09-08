import { describe, expect, it } from 'vitest'

import * as api from '../useApi'
import { useAppStore } from '../../stores/appStore'

/**
 * Snapshot V2's browser is gone and stays gone.
 *
 * This used to assert that *nothing* matching `/snapshot/i` existed on either
 * surface, which was right while there was no such feature. Part B is a
 * different one -- decision snapshots, restored into a new session, with no
 * tree, no branch model and no way to create one by hand -- so the ratchet
 * names V2's own components instead. Anything from that list coming back is a
 * regression; the four calls below are the whole of the replacement.
 */

/** V2's browser, dialogs and controller. None of these may return. */
const RETIRED_V2_SURFACE = [
  'SnapshotPanelView',
  'SnapshotTree',
  'SnapshotPreview',
  'StorageSummary',
  'RestoreProgressDialog',
  'RuntimeChoiceDialog',
  'DiscardCurrentDialog',
  'useSnapshotPanelController',
  'createSnapshot',
  'listSnapshotObjects',
]

describe('no Snapshot V2 surface survives', () => {
  it('exposes none of the retired names', () => {
    const exported = new Set(Object.keys(api))
    expect(RETIRED_V2_SURFACE.filter(name => exported.has(name))).toEqual([])
  })

  it('holds no snapshot state in the store', () => {
    const keys = Object.keys(useAppStore.getState()).filter(
      name => /snapshot/i.test(name),
    )
    expect(keys).toEqual([])
  })

  it('exposes exactly the Part B calls, and no way to create one', () => {
    const snapshotCalls = Object.keys(api)
      .filter(name => /snapshot/i.test(name))
      .sort()
    expect(snapshotCalls).toEqual([
      'deleteSnapshot', 'listSnapshots', 'restoreSnapshot',
    ])
  })

  it('keeps the runtime disclosure the snapshot slice used to carry', () => {
    const state = useAppStore.getState()

    expect(state.runtimeRestoreStatus).toBe('complete')
    expect(state.isForwardResumeBlocked()).toBe(false)

    useAppStore.setState({ runtimeRestoreStatus: 'unavailable' })
    expect(useAppStore.getState().isForwardResumeBlocked()).toBe(true)
    expect(useAppStore.getState().isRerunEnabled()).toBe(true)

    useAppStore.setState({ runtimeRestoreStatus: 'complete' })
  })
})
