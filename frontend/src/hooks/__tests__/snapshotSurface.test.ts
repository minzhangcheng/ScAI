import { describe, expect, it } from 'vitest'

import * as api from '../useApi'
import { useAppStore } from '../../stores/appStore'

/**
 * The snapshot browser is gone, and nothing may reach for it.
 *
 * Every one of its calls hit a route the backend no longer registers, so a
 * survivor is not a dangling import `tsc` would catch — it is a button that
 * 404s. These assert on the two surfaces a new caller would reach through: the
 * API module's exports and the store's keys.
 */
describe('no snapshot surface survives', () => {
  it('exposes no snapshot API function', () => {
    expect(Object.keys(api).filter(name => /snapshot/i.test(name))).toEqual([])
  })

  it('holds no snapshot state', () => {
    const keys = Object.keys(useAppStore.getState()).filter(
      name => /snapshot/i.test(name),
    )
    expect(keys).toEqual([])
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
