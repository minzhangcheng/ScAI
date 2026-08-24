import { beforeEach, describe, expect, it } from 'vitest'
import { partialize, useAppStore } from '../appStore'
import {
  DEFAULT_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
} from '../../constants/layout'

describe('appStore sidebar layout', () => {
  beforeEach(() => {
    useAppStore.setState({
      sidebarOpen: true,
      sidebarWidth: DEFAULT_SIDEBAR_WIDTH,
    })
  })

  it('persists only valid clamped width and collapse', () => {
    const store = useAppStore.getState()

    store.setSidebarWidth(MIN_SIDEBAR_WIDTH - 100)
    expect(useAppStore.getState().sidebarWidth).toBe(MIN_SIDEBAR_WIDTH)

    store.setSidebarWidth(MAX_SIDEBAR_WIDTH + 100)
    expect(useAppStore.getState().sidebarWidth).toBe(MAX_SIDEBAR_WIDTH)

    store.setSidebarWidth(Number.NaN)
    expect(useAppStore.getState().sidebarWidth).toBe(DEFAULT_SIDEBAR_WIDTH)

    store.setSidebarWidth(360)
    store.setSidebarOpen(false)
    const persisted = partialize(useAppStore.getState()) as any
    expect(persisted.sidebarWidth).toBe(360)
    expect(persisted.sidebarOpen).toBe(false)
  })
})
