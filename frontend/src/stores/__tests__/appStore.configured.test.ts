import { it, expect } from 'vitest'
import { useAppStore } from '../appStore'

it('tracks server-side apiConfigured flag', () => {
  useAppStore.getState().setApiConfigured(true)
  expect(useAppStore.getState().apiConfigured).toBe(true)
})
