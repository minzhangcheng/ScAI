import { describe, it, expect, beforeEach } from 'vitest'
import { applyStoreUpdates } from './useApi'
import { useAppStore } from '../stores/appStore'

describe('applyStoreUpdates', () => {
  beforeEach(() => {
    useAppStore.getState().setDeepResearchApiKey('local-unsaved-dr-key')
    useAppStore.getState().setGoogleApiKey('local-unsaved-google-key')
  })

  it('does not overwrite local deep research key with masked sentinel', () => {
    applyStoreUpdates({ deepResearchApiKey: '********' })

    expect(useAppStore.getState().deepResearchApiKey).toBe('local-unsaved-dr-key')
  })

  it('does not overwrite local google key with masked sentinel', () => {
    applyStoreUpdates({ googleApiKey: '********' })

    expect(useAppStore.getState().googleApiKey).toBe('local-unsaved-google-key')
  })

  it('hydrates masked deep research sentinel when local key is empty', () => {
    useAppStore.getState().setDeepResearchApiKey('')

    applyStoreUpdates({ deepResearchApiKey: '********' })

    expect(useAppStore.getState().deepResearchApiKey).toBe('********')
  })

  it('hydrates masked google sentinel when local key is empty', () => {
    useAppStore.getState().setGoogleApiKey('')

    applyStoreUpdates({ googleApiKey: '********' })

    expect(useAppStore.getState().googleApiKey).toBe('********')
  })

  it('keeps masked sentinel when both local and incoming deep research keys are masked', () => {
    useAppStore.getState().setDeepResearchApiKey('********')

    applyStoreUpdates({ deepResearchApiKey: '********' })

    expect(useAppStore.getState().deepResearchApiKey).toBe('********')
  })
})
