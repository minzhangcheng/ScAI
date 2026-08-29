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

  it('hydrates the authoritative session agent settings', () => {
    useAppStore.getState().setAgentServerConfig('pi', {
      llmBaseUrl: 'https://old.example/v1',
      apiKey: 'local-agent-key',
    })

    applyStoreUpdates({
      models: { ...useAppStore.getState().models, pi: 'mimo-v2.5' },
      agentServerConfigs: {
        pi: { llmBaseUrl: 'https://new.example/v1', apiKey: '********' },
      },
    })

    expect(useAppStore.getState().models.pi).toBe('mimo-v2.5')
    expect(useAppStore.getState().agentServerConfigs.pi).toEqual({
      llmBaseUrl: 'https://new.example/v1',
      apiKey: '********',
    })
  })

  it('applies snapshot transaction hydration', () => {
    const transaction = {
      transaction_id: 'restore-job-reload',
      session_id: 'session-1',
      snapshot_id: 'snapshot-1',
      status: 'queued' as const,
      recovery_mode: 'rollback' as const,
      failure: null,
    }
    applyStoreUpdates({ snapshotTransaction: transaction })
    expect(useAppStore.getState().snapshotTransaction).toEqual(transaction)
    applyStoreUpdates({ snapshotTransaction: null })
    expect(useAppStore.getState().snapshotTransaction).toBeNull()
  })
})
