import { it, expect } from 'vitest'
import { partialize } from '../appStore'

it('does not persist api keys (nested or top-level)', () => {
  const state = {
    apiConfig: { apiKey: 'sk-1', llmBaseUrl: 'http://x', proxyUrl: '' },
    deepResearchApiKey: 'sk-2',
    googleApiKey: 'sk-3',
    language: 'en',
    prompts: {},
    models: {},
    temperatures: {},
    maxRounds: 5,
    roundWaitSeconds: 300,
    agentServerConfigs: {},
    agentVisionSettings: {},
    literatureMode: 'disabled',
    deepResearchModel: '',
    deepResearchLlmBaseUrl: '',
    pubmedEmail: '',
    literatureMaxSearchRounds: 3,
    messagePanelWidth: 400,
    chatMessages: [],
    messages: [],
  } as any
  const persisted = partialize(state)
  expect(persisted.apiConfig?.apiKey ?? '').toBe('')
  expect((persisted as any).deepResearchApiKey).toBeUndefined()
  expect((persisted as any).googleApiKey).toBeUndefined()
})
