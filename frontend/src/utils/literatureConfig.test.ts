import { describe, it, expect } from 'vitest'
import { buildLiteratureConfigPayload } from './literatureConfig'

const baseState = {
  literatureMode: 'deep_research' as const,
  deepResearchModel: 'o4-mini-deep-research',
  deepResearchLlmBaseUrl: '',
  deepResearchApiKey: '',
  googleApiKey: '',
  pubmedEmail: 'user@example.com',
  literatureMaxSearchRounds: 5,
  apiConfig: { llmBaseUrl: 'https://global.example/v1', apiKey: 'global-key', proxyUrl: '' },
}

describe('buildLiteratureConfigPayload', () => {
  it('uses raw store values and does not persist global fallback credentials', () => {
    const payload = buildLiteratureConfigPayload(baseState)
    expect(payload).toEqual({
      mode: 'deep_research',
      deep_research_model: 'o4-mini-deep-research',
      deep_research_llm_base_url: '',
      deep_research_api_key: '',
      google_api_key: '',
      pubmed_email: 'user@example.com',
      max_search_rounds: 5,
    })
  })

  it('preserves masked deep research secret sentinel in payload', () => {
    const state = {
      ...baseState,
      deepResearchApiKey: '********',
      apiConfig: { ...baseState.apiConfig, apiKey: 'sk-global' },
    }

    expect(buildLiteratureConfigPayload(state).deep_research_api_key).toBe('********')
  })
})
