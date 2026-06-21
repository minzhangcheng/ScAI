import { describe, it, expect } from 'vitest'
import { buildLiteratureConfigPayload, resolveDeepResearchServerConfig, isMaskedSecret } from './literatureConfig'

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

describe('isMaskedSecret', () => {
  it('returns true for masked sentinel', () => {
    expect(isMaskedSecret('********')).toBe(true)
  })

  it('returns false for non-masked values', () => {
    expect(isMaskedSecret('sk-real-key')).toBe(false)
    expect(isMaskedSecret('')).toBe(false)
    expect(isMaskedSecret(undefined)).toBe(false)
  })
})

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

  it('uses store values when deep research fields are set', () => {
    const state = {
      ...baseState,
      deepResearchLlmBaseUrl: 'https://custom.server/v1',
      deepResearchApiKey: 'sk-custom',
    }
    expect(buildLiteratureConfigPayload(state)).toMatchObject({
      deep_research_llm_base_url: 'https://custom.server/v1',
      deep_research_api_key: 'sk-custom',
    })
  })

  it('does not persist global deep research fallback credentials when literature fields are empty', () => {
    const state = {
      ...baseState,
      deepResearchLlmBaseUrl: '',
      deepResearchApiKey: '',
      apiConfig: { ...baseState.apiConfig, llmBaseUrl: 'https://global.example/v1', apiKey: 'sk-global' },
    }

    expect(buildLiteratureConfigPayload(state)).toMatchObject({
      deep_research_llm_base_url: '',
      deep_research_api_key: '',
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

describe('resolveDeepResearchServerConfig', () => {
  it('returns store values when both are set', () => {
    const result = resolveDeepResearchServerConfig({
      deepResearchLlmBaseUrl: 'https://custom.server.com/v1',
      deepResearchApiKey: 'sk-custom-key',
      apiConfig: { llmBaseUrl: 'https://openrouter.ai/api/v1', apiKey: 'sk-default', proxyUrl: '' },
    })
    expect(result.deepResearchLlmBaseUrl).toBe('https://custom.server.com/v1')
    expect(result.deepResearchApiKey).toBe('sk-custom-key')
  })

  it('falls back to apiConfig when store values are empty', () => {
    const result = resolveDeepResearchServerConfig({
      deepResearchLlmBaseUrl: '',
      deepResearchApiKey: '',
      apiConfig: { llmBaseUrl: 'https://my-proxy.com/v1', apiKey: 'sk-proxy', proxyUrl: '' },
    })
    expect(result.deepResearchLlmBaseUrl).toBe('https://my-proxy.com/v1')
    expect(result.deepResearchApiKey).toBe('sk-proxy')
  })

  it('falls back to hardcoded default for base URL when store and apiConfig are empty', () => {
    const result = resolveDeepResearchServerConfig({
      deepResearchLlmBaseUrl: '',
      deepResearchApiKey: '',
      apiConfig: { llmBaseUrl: '', apiKey: '', proxyUrl: '' },
    })
    expect(result.deepResearchLlmBaseUrl).toBe('https://openrouter.ai/api/v1')
    expect(result.deepResearchApiKey).toBe('')
  })

  it('falls back to apiConfig when store key is empty but URL is set', () => {
    const result = resolveDeepResearchServerConfig({
      deepResearchLlmBaseUrl: 'https://custom.server.com/v1',
      deepResearchApiKey: '',
      apiConfig: { llmBaseUrl: 'https://openrouter.ai/api/v1', apiKey: 'sk-default', proxyUrl: '' },
    })
    expect(result.deepResearchLlmBaseUrl).toBe('https://custom.server.com/v1')
    expect(result.deepResearchApiKey).toBe('sk-default')
  })

  it('treats nullish values as empty', () => {
    const result = resolveDeepResearchServerConfig({
      deepResearchLlmBaseUrl: null as unknown as string,
      deepResearchApiKey: undefined as unknown as string,
      apiConfig: { llmBaseUrl: 'https://proxy.com/v1', apiKey: 'sk-proxy', proxyUrl: '' },
    })
    expect(result.deepResearchLlmBaseUrl).toBe('https://proxy.com/v1')
    expect(result.deepResearchApiKey).toBe('sk-proxy')
  })
})
