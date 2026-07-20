import { beforeEach, describe, expect, it } from 'vitest'
import { DEFAULT_MODELS, useAppStore } from '../appStore'

describe('Agent page defaults', () => {
  beforeEach(() => {
    useAppStore.getState().resetModelsToDefault()
  })

  it('uses the configured default model for every agent', () => {
    expect(DEFAULT_MODELS).toEqual({
      pi: 'gpt-5.6-sol',
      pathology: 'gemini-3.1-pro-preview',
      bioinformatics: 'gpt-5.6-sol',
      biochemistry: 'gemini-3.1-pro-preview',
      literature: 'gpt-5.4-mini',
      painter: 'gpt-image-2',
      supervisor: 'gpt-5.4-mini',
      technique: 'gpt-5.4-mini',
    })
  })

  it('defaults Literature Agent to PubMed + LLM Only', () => {
    const state = useAppStore.getState()
    expect(state.literatureMode).toBe('pubmed_llm')
    expect(state.deepResearchModel).toBe('o4-mini-deep-research')
    expect(state.literatureMaxSearchRounds).toBe(20)
  })
})
