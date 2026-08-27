import type { StoreApi } from 'zustand'

import { getDefaultPrompts, type Language } from '../../i18n/translations'
import type { AppState } from '../appState'
import { DEFAULT_MODELS, DEFAULT_TEMPERATURES, DEFAULT_VISION_SETTINGS } from '../domain'

type SetState = StoreApi<AppState>['setState']
type GetState = StoreApi<AppState>['getState']

type BasicConfigKeys = 'language' | 'setLanguage' | 'apiConfig' | 'setApiConfig'

export function createBasicConfigActions(set: SetState): Pick<AppState, BasicConfigKeys> {
  return {
    language: 'en' as Language,
    setLanguage: language => set({ language }),
    apiConfig: { apiKey: '', llmBaseUrl: 'https://openrouter.ai/api/v1', proxyUrl: '' },
    setApiConfig: config => set(state => ({ apiConfig: { ...state.apiConfig, ...config } })),
  }
}

type AgentConfigKeys =
  | 'prompts' | 'setPrompt' | 'resetPromptsToDefault' | 'getDefaultPromptsForLanguage'
  | 'models' | 'setModel' | 'resetModelsToDefault'
  | 'temperatures' | 'setTemperature' | 'resetTemperaturesToDefault'
  | 'maxRounds' | 'setMaxRounds' | 'roundWaitSeconds' | 'setRoundWaitSeconds'
  | 'agentServerConfigs' | 'setAgentServerConfig'
  | 'agentVisionSettings' | 'setAgentVisionSetting'
  | 'literatureMode' | 'setLiteratureMode' | 'deepResearchModel' | 'setDeepResearchModel'
  | 'deepResearchLlmBaseUrl' | 'setDeepResearchLlmBaseUrl'
  | 'deepResearchApiKey' | 'setDeepResearchApiKey' | 'googleApiKey' | 'setGoogleApiKey'
  | 'pubmedEmail' | 'setPubmedEmail' | 'literatureMaxSearchRounds' | 'setLiteratureMaxSearchRounds'

export function createAgentConfigActions(
  set: SetState,
  get: GetState,
): Pick<AppState, AgentConfigKeys> {
  return {
    prompts: getDefaultPrompts('en'),
    setPrompt: (agent, prompt) => set(state => ({ prompts: { ...state.prompts, [agent]: prompt } })),
    resetPromptsToDefault: () => set(state => ({ prompts: getDefaultPrompts(state.language) })),
    getDefaultPromptsForLanguage: () => getDefaultPrompts(get().language),
    models: { ...DEFAULT_MODELS },
    setModel: (agent, model) => set(state => ({ models: { ...state.models, [agent]: model } })),
    resetModelsToDefault: () => set({ models: { ...DEFAULT_MODELS } }),
    temperatures: { ...DEFAULT_TEMPERATURES },
    setTemperature: (agent, temperature) => set(state => ({
      temperatures: { ...state.temperatures, [agent]: temperature },
    })),
    resetTemperaturesToDefault: () => set({ temperatures: { ...DEFAULT_TEMPERATURES } }),
    maxRounds: 3,
    setMaxRounds: maxRounds => set({ maxRounds }),
    roundWaitSeconds: 300,
    setRoundWaitSeconds: roundWaitSeconds => set({ roundWaitSeconds }),
    agentServerConfigs: {},
    setAgentServerConfig: (agent, config) => set(state => ({
      agentServerConfigs: { ...state.agentServerConfigs, [agent]: config },
    })),
    agentVisionSettings: { ...DEFAULT_VISION_SETTINGS },
    setAgentVisionSetting: (agent, enabled) => set(state => ({
      agentVisionSettings: { ...state.agentVisionSettings, [agent]: enabled },
    })),
    literatureMode: 'pubmed_llm',
    setLiteratureMode: literatureMode => set({ literatureMode }),
    deepResearchModel: 'o4-mini-deep-research',
    setDeepResearchModel: deepResearchModel => set({ deepResearchModel }),
    deepResearchLlmBaseUrl: '',
    setDeepResearchLlmBaseUrl: deepResearchLlmBaseUrl => set({ deepResearchLlmBaseUrl }),
    deepResearchApiKey: '',
    setDeepResearchApiKey: deepResearchApiKey => set({ deepResearchApiKey }),
    googleApiKey: '',
    setGoogleApiKey: googleApiKey => set({ googleApiKey }),
    pubmedEmail: '',
    setPubmedEmail: pubmedEmail => set({ pubmedEmail }),
    literatureMaxSearchRounds: 20,
    setLiteratureMaxSearchRounds: rounds => set({
      literatureMaxSearchRounds: Math.max(1, Math.min(50, rounds)),
    }),
  }
}
