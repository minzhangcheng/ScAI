import type { AppState } from '../stores/appStore'

// API contract: must stay in sync with MASKED_SECRET_VALUE in
// backend/utils/settings_provider.py (the masking sentinel for secret fields).
export const MASKED_SECRET_VALUE = '********'

type LiteratureConfigState = Pick<
  AppState,
  | 'literatureMode'
  | 'deepResearchModel'
  | 'deepResearchLlmBaseUrl'
  | 'deepResearchApiKey'
  | 'googleApiKey'
  | 'pubmedEmail'
  | 'literatureMaxSearchRounds'
  | 'apiConfig'
>

export function resolveDeepResearchServerConfig(
  state: Pick<AppState, 'deepResearchLlmBaseUrl' | 'deepResearchApiKey' | 'apiConfig'>
): { deepResearchLlmBaseUrl: string; deepResearchApiKey: string } {
  return {
    deepResearchLlmBaseUrl: state.deepResearchLlmBaseUrl || state.apiConfig.llmBaseUrl || 'https://openrouter.ai/api/v1',
    deepResearchApiKey: state.deepResearchApiKey || state.apiConfig.apiKey || '',
  }
}

export function buildLiteratureConfigPayload(state: LiteratureConfigState) {
  return {
    mode: state.literatureMode,
    deep_research_model: state.deepResearchModel,
    deep_research_llm_base_url: state.deepResearchLlmBaseUrl || '',
    deep_research_api_key: isMaskedSecret(state.deepResearchApiKey)
      ? MASKED_SECRET_VALUE
      : state.deepResearchApiKey || '',
    google_api_key: isMaskedSecret(state.googleApiKey)
      ? MASKED_SECRET_VALUE
      : state.googleApiKey || '',
    pubmed_email: state.pubmedEmail,
    max_search_rounds: state.literatureMaxSearchRounds,
  }
}

export function isMaskedSecret(value: string | undefined) {
  return value === MASKED_SECRET_VALUE
}
