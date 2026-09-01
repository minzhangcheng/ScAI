import { getDefaultPrompts, type Language } from '../i18n/translations'
import {
  DEFAULT_MODELS,
  DEFAULT_TEMPERATURES,
  DEFAULT_VISION_SETTINGS,
  useAppStore,
  type AgentType,
} from '../stores/appStore'
import type { AgentSettingsResponse } from '../types'

const STORAGE_PREFIX = 'scai-last-agent-settings:'
const AGENTS: readonly AgentType[] = [
  'pi', 'pathology', 'bioinformatics', 'biochemistry',
  'literature', 'painter', 'supervisor', 'technique',
]

export interface StoredAgentSettings {
  version: 1
  savedAt: number
  settings: AgentSettingsResponse
}

export interface AgentDefaultSecretsResponse {
  agent_api_keys: Partial<Record<AgentType, string>>
  literature: {
    deep_research_api_key?: string
    google_api_key?: string
  }
}

function storageKey(userId: string) {
  return `${STORAGE_PREFIX}${encodeURIComponent(userId)}`
}

export function defaultAgentSettings(language: Language): AgentSettingsResponse {
  return {
    prompts: { ...getDefaultPrompts(language) },
    models: { ...DEFAULT_MODELS },
    temperatures: { ...DEFAULT_TEMPERATURES },
    agent_server_configs: Object.fromEntries(AGENTS.map(agent => [agent, {
      llm_base_url: '',
      api_key: '',
      context_window: null,
    }])) as AgentSettingsResponse['agent_server_configs'],
    vision_settings: { ...DEFAULT_VISION_SETTINGS },
    literature_config: {
      mode: 'pubmed_llm',
      deep_research_model: 'o4-mini-deep-research',
      deep_research_llm_base_url: '',
      deep_research_api_key: '',
      google_api_key: '',
      pubmed_email: '',
      max_search_rounds: 20,
    },
  }
}

export function withoutAgentSecrets(settings: AgentSettingsResponse): AgentSettingsResponse {
  return {
    ...settings,
    agent_server_configs: Object.fromEntries(
      Object.entries(settings.agent_server_configs).map(([agent, config]) => [agent, {
        ...(config || {}),
        api_key: '',
      }]),
    ) as AgentSettingsResponse['agent_server_configs'],
    literature_config: {
      ...settings.literature_config,
      deep_research_api_key: '',
      google_api_key: '',
    },
  }
}

export function loadLastAgentSettings(userId: string): StoredAgentSettings | null {
  try {
    const raw = window.localStorage.getItem(storageKey(userId))
    if (!raw) return null
    const parsed = JSON.parse(raw) as StoredAgentSettings
    if (parsed?.version !== 1 || !parsed.settings) return null
    return { ...parsed, settings: withoutAgentSecrets(parsed.settings) }
  } catch {
    return null
  }
}

export function saveLastAgentSettings(userId: string, settings: AgentSettingsResponse): void {
  const payload: StoredAgentSettings = {
    version: 1,
    savedAt: Date.now(),
    settings: withoutAgentSecrets(settings),
  }
  try {
    window.localStorage.setItem(storageKey(userId), JSON.stringify(payload))
  } catch {
    // The backend Session remains authoritative; browser defaults are best effort.
  }
}

export function agentSettingsFromStore(): AgentSettingsResponse {
  const state = useAppStore.getState()
  return {
    prompts: { ...state.prompts },
    models: { ...state.models },
    temperatures: { ...state.temperatures },
    agent_server_configs: Object.fromEntries(AGENTS.map(agent => {
      const config = state.agentServerConfigs[agent]
      return [agent, {
        llm_base_url: config?.llmBaseUrl || '',
        api_key: config?.apiKey || '',
        context_window: config?.contextWindow ?? null,
      }]
    })) as AgentSettingsResponse['agent_server_configs'],
    vision_settings: { ...state.agentVisionSettings },
    literature_config: {
      mode: state.literatureMode,
      deep_research_model: state.deepResearchModel,
      deep_research_llm_base_url: state.deepResearchLlmBaseUrl,
      deep_research_api_key: state.deepResearchApiKey,
      google_api_key: state.googleApiKey,
      pubmed_email: state.pubmedEmail,
      max_search_rounds: state.literatureMaxSearchRounds,
    },
  }
}

export function mergeSavedAgentSetting(
  previous: AgentSettingsResponse,
  saved: AgentSettingsResponse,
  agent: AgentType,
): AgentSettingsResponse {
  return {
    ...previous,
    prompts: { ...previous.prompts, [agent]: saved.prompts[agent] },
    models: { ...previous.models, [agent]: saved.models[agent] },
    temperatures: { ...previous.temperatures, [agent]: saved.temperatures[agent] },
    agent_server_configs: {
      ...previous.agent_server_configs,
      [agent]: saved.agent_server_configs[agent],
    },
    vision_settings: {
      ...previous.vision_settings,
      [agent]: saved.vision_settings[agent],
    },
    literature_config: agent === 'literature'
      ? { ...saved.literature_config }
      : { ...previous.literature_config },
  }
}

export function mergeAgentDefaultSecrets(
  settings: AgentSettingsResponse,
  secrets: AgentDefaultSecretsResponse,
): AgentSettingsResponse {
  const agentServerConfigs = { ...settings.agent_server_configs }
  for (const agent of AGENTS) {
    const apiKey = secrets.agent_api_keys?.[agent] || ''
    agentServerConfigs[agent] = {
      ...(agentServerConfigs[agent] || {}),
      api_key: apiKey,
    }
  }
  return {
    ...settings,
    agent_server_configs: agentServerConfigs,
    literature_config: {
      ...settings.literature_config,
      deep_research_api_key: secrets.literature?.deep_research_api_key || '',
      google_api_key: secrets.literature?.google_api_key || '',
    },
  }
}

export function newSessionAgentSettingsPayload(settings: AgentSettingsResponse) {
  const sanitized = withoutAgentSecrets(settings)
  return {
    prompts: sanitized.prompts,
    models: sanitized.models,
    temperatures: sanitized.temperatures,
    agent_server_configs: Object.fromEntries(
      Object.entries(sanitized.agent_server_configs).map(([agent, config]) => [agent, {
        llm_base_url: config?.llm_base_url || '',
        ...(typeof config?.context_window === 'number'
          ? { context_window: config.context_window }
          : {}),
      }]),
    ),
    vision_settings: sanitized.vision_settings,
    literature_config: {
      mode: sanitized.literature_config.mode,
      deep_research_model: sanitized.literature_config.deep_research_model,
      deep_research_llm_base_url: sanitized.literature_config.deep_research_llm_base_url,
      pubmed_email: sanitized.literature_config.pubmed_email,
      max_search_rounds: sanitized.literature_config.max_search_rounds,
    },
  }
}
