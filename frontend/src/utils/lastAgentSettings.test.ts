import { beforeEach, describe, expect, it } from 'vitest'

import type { AgentSettingsResponse } from '../types'
import {
  defaultAgentSettings,
  loadLastAgentSettings,
  mergeSavedAgentSetting,
  newSessionAgentSettingsPayload,
  saveLastAgentSettings,
} from './lastAgentSettings'

describe('last Agent settings persistence', () => {
  beforeEach(() => localStorage.clear())

  it('stores nonsecret settings per user and excludes every API key', () => {
    const settings: AgentSettingsResponse = defaultAgentSettings('en')
    settings.models.pi = 'saved-model'
    settings.agent_server_configs.pi = {
      llm_base_url: 'http://localhost:8000/v1',
      api_key: 'agent-secret',
    }
    settings.literature_config.deep_research_api_key = 'research-secret'
    settings.literature_config.google_api_key = 'google-secret'

    saveLastAgentSettings('user-a', settings)

    const raw = localStorage.getItem('scai-last-agent-settings:user-a') || ''
    expect(raw).not.toContain('agent-secret')
    expect(raw).not.toContain('research-secret')
    expect(raw).not.toContain('google-secret')
    expect(loadLastAgentSettings('user-a')?.settings.models.pi).toBe('saved-model')
    expect(loadLastAgentSettings('user-b')).toBeNull()
    expect(newSessionAgentSettingsPayload(settings)).not.toHaveProperty(
      'agent_server_configs.pi.api_key',
    )
  })

  it('updates only the Agent that was explicitly saved from an existing analysis', () => {
    const previous = defaultAgentSettings('en')
    previous.models.pathology = 'previous-pathology'
    const session = defaultAgentSettings('en')
    session.models.pi = 'saved-pi'
    session.models.pathology = 'unselected-pathology'

    const merged = mergeSavedAgentSetting(previous, session, 'pi')

    expect(merged.models.pi).toBe('saved-pi')
    expect(merged.models.pathology).toBe('previous-pathology')
  })
})
