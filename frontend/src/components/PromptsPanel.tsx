import { useEffect, useState } from 'react'

import { apiFetch, applyStoreUpdates } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import type { AgentSettingsResponse } from '../types'
import {
  loadLastAgentSettings,
  mergeAgentDefaultSecrets,
  type AgentDefaultSecretsResponse,
} from '../utils/lastAgentSettings'
import { transformAgentSettings } from '../utils/transformLoadResponse'
import { PromptEditor } from './prompts/PromptEditor'
import { usePromptsController } from './prompts/usePromptsController'

function PromptsEditor() {
  const controller = usePromptsController()
  return controller ? <PromptEditor controller={controller} /> : null
}

export function PromptsPanel() {
  const currentSessionId = useAppStore(s => s.currentSession?.id)
  const authUser = useAppStore(s => s.authUser)
  const hydrated = useAppStore(s => s._hydrated)
  const language = useAppStore(s => s.language)
  const [readyKey, setReadyKey] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [retryGeneration, setRetryGeneration] = useState(0)
  const targetKey = currentSessionId ? `session:${currentSessionId}` : `defaults:${authUser?.id || ''}`

  useEffect(() => {
    if (!hydrated || !authUser) return
    let active = true
    setReadyKey(null)
    setLoadError(null)

    const load = async () => {
      try {
        let settings: AgentSettingsResponse
        if (currentSessionId) {
          settings = await apiFetch<AgentSettingsResponse>(
            `/session/${currentSessionId}/agent-settings`,
          )
        } else {
          const stored = loadLastAgentSettings(authUser.id)
          const [baseSettings, secrets] = await Promise.all([
            stored
              ? Promise.resolve(stored.settings)
              : apiFetch<AgentSettingsResponse>('/user/agent-default-settings'),
            apiFetch<AgentDefaultSecretsResponse>('/user/agent-default-secrets'),
          ])
          const pubmedEmail = useAppStore.getState().pubmedEmail
          settings = mergeAgentDefaultSecrets(
            !stored && pubmedEmail
              ? {
                  ...baseSettings,
                  literature_config: {
                    ...baseSettings.literature_config,
                    pubmed_email: pubmedEmail,
                  },
                }
              : baseSettings,
            secrets,
          )
        }
        if (!active) return
        useAppStore.setState({
          agentServerConfigs: {},
          deepResearchApiKey: '',
          googleApiKey: '',
        })
        applyStoreUpdates(transformAgentSettings(settings))
        setReadyKey(targetKey)
      } catch (error) {
        if (active) {
          setLoadError(error instanceof Error ? error.message : 'Failed to load Agent settings')
        }
      }
    }
    void load()
    return () => { active = false }
  }, [authUser, currentSessionId, hydrated, language, retryGeneration, targetKey])

  if (!hydrated || !authUser || readyKey !== targetKey) {
    return (
      <div className="flex min-h-48 items-center justify-center text-sm text-surface-500 dark:text-surface-400">
        {loadError ? (
          <div className="text-center">
            <p>{loadError}</p>
            <button
              type="button"
              onClick={() => setRetryGeneration(value => value + 1)}
              className="mt-3 rounded-lg bg-accent-600 px-4 py-2 text-white hover:bg-accent-700"
            >
              {language === 'zh' ? '重试' : 'Retry'}
            </button>
          </div>
        ) : (language === 'zh' ? '正在加载 Agent 设置…' : 'Loading Agent settings…')}
      </div>
    )
  }

  return <PromptsEditor key={targetKey} />
}
