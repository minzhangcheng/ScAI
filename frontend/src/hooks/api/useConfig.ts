// System config, agent prompts, and per-user LLM credentials.
// Split out of frontend/src/hooks/useApi.ts verbatim.

import { useState, useCallback } from 'react'
import { useAppStore, AgentType } from '../../stores/appStore'
import { apiFetch } from './client'

export function useConfig() {
  const { apiConfig, setApiConfig, setSnapshotV2Enabled } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const getConfig = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<{
        configured: boolean
        has_api_key: boolean
        llm_base_url: string
        masked_api_key: string
        has_proxy_url: boolean
      }>('/config')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const saveConfig = useCallback(async (config = apiConfig) => {
    setLoading(true)
    setError(null)
    
    try {
      await apiFetch('/config', {
        method: 'POST',
        body: JSON.stringify({
          api_key: config.apiKey,
          llm_base_url: config.llmBaseUrl,
          proxy_url: config.proxyUrl || null,
        }),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [apiConfig])
  
  const checkStatus = useCallback(async () => {
    try {
      const status = await apiFetch<{
        configured: boolean
        has_api_key: boolean
        snapshot_v2_enabled: boolean
      }>('/config/status')
      setSnapshotV2Enabled(Boolean(status.snapshot_v2_enabled))
      return status
    } catch (e) {
      console.warn('[ScAI] checkStatus failed:', e)
      setSnapshotV2Enabled(false)
      return {
        configured: false,
        has_api_key: false,
        snapshot_v2_enabled: false,
      }
    }
  }, [setSnapshotV2Enabled])

  const validateApiKey = useCallback(async () => {
    try {
      const result = await apiFetch<{
        valid: boolean
        message: string
        status_code?: number
        uncertain?: boolean
      }>('/config/validate', {
        method: 'POST',
        body: JSON.stringify({
          api_key: apiConfig.apiKey,
          llm_base_url: apiConfig.llmBaseUrl,
          proxy_url: apiConfig.proxyUrl || null,
        }),
      })
      return result
    } catch {
      return {
        valid: false,
        message: 'Failed to reach validation endpoint',
        uncertain: true,
      }
    }
  }, [apiConfig])

  const validateSavedConfig = useCallback(async () => {
    // Validate the key already saved in the global config (admin has no local key,
    // e.g. after localStorage was cleared). No key is sent in the request body.
    try {
      return await apiFetch<{
        valid: boolean
        message: string
        uncertain?: boolean
      }>('/config/validate-saved', { method: 'POST' })
    } catch {
      return {
        valid: false,
        message: 'Failed to reach validation endpoint',
        uncertain: true,
      }
    }
  }, [])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return { apiConfig, setApiConfig, getConfig, saveConfig, checkStatus, validateApiKey, validateSavedConfig, loading, error, clearError }
}
// Prompts hook
export function usePrompts() {
  const { currentSession } = useAppStore()
  const { prompts, setPrompt, models, setModel, temperatures, setTemperature } = useAppStore()
  
  const savePrompt = useCallback(async (agent: AgentType, prompt: string) => {
    if (!currentSession) return
    
    await apiFetch(`/session/${currentSession.id}/prompts`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: currentSession.id,
        agent,
        prompt,
      }),
    })
    
    setPrompt(agent, prompt)
  }, [currentSession, setPrompt])
  
  const saveModel = useCallback(async (agent: AgentType, model: string) => {
    if (!currentSession) return
    
    await apiFetch(`/session/${currentSession.id}/models`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: currentSession.id,
        agent,
        model,
      }),
    })
    
    setModel(agent, model)
  }, [currentSession, setModel])
  
  const saveTemperature = useCallback(async (agent: AgentType, temperature: number) => {
    if (!currentSession) return
    
    await apiFetch(`/session/${currentSession.id}/temperatures`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: currentSession.id,
        agent,
        temperature,
      }),
    })
    
    setTemperature(agent, temperature)
  }, [currentSession, setTemperature])
  
  const loadPrompts = useCallback(async () => {
    if (!currentSession) return prompts
    
    try {
      const data = await apiFetch<Record<string, string>>(
        `/session/${currentSession.id}/prompts`
      )
      
      Object.entries(data).forEach(([agent, prompt]) => {
        setPrompt(agent as AgentType, prompt)
      })
      
      return data
    } catch {
      return prompts
    }
  }, [currentSession, prompts, setPrompt])
  
  return { prompts, setPrompt, savePrompt, models, setModel, saveModel, temperatures, setTemperature, saveTemperature, loadPrompts }
}

// User config hook
interface UserConfigStatus {
  configured: boolean
  has_api_key?: boolean
  llm_base_url?: string
  masked_api_key?: string
  has_proxy_url?: boolean
}

interface ValidateResult {
  valid: boolean
  uncertain?: boolean
  message?: string
}

export function useUserConfig() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const getUserConfig = useCallback(async (): Promise<UserConfigStatus> => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<UserConfigStatus>('/user/config', { method: 'GET' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get user config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const saveUserConfig = useCallback(async (
    api_key: string,
    llm_base_url: string,
    proxy_url: string
  ): Promise<void> => {
    setLoading(true)
    setError(null)
    try {
      await apiFetch<void>('/user/config', {
        method: 'POST',
        body: JSON.stringify({ api_key, llm_base_url, proxy_url: proxy_url || null }),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save user config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const getUserConfigStatus = useCallback(async (): Promise<UserConfigStatus> => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<UserConfigStatus>('/user/config/status', { method: 'GET' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get user config status')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const validateUserConfig = useCallback(async (
    api_key: string,
    llm_base_url: string,
    proxy_url: string
  ): Promise<ValidateResult> => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<ValidateResult>('/user/config/validate', {
        method: 'POST',
        body: JSON.stringify({ api_key, llm_base_url, proxy_url: proxy_url || null }),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to validate user config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const validateSavedConfig = useCallback(async (): Promise<ValidateResult> => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<ValidateResult>('/user/config/validate-saved', { method: 'POST' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to validate saved config')
      return { valid: false, message: 'Failed to reach validation endpoint', uncertain: true }
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteUserConfig = useCallback(async (): Promise<void> => {
    setLoading(true)
    setError(null)
    try {
      await apiFetch<void>('/user/config', { method: 'DELETE' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete user config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    loading, error,
    getUserConfig, saveUserConfig, getUserConfigStatus,
    validateUserConfig, validateSavedConfig, deleteUserConfig,
  }
}
