import { useEffect, useRef, useState } from 'react'
import { BookOpen, Code2, Eye, FlaskConical, Microscope, Palette, Users, Wrench } from 'lucide-react'

import { apiFetch, applyStoreUpdates } from '../../hooks/useApi'
import { getDefaultPrompts, useTranslation } from '../../i18n/translations'
import {
  DEFAULT_MODELS, DEFAULT_TEMPERATURES, DEFAULT_VISION_SETTINGS, useAppStore,
  type AgentType,
} from '../../stores/appStore'
import { buildLiteratureConfigPayload, resolveDeepResearchServerConfig } from '../../utils/literatureConfig'
import {
  agentSettingsFromStore,
  loadLastAgentSettings,
  mergeSavedAgentSetting,
  saveLastAgentSettings,
  type AgentDefaultSecretsResponse,
} from '../../utils/lastAgentSettings'
import { transformAgentSettings } from '../../utils/transformLoadResponse'
import type { AgentSettingsResponse } from '../../types'

const colorStyles: Record<string, {
  bgSelected: string
  borderSelected: string
  bgIcon: string
  bgIconSelected: string
  textIcon: string
  textIconSelected: string
}> = {
  violet: {
    bgSelected: 'bg-violet-50 dark:bg-violet-950/40',
    borderSelected: 'border-violet-200 dark:border-violet-800',
    bgIcon: 'bg-slate-100 dark:bg-slate-800',
    bgIconSelected: 'bg-violet-100 dark:bg-violet-900/60',
    textIcon: 'text-slate-500 dark:text-slate-400',
    textIconSelected: 'text-violet-600 dark:text-violet-400'
  },
  red: {
    bgSelected: 'bg-red-50 dark:bg-red-950/40',
    borderSelected: 'border-red-200 dark:border-red-800',
    bgIcon: 'bg-slate-100 dark:bg-slate-800',
    bgIconSelected: 'bg-red-100 dark:bg-red-900/60',
    textIcon: 'text-slate-500 dark:text-slate-400',
    textIconSelected: 'text-red-600 dark:text-red-400'
  },
  green: {
    bgSelected: 'bg-green-50 dark:bg-green-950/40',
    borderSelected: 'border-green-200 dark:border-green-800',
    bgIcon: 'bg-slate-100 dark:bg-slate-800',
    bgIconSelected: 'bg-green-100 dark:bg-green-900/60',
    textIcon: 'text-slate-500 dark:text-slate-400',
    textIconSelected: 'text-green-600 dark:text-green-400'
  },
  amber: {
    bgSelected: 'bg-amber-50 dark:bg-amber-950/40',
    borderSelected: 'border-amber-200 dark:border-amber-800',
    bgIcon: 'bg-slate-100 dark:bg-slate-800',
    bgIconSelected: 'bg-amber-100 dark:bg-amber-900/60',
    textIcon: 'text-slate-500 dark:text-slate-400',
    textIconSelected: 'text-amber-600 dark:text-amber-400'
  },
  blue: {
    bgSelected: 'bg-blue-50 dark:bg-blue-950/40',
    borderSelected: 'border-blue-200 dark:border-blue-800',
    bgIcon: 'bg-slate-100 dark:bg-slate-800',
    bgIconSelected: 'bg-blue-100 dark:bg-blue-900/60',
    textIcon: 'text-slate-500 dark:text-slate-400',
    textIconSelected: 'text-blue-600 dark:text-blue-400'
  },
  pink: {
    bgSelected: 'bg-pink-50 dark:bg-pink-950/40',
    borderSelected: 'border-pink-200 dark:border-pink-800',
    bgIcon: 'bg-slate-100 dark:bg-slate-800',
    bgIconSelected: 'bg-pink-100 dark:bg-pink-900/60',
    textIcon: 'text-slate-500 dark:text-slate-400',
    textIconSelected: 'text-pink-600 dark:text-pink-400'
  },
  cyan: {
    bgSelected: 'bg-cyan-50 dark:bg-cyan-950/40',
    borderSelected: 'border-cyan-200 dark:border-cyan-800',
    bgIcon: 'bg-slate-100 dark:bg-slate-800',
    bgIconSelected: 'bg-cyan-100 dark:bg-cyan-900/60',
    textIcon: 'text-slate-500 dark:text-slate-400',
    textIconSelected: 'text-cyan-600 dark:text-cyan-400'
  },
  teal: {
    bgSelected: 'bg-teal-50 dark:bg-teal-950/40',
    borderSelected: 'border-teal-200 dark:border-teal-800',
    bgIcon: 'bg-slate-100 dark:bg-slate-800',
    bgIconSelected: 'bg-teal-100 dark:bg-teal-900/60',
    textIcon: 'text-slate-500 dark:text-slate-400',
    textIconSelected: 'text-teal-600 dark:text-teal-400'
  }
}

async function loadPreviousAgentDefaults(userId?: string): Promise<AgentSettingsResponse> {
  const storedDefaults = userId
    ? loadLastAgentSettings(userId)?.settings
    : undefined
  return storedDefaults ?? apiFetch<AgentSettingsResponse>('/user/agent-default-settings')
}

async function saveAgentDefaultSecrets(
  agent: AgentType,
  apiKey: string,
  deepResearchApiKey: string,
  googleApiKey: string,
) {
  await apiFetch<AgentDefaultSecretsResponse>('/user/agent-default-secrets', {
    method: 'POST',
    body: JSON.stringify({
      agent,
      api_key: apiKey,
      ...(agent === 'literature'
        ? { deep_research_api_key: deepResearchApiKey, google_api_key: googleApiKey }
        : {}),
    }),
  })
}

function resolveServerConfig(
  llmBaseUrl: string,
  apiKey: string,
  globalConfig: { llmBaseUrl: string; apiKey: string },
) {
  return llmBaseUrl !== globalConfig.llmBaseUrl || apiKey !== globalConfig.apiKey
    ? { llmBaseUrl, apiKey }
    : { llmBaseUrl: '', apiKey: '' }
}

function defaultSaveError(language: string) {
  return language === 'zh'
    ? '会话设置已保存，但新建分析的默认设置保存失败，请重试。'
    : 'Session settings were saved, but defaults for new analyses could not be saved. Please try again.'
}

export function usePromptsController() {
  const {
    currentSession, language, apiConfig, authUser,
    prompts, setPrompt,
    models, setModel,
    temperatures, setTemperature,
    agentServerConfigs, setAgentServerConfig,
    agentVisionSettings, setAgentVisionSetting,
    literatureMode, setLiteratureMode,
    deepResearchModel, setDeepResearchModel,
    deepResearchLlmBaseUrl, setDeepResearchLlmBaseUrl,
    deepResearchApiKey, setDeepResearchApiKey,
    googleApiKey, setGoogleApiKey,
    literatureMaxSearchRounds, setLiteratureMaxSearchRounds,
  } = useAppStore()
  const resolvedDeepResearch = resolveDeepResearchServerConfig({
    deepResearchLlmBaseUrl,
    deepResearchApiKey,
    apiConfig,
  })
  const llmModifyPending = useAppStore(s => s.llmModifyPending)
  const llmFailureInfo = useAppStore(s => s.llmFailureInfo)
  const { t } = useTranslation(language)
  const defaultPrompts = getDefaultPrompts(language)
  const [selectedAgent, setSelectedAgent] = useState<AgentType>('pi')
  const [editedPrompt, setEditedPrompt] = useState(prompts[selectedAgent] ?? defaultPrompts[selectedAgent] ?? '')
  const [editedModel, setEditedModel] = useState(models[selectedAgent] ?? DEFAULT_MODELS[selectedAgent] ?? '')
  const [editedTemperature, setEditedTemperature] = useState(temperatures[selectedAgent] ?? DEFAULT_TEMPERATURES[selectedAgent] ?? 0.5)
  const [editedLlmBaseUrl, setEditedLlmBaseUrl] = useState(agentServerConfigs[selectedAgent]?.llmBaseUrl || apiConfig.llmBaseUrl)
  const [editedApiKey, setEditedApiKey] = useState(agentServerConfigs[selectedAgent]?.apiKey || apiConfig.apiKey)
  const [editedContextWindow, setEditedContextWindow] = useState<number | undefined>(
    agentServerConfigs[selectedAgent]?.contextWindow,
  )
  const [contextWindowTouched, setContextWindowTouched] = useState(false)
  const [editedVisionEnabled, setEditedVisionEnabled] = useState(agentVisionSettings[selectedAgent] ?? DEFAULT_VISION_SETTINGS[selectedAgent] ?? true)
  const [showAgentApiKey, setShowAgentApiKey] = useState(false)
  const [showGoogleKey, setShowGoogleKey] = useState(false)
  const [showDeepResearchApiKey, setShowDeepResearchApiKey] = useState(false)
  const [serverExpanded, setServerExpanded] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
    }
  }, [])

  // Update edited prompt, model, temperature, and server config when selected agent changes
  useEffect(() => {
    setEditedPrompt(prompts[selectedAgent] ?? defaultPrompts[selectedAgent] ?? '')
    setEditedModel(models[selectedAgent] ?? DEFAULT_MODELS[selectedAgent] ?? '')
    setEditedTemperature(temperatures[selectedAgent] ?? DEFAULT_TEMPERATURES[selectedAgent] ?? 0.5)
    setEditedLlmBaseUrl(agentServerConfigs[selectedAgent]?.llmBaseUrl || apiConfig.llmBaseUrl)
    setEditedApiKey(agentServerConfigs[selectedAgent]?.apiKey || apiConfig.apiKey)
    if (!contextWindowTouched) {
      setEditedContextWindow(agentServerConfigs[selectedAgent]?.contextWindow)
    }
    setEditedVisionEnabled(agentVisionSettings[selectedAgent] ?? DEFAULT_VISION_SETTINGS[selectedAgent] ?? true)
  }, [
    selectedAgent, prompts, models, temperatures, defaultPrompts,
    agentServerConfigs, apiConfig, agentVisionSettings, contextWindowTouched,
  ])

  const agentConfig: {
    id: AgentType
    name: string
    description: string
    icon: typeof Users
    color: string
  }[] = [
    {
      id: 'pi',
      name: t('piAgent'),
      description: t('piAgentDesc'),
      icon: Users,
      color: 'violet'
    },
    {
      id: 'pathology',
      name: t('pathologyAgent'),
      description: t('pathologyAgentDesc'),
      icon: Microscope,
      color: 'red'
    },
    {
      id: 'bioinformatics',
      name: t('bioinformaticsAgent'),
      description: t('bioinformaticsAgentDesc'),
      icon: Code2,
      color: 'green'
    },
    {
      id: 'biochemistry',
      name: t('biochemistryAgent'),
      description: t('biochemistryAgentDesc'),
      icon: FlaskConical,
      color: 'amber'
    },
    {
      id: 'literature',
      name: t('literatureAgent'),
      description: t('literatureAgentDesc'),
      icon: BookOpen,
      color: 'blue'
    },
    {
      id: 'painter',
      name: t('painterAgent'),
      description: t('painterAgentDesc'),
      icon: Palette,
      color: 'pink'
    },
    {
      id: 'supervisor',
      name: t('supervisorAgent'),
      description: t('supervisorAgentDesc'),
      icon: Eye,
      color: 'cyan'
    },
    {
      id: 'technique',
      name: t('techniqueAgent'),
      description: t('techniqueAgentDesc'),
      icon: Wrench,
      color: 'teal'
    },
  ]

  // Auto-select the agent that had the LLM failure when modify is pending
  useEffect(() => {
    if (llmModifyPending && llmFailureInfo) {
      const agentMap: Record<string, AgentType> = {
        pi: 'pi',
        pathology: 'pathology',
        bioinformatics: 'bioinformatics',
        biochemistry: 'biochemistry',
        literature: 'literature',
      }
      const agentType = agentMap[llmFailureInfo.agent]
      if (agentType) {
        handleSelectAgent(agentType)
      }
    }
  }, [llmModifyPending, llmFailureInfo])

  const handleSelectAgent = (agentId: AgentType) => {
    setSelectedAgent(agentId)
    setEditedPrompt(prompts[agentId] ?? defaultPrompts[agentId] ?? '')
    setEditedModel(models[agentId] ?? DEFAULT_MODELS[agentId] ?? '')
    setEditedTemperature(temperatures[agentId] ?? DEFAULT_TEMPERATURES[agentId] ?? 0.5)
    setEditedLlmBaseUrl(agentServerConfigs[agentId]?.llmBaseUrl || apiConfig.llmBaseUrl)
    setEditedApiKey(agentServerConfigs[agentId]?.apiKey || apiConfig.apiKey)
    setEditedContextWindow(agentServerConfigs[agentId]?.contextWindow)
    setContextWindowTouched(false)
    setServerExpanded(false)
    setSaved(false)
  }
  const handleSave = async () => {
    setSaving(true)
    setSaveError(null)
    try {
      const serverConfig = resolveServerConfig(editedLlmBaseUrl, editedApiKey, apiConfig)

      let savedSettings: AgentSettingsResponse
      if (currentSession) {
        const response = await apiFetch<AgentSettingsResponse>(
          `/session/${currentSession.id}/agent-settings`,
          {
            method: 'POST',
            body: JSON.stringify({
              agent: selectedAgent,
              prompt: editedPrompt,
              model: editedModel,
              temperature: editedTemperature,
              llm_base_url: serverConfig.llmBaseUrl,
              api_key: serverConfig.apiKey,
              ...(contextWindowTouched
                ? { context_window: editedContextWindow ?? null }
                : {}),
              vision_enabled: editedVisionEnabled,
              literature_config: selectedAgent === 'literature'
                ? buildLiteratureConfigPayload(useAppStore.getState())
                : undefined,
              save_as_user_default: true,
            }),
          },
        )
        applyStoreUpdates(transformAgentSettings(response))
        if (response.default_settings_saved === false) {
          setContextWindowTouched(false)
          setSaveError(defaultSaveError(language))
          return
        }
        const previousDefaults = await loadPreviousAgentDefaults(authUser?.id)
        savedSettings = mergeSavedAgentSetting(previousDefaults, response, selectedAgent)
      } else {
        setPrompt(selectedAgent, editedPrompt)
        setModel(selectedAgent, editedModel)
        setTemperature(selectedAgent, editedTemperature)
        const storedContextWindow = contextWindowTouched
          ? editedContextWindow
          : editedModel !== (models[selectedAgent] ?? DEFAULT_MODELS[selectedAgent] ?? '') ? undefined : agentServerConfigs[selectedAgent]?.contextWindow
        setAgentServerConfig(selectedAgent, {
          ...serverConfig,
          ...(typeof storedContextWindow === 'number'
            ? { contextWindow: storedContextWindow }
            : {}),
        })
        setAgentVisionSetting(selectedAgent, editedVisionEnabled)
        savedSettings = agentSettingsFromStore()
      }
      if (!currentSession) {
        await saveAgentDefaultSecrets(
          selectedAgent,
          serverConfig.apiKey,
          deepResearchApiKey,
          googleApiKey,
        )
      }
      if (authUser) {
        saveLastAgentSettings(authUser.id, savedSettings)
      }
      setContextWindowTouched(false)
      setSaved(true)
      // If this save was triggered from LLM modify flow, mark it as saved only for the failing agent
      if (useAppStore.getState().llmModifyPending) {
        const failureAgent = useAppStore.getState().llmFailureInfo?.agent
        if (selectedAgent === failureAgent) {
          useAppStore.getState().setLlmModifySaved(true)
        }
      }
      if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
      savedTimerRef.current = setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      console.error('Failed to save prompt:', e)
      setSaveError('Failed to save configuration. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    // Get the default prompt for the current language
    const defaults = getDefaultPrompts(language)
    setEditedPrompt(defaults[selectedAgent])
    setPrompt(selectedAgent, defaults[selectedAgent])
    // Reset model to default
    setEditedModel(DEFAULT_MODELS[selectedAgent])
    setModel(selectedAgent, DEFAULT_MODELS[selectedAgent])
    // Reset temperature to default
    setEditedTemperature(DEFAULT_TEMPERATURES[selectedAgent])
    setTemperature(selectedAgent, DEFAULT_TEMPERATURES[selectedAgent])
    // Reset server config to global defaults
    setEditedLlmBaseUrl(apiConfig.llmBaseUrl)
    setEditedApiKey(apiConfig.apiKey)
    setEditedContextWindow(undefined)
    setContextWindowTouched(true)
    setAgentServerConfig(selectedAgent, { llmBaseUrl: '', apiKey: '' })
    // Reset vision setting to default
    setEditedVisionEnabled(DEFAULT_VISION_SETTINGS[selectedAgent])
    setAgentVisionSetting(selectedAgent, DEFAULT_VISION_SETTINGS[selectedAgent])

    // Reset literature config if on literature agent
    if (selectedAgent === 'literature') {
      setLiteratureMode('pubmed_llm')
      setDeepResearchModel('o4-mini-deep-research')
      setLiteratureMaxSearchRounds(20)
    }

    setSaved(false)
  }

  const selectedConfig = agentConfig.find(a => a.id === selectedAgent)
  if (!selectedConfig) {
    console.warn('Selected agent config not found:', selectedAgent)
    return null
  }
  const Icon = selectedConfig.icon
  const selectedStyles = colorStyles[selectedConfig.color] || colorStyles.violet

  return {
    language, t, apiConfig, agentConfig, agentServerConfigs, colorStyles, selectedAgent, handleSelectAgent, selectedConfig, Icon, selectedStyles,
    editedPrompt, setEditedPrompt, editedModel, setEditedModel,
    editedTemperature, setEditedTemperature, editedLlmBaseUrl, setEditedLlmBaseUrl,
    editedApiKey, setEditedApiKey, editedVisionEnabled, setEditedVisionEnabled,
    editedContextWindow, setEditedContextWindow, contextWindowTouched,
    setContextWindowTouched,
    showAgentApiKey, setShowAgentApiKey, showGoogleKey, setShowGoogleKey,
    showDeepResearchApiKey, setShowDeepResearchApiKey, serverExpanded, setServerExpanded,
    saving, saved, saveError, setSaved, handleSave, handleReset,
    literatureMode, setLiteratureMode, deepResearchModel, setDeepResearchModel,
    deepResearchLlmBaseUrl, setDeepResearchLlmBaseUrl, deepResearchApiKey, setDeepResearchApiKey,
    googleApiKey, setGoogleApiKey, literatureMaxSearchRounds, setLiteratureMaxSearchRounds,
    resolvedDeepResearch,
  }
}

export type PromptsController = NonNullable<ReturnType<typeof usePromptsController>>
