import { useState, useEffect, useRef } from 'react'
import { usePrompts } from '../hooks/useApi'
import { useAppStore, AgentType, DEFAULT_MODELS, DEFAULT_TEMPERATURES, DEFAULT_VISION_SETTINGS } from '../stores/appStore'
import { useTranslation, getDefaultPrompts } from '../i18n/translations'
import { apiFetch } from '../hooks/useApi'
import { buildLiteratureConfigPayload, resolveDeepResearchServerConfig } from '../utils/literatureConfig'
import {
  Users,
  Microscope,
  Code2,
  FlaskConical,
  BookOpen,
  Palette,
  Eye,
  EyeOff,
  Save,
  RotateCcw,
  Cpu,
  Thermometer,
  Search,
  AlertCircle,
  Server,
  Key,
  ChevronDown,
  ChevronRight,
  Wrench,
} from 'lucide-react'

// Color styles mapping for Tailwind - use complete class names to prevent purging
const colorStyles: Record<string, {
  bgSelected: string
  borderSelected: string
  bgIcon: string
  bgIconSelected: string
  textIcon: string
  textIconSelected: string
}> = {
  violet: {
    bgSelected: 'bg-violet-50',
    borderSelected: 'border-violet-200',
    bgIcon: 'bg-slate-100',
    bgIconSelected: 'bg-violet-100',
    textIcon: 'text-slate-500',
    textIconSelected: 'text-violet-600'
  },
  red: {
    bgSelected: 'bg-red-50',
    borderSelected: 'border-red-200',
    bgIcon: 'bg-slate-100',
    bgIconSelected: 'bg-red-100',
    textIcon: 'text-slate-500',
    textIconSelected: 'text-red-600'
  },
  green: {
    bgSelected: 'bg-green-50',
    borderSelected: 'border-green-200',
    bgIcon: 'bg-slate-100',
    bgIconSelected: 'bg-green-100',
    textIcon: 'text-slate-500',
    textIconSelected: 'text-green-600'
  },
  amber: {
    bgSelected: 'bg-amber-50',
    borderSelected: 'border-amber-200',
    bgIcon: 'bg-slate-100',
    bgIconSelected: 'bg-amber-100',
    textIcon: 'text-slate-500',
    textIconSelected: 'text-amber-600'
  },
  blue: {
    bgSelected: 'bg-blue-50',
    borderSelected: 'border-blue-200',
    bgIcon: 'bg-slate-100',
    bgIconSelected: 'bg-blue-100',
    textIcon: 'text-slate-500',
    textIconSelected: 'text-blue-600'
  },
  pink: {
    bgSelected: 'bg-pink-50',
    borderSelected: 'border-pink-200',
    bgIcon: 'bg-slate-100',
    bgIconSelected: 'bg-pink-100',
    textIcon: 'text-slate-500',
    textIconSelected: 'text-pink-600'
  },
  cyan: {
    bgSelected: 'bg-cyan-50',
    borderSelected: 'border-cyan-200',
    bgIcon: 'bg-slate-100',
    bgIconSelected: 'bg-cyan-100',
    textIcon: 'text-slate-500',
    textIconSelected: 'text-cyan-600'
  },
  teal: {
    bgSelected: 'bg-teal-50',
    borderSelected: 'border-teal-200',
    bgIcon: 'bg-slate-100',
    bgIconSelected: 'bg-teal-100',
    textIcon: 'text-slate-500',
    textIconSelected: 'text-teal-600'
  }
}

export function PromptsPanel() {
  const {
    currentSession, language, apiConfig,
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
  const { savePrompt, saveModel, saveTemperature } = usePrompts()
  const { t } = useTranslation(language)
  const defaultPrompts = getDefaultPrompts(language)
  const [selectedAgent, setSelectedAgent] = useState<AgentType>('pi')
  const [editedPrompt, setEditedPrompt] = useState(prompts[selectedAgent] ?? defaultPrompts[selectedAgent] ?? '')
  const [editedModel, setEditedModel] = useState(models[selectedAgent] ?? DEFAULT_MODELS[selectedAgent] ?? '')
  const [editedTemperature, setEditedTemperature] = useState(temperatures[selectedAgent] ?? DEFAULT_TEMPERATURES[selectedAgent] ?? 0.5)
  const [editedLlmBaseUrl, setEditedLlmBaseUrl] = useState(agentServerConfigs[selectedAgent]?.llmBaseUrl || apiConfig.llmBaseUrl)
  const [editedApiKey, setEditedApiKey] = useState(agentServerConfigs[selectedAgent]?.apiKey || apiConfig.apiKey)
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
    setEditedVisionEnabled(agentVisionSettings[selectedAgent] ?? DEFAULT_VISION_SETTINGS[selectedAgent] ?? true)
  }, [selectedAgent, prompts, models, temperatures, defaultPrompts, agentServerConfigs, apiConfig, agentVisionSettings])

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
    setServerExpanded(false)
    setSaved(false)
  }

  const handleSave = async () => {
    setSaving(true)
    setSaveError(null)
    try {
      // Only persist per-agent server config if different from global defaults
      const isCustomServer = editedLlmBaseUrl !== apiConfig.llmBaseUrl || editedApiKey !== apiConfig.apiKey
      if (isCustomServer) {
        setAgentServerConfig(selectedAgent, { llmBaseUrl: editedLlmBaseUrl, apiKey: editedApiKey })
      } else {
        // Clear per-agent override (use global)
        setAgentServerConfig(selectedAgent, { llmBaseUrl: '', apiKey: '' })
      }

      if (currentSession) {
        await savePrompt(selectedAgent, editedPrompt)
        await saveModel(selectedAgent, editedModel)
        await saveTemperature(selectedAgent, editedTemperature)

        // Save per-agent server config to backend
        try {
          await apiFetch(`/session/${currentSession.id}/agent-server-config`, {
            method: 'POST',
            body: JSON.stringify({
              agent: selectedAgent,
              llm_base_url: editedLlmBaseUrl,
              api_key: editedApiKey,
            }),
          })
        } catch (e) {
          console.error('Failed to save agent server config:', e)
          throw e
        }

        // Save vision setting to backend
        try {
          await apiFetch(`/session/${currentSession.id}/vision-settings`, {
            method: 'POST',
            body: JSON.stringify({
              agent: selectedAgent,
              vision_enabled: editedVisionEnabled,
            }),
          })
        } catch (e) {
          console.error('Failed to save vision setting:', e)
          throw e
        }
        setAgentVisionSetting(selectedAgent, editedVisionEnabled)

        // Also save literature config if on literature agent
        if (selectedAgent === 'literature') {
          try {
            const literaturePayload = buildLiteratureConfigPayload(useAppStore.getState())
            await apiFetch(`/session/${currentSession.id}/literature-config`, {
              method: 'POST',
              body: JSON.stringify(literaturePayload),
            })
          } catch (e) {
            console.error('Failed to save literature config:', e)
            throw e
          }
        }
      } else {
        setPrompt(selectedAgent, editedPrompt)
        setModel(selectedAgent, editedModel)
        setTemperature(selectedAgent, editedTemperature)
        setAgentVisionSetting(selectedAgent, editedVisionEnabled)
      }
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
    setAgentServerConfig(selectedAgent, { llmBaseUrl: '', apiKey: '' })
    // Reset vision setting to default
    setEditedVisionEnabled(DEFAULT_VISION_SETTINGS[selectedAgent])
    setAgentVisionSetting(selectedAgent, DEFAULT_VISION_SETTINGS[selectedAgent])

    // Reset literature config if on literature agent
    if (selectedAgent === 'literature') {
      setLiteratureMode(googleApiKey ? 'gemini_deep_research' : 'pubmed_llm')
      setDeepResearchModel('')
      setLiteratureMaxSearchRounds(5)
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

  return (
    <div className="flex gap-6 h-full">
      {/* Agent List */}
      <div className="w-64 shrink-0">
        <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wide mb-3">
          {t('selectAgent')}
        </h3>
        <div className="space-y-2">
          {agentConfig.map((agent) => {
            const AgentIcon = agent.icon
            const isSelected = selectedAgent === agent.id
            const styles = colorStyles[agent.color] || colorStyles.violet

            return (
              <button
                key={agent.id}
                onClick={() => handleSelectAgent(agent.id)}
                className={`w-full flex items-start gap-3 p-3 rounded-xl text-left transition-all ${
                  isSelected
                    ? `${styles.bgSelected} border-2 ${styles.borderSelected}`
                    : 'hover:bg-slate-50 border-2 border-transparent'
                }`}
              >
                <div className={`p-2 rounded-lg ${
                  isSelected ? styles.bgIconSelected : styles.bgIcon
                }`}>
                  <AgentIcon className={`w-4 h-4 ${
                    isSelected ? styles.textIconSelected : styles.textIcon
                  }`} />
                </div>
                <div>
                  <p className={`font-medium ${
                    isSelected ? 'text-slate-900' : 'text-slate-700'
                  }`}>
                    {agent.name}
                  </p>
                  <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">
                    {agent.description}
                  </p>
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Prompt Editor */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${selectedStyles.bgIconSelected}`}>
              <Icon className={`w-5 h-5 ${selectedStyles.textIconSelected}`} />
            </div>
            <div>
              <h3 className="font-display font-semibold text-slate-900">{selectedConfig.name}</h3>
              <p className="text-sm text-slate-500">{selectedConfig.description}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-3 py-2 text-slate-600 hover:bg-slate-100 rounded-lg text-sm"
            >
              <RotateCcw className="w-4 h-4" />
              {t('reset')}
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                saved
                  ? 'bg-green-100 text-green-700'
                  : 'bg-accent-600 text-white hover:bg-accent-700'
              }`}
            >
              <Save className="w-4 h-4" />
              {saved ? t('saved') : saving ? t('saving') : t('save')}
            </button>
          </div>
          {saveError && (
            <p className="text-red-600 text-xs mt-1">{saveError}</p>
          )}
        </div>

        {/* Literature Agent: Research Mode Selector */}
        {selectedAgent === 'literature' && (
          <>
            {/* Research Mode Selector */}
            <div className="mb-4 p-4 bg-blue-50/50 rounded-xl border border-blue-100">
              <div className="flex items-center gap-2 mb-3">
                <Search className="w-4 h-4 text-blue-600" />
                <label className="text-sm font-medium text-slate-700">
                  {t('literatureResearchMode')}
                </label>
              </div>

              <div className="space-y-2">
                {/* Disabled */}
                <div className={`rounded-lg transition-all ${
                  literatureMode === 'disabled' ? 'bg-white border border-blue-200 shadow-sm' : 'hover:bg-white/50'
                }`}>
                  <label className="flex items-start gap-3 p-3 cursor-pointer">
                    <input
                      type="radio"
                      name="literatureMode"
                      value="disabled"
                      checked={literatureMode === 'disabled'}
                      onChange={() => { setLiteratureMode('disabled'); setSaved(false) }}
                      className="mt-0.5 accent-blue-600"
                    />
                    <div className="flex-1">
                      <span className="text-sm font-medium text-slate-800">{language === 'zh' ? '不启用' : 'Disabled'}</span>
                      <p className="text-xs text-slate-500 mt-0.5">{language === 'zh' ? '不使用文献 Agent 进行文献检索' : 'Do not use Literature Agent for literature search'}</p>
                    </div>
                  </label>
                </div>

                {/* Gemini Deep Research */}
                <div className={`rounded-lg transition-all ${
                  literatureMode === 'gemini_deep_research' ? 'bg-white border border-blue-200 shadow-sm' : 'hover:bg-white/50'
                }`}>
                  <label className="flex items-start gap-3 p-3 cursor-pointer">
                    <input
                      type="radio"
                      name="literatureMode"
                      value="gemini_deep_research"
                      checked={literatureMode === 'gemini_deep_research'}
                      onChange={() => { setLiteratureMode('gemini_deep_research'); setSaved(false) }}
                      className="mt-0.5 accent-blue-600"
                    />
                    <div className="flex-1">
                      <span className="text-sm font-medium text-slate-800">{t('literatureGeminiDeepResearch')}</span>
                      <p className="text-xs text-slate-500 mt-0.5">{t('literatureGeminiDesc')}</p>
                      {!googleApiKey && (
                        <p className="text-xs text-amber-600 mt-1">
                          {t('literatureGeminiNote')}
                        </p>
                      )}
                      <div className="flex items-start gap-2 mt-2 p-2 bg-amber-50 border border-amber-200 rounded-lg">
                        <AlertCircle className="w-3.5 h-3.5 text-amber-500 flex-shrink-0 mt-0.5" />
                        <p className="text-xs text-amber-700">
                          {language === 'zh'
                            ? 'Deep Research 每次检索耗时较长（数分钟至最多60分钟），费用约 $2-5/次。'
                            : 'Deep Research takes minutes to complete (up to 60 min) and costs ~$2-5 per task.'}{' '}
                          <a href="https://ai.google.dev/gemini-api/docs/deep-research" target="_blank" rel="noopener noreferrer" className="text-amber-800 underline hover:text-amber-900">
                            {language === 'zh' ? '参考 Google 文档' : 'See Google docs'}
                          </a>
                        </p>
                      </div>
                      {literatureMode === 'gemini_deep_research' && (
                        <div className="mt-2 space-y-1.5">
                          <label className="block text-xs font-medium text-slate-600">
                            {t('literatureGoogleApiKey')}
                          </label>
                          <div className="relative">
                            <input
                              type={showGoogleKey ? 'text' : 'password'}
                              value={googleApiKey}
                              onChange={(e) => { setGoogleApiKey(e.target.value); setSaved(false) }}
                              placeholder="AIza..."
                              className="w-full px-3 py-1.5 pr-10 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                            <button
                              type="button"
                              onClick={() => setShowGoogleKey(!showGoogleKey)}
                              className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600"
                            >
                              {showGoogleKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                            </button>
                          </div>
                          <p className="text-xs text-slate-400">
                            <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                              {language === 'zh' ? '获取 Google AI API Key' : 'Get Google AI API Key'}
                            </a>
                          </p>
                        </div>
                      )}
                    </div>
                  </label>
                </div>

                {/* Deep Research (OpenRouter) */}
                <div className={`rounded-lg transition-all ${
                  literatureMode === 'deep_research' ? 'bg-white border border-blue-200 shadow-sm' : 'hover:bg-white/50'
                }`}>
                  <label className="flex items-start gap-3 p-3 cursor-pointer">
                    <input
                      type="radio"
                      name="literatureMode"
                      value="deep_research"
                      checked={literatureMode === 'deep_research'}
                      onChange={() => { setLiteratureMode('deep_research'); setSaved(false) }}
                      className="mt-0.5 accent-blue-600"
                    />
                    <div className="flex-1">
                      <span className="text-sm font-medium text-slate-800">{t('literatureDeepResearch')}</span>
                      <p className="text-xs text-slate-500 mt-0.5">{t('literatureDeepResearchDesc')}</p>
                      {literatureMode === 'deep_research' && (
                        <div className="mt-2 space-y-2.5">
                          <div className="space-y-1.5">
                            <label className="block text-xs font-medium text-slate-600">
                              {t('literatureDeepResearchModel')}
                            </label>
                            <input
                              type="text"
                              value={deepResearchModel}
                              placeholder={t('literatureDeepResearchModelPlaceholder')}
                              onChange={(e) => { setDeepResearchModel(e.target.value); setSaved(false) }}
                              className="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                            <p className="text-xs text-slate-400">{t('literatureDeepResearchNote')}</p>
                          </div>
                          {/* Collapsible server/key for deep research */}
                          <div className="border border-slate-200 rounded-lg">
                            <button
                              type="button"
                              onClick={() => setServerExpanded(!serverExpanded)}
                              className="flex items-center gap-2 w-full text-left p-2"
                            >
                              {serverExpanded ? <ChevronDown className="w-3.5 h-3.5 text-slate-500" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />}
                              <Server className="w-3.5 h-3.5 text-slate-500" />
                              <span className="text-xs font-medium text-slate-700">
                                {t('literatureDeepResearchServer')} / {t('literatureDeepResearchApiKey')}
                              </span>
                              {!serverExpanded && (
                                <span className="text-xs text-slate-500 ml-auto truncate max-w-[200px]">
                                  {resolvedDeepResearch.deepResearchLlmBaseUrl}
                                </span>
                              )}
                            </button>
                            {serverExpanded && (
                              <div className="px-2 pb-2 space-y-1.5">
                                <input
                                  type="text"
                                  value={resolvedDeepResearch.deepResearchLlmBaseUrl}
                                  onChange={(e) => { setDeepResearchLlmBaseUrl(e.target.value); setSaved(false) }}
                                  className="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                />
                                <div className="relative">
                                  <input
                                    type={showDeepResearchApiKey ? 'text' : 'password'}
                                    value={resolvedDeepResearch.deepResearchApiKey}
                                    onChange={(e) => { setDeepResearchApiKey(e.target.value); setSaved(false) }}
                                    className="w-full px-3 py-1.5 pr-10 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => setShowDeepResearchApiKey(!showDeepResearchApiKey)}
                                    className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600"
                                  >
                                    {showDeepResearchApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                                  </button>
                                </div>
                                <p className="text-xs text-slate-400">{t('literatureDeepResearchServerHint')}</p>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </label>
                </div>

                {/* PubMed Search Only */}
                <div className={`rounded-lg transition-all ${
                  literatureMode === 'pubmed_llm' ? 'bg-white border border-blue-200 shadow-sm' : 'hover:bg-white/50'
                }`}>
                  <label className="flex items-start gap-3 p-3 cursor-pointer">
                    <input
                      type="radio"
                      name="literatureMode"
                      value="pubmed_llm"
                      checked={literatureMode === 'pubmed_llm'}
                      onChange={() => { setLiteratureMode('pubmed_llm'); setSaved(false) }}
                      className="mt-0.5 accent-blue-600"
                    />
                    <div className="flex-1">
                      <span className="text-sm font-medium text-slate-800">{t('literaturePubmedLlm')}</span>
                      <p className="text-xs text-slate-500 mt-0.5">{t('literaturePubmedDesc')}</p>
                    </div>
                  </label>
                </div>
              </div>
            </div>

            {/* PubMed Search Settings - includes LLM config, greyed out when disabled */}
            <div className={`mb-4 p-4 rounded-xl border transition-all ${
              literatureMode === 'disabled'
                ? 'bg-slate-100 border-slate-200 opacity-50 pointer-events-none'
                : 'bg-slate-50 border-slate-200'
            }`}>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-slate-700">
                  {t('literaturePubmedSettings')}
                </label>
                <span className="text-xs text-slate-400">{t('literaturePubmedSettingsNote')}</span>
              </div>
              <div className="space-y-3">
                {/* Max Search Rounds */}
                <div className="space-y-1.5">
                  <label className="block text-xs font-medium text-slate-600">
                    {t('literatureMaxSearchRounds')}
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={literatureMaxSearchRounds}
                    disabled={literatureMode === 'disabled'}
                    onChange={(e) => {
                      setLiteratureMaxSearchRounds(parseInt(e.target.value) || 5)
                      setSaved(false)
                    }}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-slate-100 disabled:text-slate-400"
                  />
                  <p className="text-xs text-slate-400">{t('literatureMaxSearchRoundsNote')}</p>
                </div>

                {/* Model */}
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-slate-500" />
                    <label className="block text-xs font-medium text-slate-600">
                      {language === 'zh' ? '模型' : 'Model'}
                    </label>
                  </div>
                  <input
                    type="text"
                    value={editedModel}
                    disabled={literatureMode === 'disabled'}
                    onChange={(e) => {
                      setEditedModel(e.target.value)
                      setSaved(false)
                    }}
                    className="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-slate-100 disabled:text-slate-400"
                    placeholder="e.g., google/gemini-3.1-pro-preview"
                  />
                  <p className="text-xs text-slate-400">
                    {language === 'zh'
                      ? '用于 PubMed 模式文献分析的 LLM 模型，推荐: google/gemini-3.1-flash-lite-preview、google/gemini-3-flash-preview、openai/gpt-5.4-nano、openai/gpt-5.4-mini'
                      : 'LLM model for PubMed mode literature analysis. Recommended: google/gemini-3.1-flash-lite-preview, google/gemini-3-flash-preview, openai/gpt-5.4-nano, openai/gpt-5.4-mini'}
                  </p>
                </div>

                {/* Collapsible LLM Server / API Key */}
                <div className="border border-slate-200 rounded-lg">
                  <button
                    type="button"
                    onClick={() => { if (literatureMode !== 'disabled') setServerExpanded(!serverExpanded) }}
                    className="flex items-center gap-2 w-full text-left p-2"
                  >
                    {serverExpanded ? <ChevronDown className="w-3.5 h-3.5 text-slate-500" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />}
                    <Server className="w-3.5 h-3.5 text-slate-500" />
                    <span className="text-xs font-medium text-slate-700">
                      {t('agentLlmServer')} / {t('agentApiKey')}
                    </span>
                    {!serverExpanded && (
                      <span className="text-xs text-slate-500 ml-auto truncate max-w-[180px]">
                        {editedLlmBaseUrl || apiConfig.llmBaseUrl || 'https://openrouter.ai/api/v1'}
                      </span>
                    )}
                  </button>
                  {serverExpanded && literatureMode !== 'disabled' && (
                    <div className="px-2 pb-2 space-y-1.5">
                      <input
                        type="text"
                        value={editedLlmBaseUrl}
                        onChange={(e) => {
                          setEditedLlmBaseUrl(e.target.value)
                          setSaved(false)
                        }}
                        className="w-full px-3 py-1.5 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        placeholder="https://openrouter.ai/api/v1"
                      />
                      <div className="relative">
                        <input
                          type={showAgentApiKey ? 'text' : 'password'}
                          value={editedApiKey}
                          onChange={(e) => {
                            setEditedApiKey(e.target.value)
                            setSaved(false)
                          }}
                          className="w-full px-3 py-1.5 pr-10 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          placeholder="sk-..."
                        />
                        <button
                          type="button"
                          onClick={() => setShowAgentApiKey(!showAgentApiKey)}
                          className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600"
                        >
                          {showAgentApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                        </button>
                      </div>
                      <p className="text-xs text-slate-400">{t('agentServerHint')}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        )}

        {/* Model + Vision Selection - hidden for literature agent (shown in PubMed settings) */}
        {selectedAgent !== 'literature' && (
        <div className="mb-4 p-4 bg-slate-50 rounded-xl">
          <div className="flex items-center gap-3">
            <Cpu className="w-5 h-5 text-slate-500" />
            <label className="text-sm font-medium text-slate-700">
              {language === 'zh' ? '模型' : 'Model'}
            </label>
            <input
              type="text"
              value={editedModel}
              onChange={(e) => {
                setEditedModel(e.target.value)
                setSaved(false)
              }}
              className="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
              placeholder="e.g., google/gemini-3.1-pro-preview"
            />
          </div>

          {/* Vision Support Toggle - between model input and description */}
          {['pi', 'bioinformatics', 'biochemistry', 'painter', 'supervisor'].includes(selectedAgent) && (
            <div className="flex items-center gap-3 mt-2 ml-8">
              <Eye className="w-4 h-4 text-slate-500" />
              <label className="text-xs font-medium text-slate-600">
                {t('visionSupport')}
              </label>
              <button
                type="button"
                onClick={() => {
                  setEditedVisionEnabled(!editedVisionEnabled)
                  setSaved(false)
                }}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                  editedVisionEnabled ? 'bg-accent-600' : 'bg-slate-300'
                }`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                    editedVisionEnabled ? 'translate-x-[18px]' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
          )}

          <p className="text-xs text-slate-500 mt-2 ml-8">
            {language === 'zh' ? (
              selectedAgent === 'pi' ? '推荐支持 Vision（图片识别）的模型用于审核图片质量。推荐: google/gemini-3.1-pro-preview、openai/gpt-5.4' :
              selectedAgent === 'pathology' ? '推荐具有病理图片识别能力的模型，强烈推荐: google/gemini-3.1-pro-preview、openai/gpt-5.4' :
              selectedAgent === 'bioinformatics' ? '推荐 coding 能力较强且支持 Vision 的模型（用于图片修订）。推荐: anthropic/claude-sonnet-4.6、anthropic/claude-opus-4.6 或 google/gemini-3.1-pro-preview' :
              selectedAgent === 'biochemistry' ? '推荐: google/gemini-3.1-pro-preview、openai/gpt-5.4。若上传了生化结果图片，推荐选择支持 Vision 的模型' :
              selectedAgent === 'painter' ? '推荐具有图片生成能力的模型，推荐: google/gemini-3.1-flash-image-preview、google/gemini-3-pro-image-preview' :
              selectedAgent === 'supervisor' ? '推荐低成本、快速且支持 Vision 的模型进行实时监控（用于评估图片质量）。推荐: google/gemini-3.1-flash-lite-preview、openai/gpt-5.4-nano' :
              '使用 OpenRouter 模型格式'
            ) : (
              selectedAgent === 'pi' ? 'Vision-capable model recommended for figure quality review. Recommended: google/gemini-3.1-pro-preview, openai/gpt-5.4' :
              selectedAgent === 'pathology' ? 'Model with pathology image recognition recommended. Strongly recommended: google/gemini-3.1-pro-preview, openai/gpt-5.4' :
              selectedAgent === 'bioinformatics' ? 'Model with strong coding capability and Vision support recommended (for figure revision). Recommended: anthropic/claude-sonnet-4.6, anthropic/claude-opus-4.6, or google/gemini-3.1-pro-preview' :
              selectedAgent === 'biochemistry' ? 'Recommended: google/gemini-3.1-pro-preview, openai/gpt-5.4. If biochemistry images are uploaded, a Vision-capable model is recommended' :
              selectedAgent === 'painter' ? 'Model with image generation capability recommended. Recommended: google/gemini-3.1-flash-image-preview, google/gemini-3-pro-image-preview' :
              selectedAgent === 'supervisor' ? 'A cheap, fast, Vision-capable model recommended for real-time monitoring (evaluates figure quality). Recommended: google/gemini-3.1-flash-lite-preview, openai/gpt-5.4-nano' :
              'Use OpenRouter model format'
            )}
          </p>
          {['pathology', 'biochemistry', 'painter', 'supervisor'].includes(selectedAgent) && (
            <p className="text-xs text-amber-600 mt-1 ml-8">
              {t('agentDisableHint')}
            </p>
          )}
        </div>
        )}

        {/* Per-Agent LLM Server - collapsible, hidden for literature agent */}
        {selectedAgent !== 'literature' && (
        <div className="mb-4 p-4 bg-slate-50 rounded-xl">
          <button
            type="button"
            onClick={() => setServerExpanded(!serverExpanded)}
            className="flex items-center gap-2 w-full text-left"
          >
            {serverExpanded ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
            <Server className="w-4 h-4 text-slate-500" />
            <span className="text-sm font-medium text-slate-700">
              {t('agentLlmServer')} / {t('agentApiKey')}
            </span>
            {!serverExpanded && (
              <span className="text-xs text-slate-400 ml-auto truncate max-w-[200px]">
                {editedLlmBaseUrl || 'https://openrouter.ai/api/v1'}
              </span>
            )}
          </button>
          {serverExpanded && (
            <div className="mt-3 space-y-2 pl-6">
              <div className="flex items-center gap-3">
                <Server className="w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  value={editedLlmBaseUrl}
                  onChange={(e) => {
                    setEditedLlmBaseUrl(e.target.value)
                    setSaved(false)
                  }}
                  className="flex-1 px-3 py-1.5 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
                  placeholder="https://openrouter.ai/api/v1"
                />
              </div>
              <div className="flex items-center gap-3">
                <Key className="w-4 h-4 text-slate-400" />
                <div className="flex-1 relative">
                  <input
                    type={showAgentApiKey ? 'text' : 'password'}
                    value={editedApiKey}
                    onChange={(e) => {
                      setEditedApiKey(e.target.value)
                      setSaved(false)
                    }}
                    className="w-full px-3 py-1.5 pr-10 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
                    placeholder="sk-..."
                  />
                  <button
                    type="button"
                    onClick={() => setShowAgentApiKey(!showAgentApiKey)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600"
                  >
                    {showAgentApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>
              <p className="text-xs text-slate-400">
                {t('agentServerHint')}
              </p>
            </div>
          )}
        </div>
        )}

        {/* Temperature Control */}
        <div className="mb-4 p-4 bg-slate-50 rounded-xl">
          <div className="flex items-center gap-3">
            <Thermometer className="w-5 h-5 text-slate-500" />
            <label className="text-sm font-medium text-slate-700">
              {language === 'zh' ? '温度' : 'Temperature'}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={editedTemperature}
              onChange={(e) => {
                setEditedTemperature(parseFloat(e.target.value))
                setSaved(false)
              }}
              className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-accent-600"
            />
            <span className="w-12 text-center text-sm font-mono text-slate-600 bg-white px-2 py-1 rounded border border-slate-200">
              {editedTemperature.toFixed(1)}
            </span>
          </div>
          <div className="flex justify-between text-xs text-slate-400 mt-2 ml-8 mr-14">
            <span>{language === 'zh' ? '确定性 (客观)' : 'Deterministic'}</span>
            <span>{language === 'zh' ? '创造性 (多样)' : 'Creative'}</span>
          </div>
          <p className="text-xs text-slate-400 mt-1 ml-8">
            {language === 'zh'
              ? `默认值: ${DEFAULT_TEMPERATURES[selectedAgent]} | 低温度(0.1-0.3)更客观一致，高温度(0.5-0.7)更有创造性`
              : `Default: ${DEFAULT_TEMPERATURES[selectedAgent]} | Low (0.1-0.3): more consistent, High (0.5-0.7): more creative`}
          </p>
        </div>

        <div className="flex-1 relative">
          {selectedAgent === 'supervisor' ? (
            <div className="w-full h-full p-4 border border-slate-200 rounded-xl bg-slate-50 flex items-center justify-center">
              <p className="text-sm text-slate-400 text-center">
                {language === 'zh'
                  ? 'Supervisor Agent 使用内置监控逻辑，不支持自定义 Prompt。可通过上方调整模型和温度。'
                  : 'Supervisor Agent uses built-in monitoring logic and does not support custom prompts. You can adjust the model and temperature above.'}
              </p>
            </div>
          ) : (
            <textarea
              value={editedPrompt}
              onChange={(e) => {
                setEditedPrompt(e.target.value)
                setSaved(false)
              }}
              className="w-full h-full p-4 border border-slate-300 rounded-xl font-mono text-sm resize-none focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
              placeholder={language === 'zh' ? '输入 Agent 的系统提示词...' : 'Enter the agent system prompt...'}
            />
          )}
        </div>

        <div className="mt-4 p-4 bg-slate-50 rounded-xl">
          <h4 className="text-sm font-medium text-slate-700 mb-2">{t('promptTips')}</h4>
          <ul className="text-sm text-slate-500 space-y-1">
            <li>• {t('promptTip1')}</li>
            <li>• {t('promptTip2')}</li>
            <li>• {t('promptTip3')}</li>
            <li>• {t('promptTip4')}</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
