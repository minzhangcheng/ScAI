import { useState, useEffect } from 'react'
import { usePrompts } from '../hooks/useApi'
import { useAppStore, AgentType, DEFAULT_MODELS, DEFAULT_TEMPERATURES } from '../stores/appStore'
import { useTranslation, getDefaultPrompts } from '../i18n/translations'
import { apiFetch } from '../hooks/useApi'
import {
  Users,
  Microscope,
  Code2,
  FlaskConical,
  BookOpen,
  Palette,
  Save,
  RotateCcw,
  Cpu,
  Thermometer,
  Search,
} from 'lucide-react'

export function PromptsPanel() {
  const {
    currentSession, language,
    prompts, setPrompt,
    models, setModel,
    temperatures, setTemperature,
    literatureMode, setLiteratureMode,
    googleApiKey,
    pubmedEmail,
    literatureMaxSearchRounds, setLiteratureMaxSearchRounds,
  } = useAppStore()
  const { savePrompt, saveModel, saveTemperature } = usePrompts()
  const { t } = useTranslation(language)
  const defaultPrompts = getDefaultPrompts(language)
  const [selectedAgent, setSelectedAgent] = useState<AgentType>('pi')
  const [editedPrompt, setEditedPrompt] = useState(prompts[selectedAgent] ?? defaultPrompts[selectedAgent] ?? '')
  const [editedModel, setEditedModel] = useState(models[selectedAgent] ?? DEFAULT_MODELS[selectedAgent] ?? '')
  const [editedTemperature, setEditedTemperature] = useState(temperatures[selectedAgent] ?? DEFAULT_TEMPERATURES[selectedAgent] ?? 0.5)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  // Auto-set literature mode based on Google API key presence
  useEffect(() => {
    if (googleApiKey) {
      setLiteratureMode('gemini_deep_research')
    } else {
      setLiteratureMode('pubmed_llm')
    }
  }, [googleApiKey, setLiteratureMode])

  // Update edited prompt, model, and temperature when selected agent changes
  useEffect(() => {
    setEditedPrompt(prompts[selectedAgent] ?? defaultPrompts[selectedAgent] ?? '')
    setEditedModel(models[selectedAgent] ?? DEFAULT_MODELS[selectedAgent] ?? '')
    setEditedTemperature(temperatures[selectedAgent] ?? DEFAULT_TEMPERATURES[selectedAgent] ?? 0.5)
  }, [selectedAgent, prompts, models, temperatures, defaultPrompts])

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
    }
  }

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
  ]

  const handleSelectAgent = (agentId: AgentType) => {
    setSelectedAgent(agentId)
    setEditedPrompt(prompts[agentId] ?? defaultPrompts[agentId] ?? '')
    setEditedModel(models[agentId] ?? DEFAULT_MODELS[agentId] ?? '')
    setEditedTemperature(temperatures[agentId] ?? DEFAULT_TEMPERATURES[agentId] ?? 0.5)
    setSaved(false)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      if (currentSession) {
        await savePrompt(selectedAgent, editedPrompt)
        await saveModel(selectedAgent, editedModel)
        await saveTemperature(selectedAgent, editedTemperature)

        // Also save literature config if on literature agent
        if (selectedAgent === 'literature') {
          try {
            await apiFetch(`/session/${currentSession.id}/literature-config`, {
              method: 'POST',
              body: JSON.stringify({
                mode: literatureMode,
                google_api_key: googleApiKey,
                pubmed_email: pubmedEmail,
                max_search_rounds: literatureMaxSearchRounds,
              }),
            })
          } catch (e) {
            console.error('Failed to save literature config:', e)
          }
        }
      } else {
        setPrompt(selectedAgent, editedPrompt)
        setModel(selectedAgent, editedModel)
        setTemperature(selectedAgent, editedTemperature)
      }
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      console.error('Failed to save prompt:', e)
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

    // Reset literature config if on literature agent
    if (selectedAgent === 'literature') {
      setLiteratureMode(googleApiKey ? 'gemini_deep_research' : 'pubmed_llm')
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
              <h3 className="font-semibold text-slate-900">{selectedConfig.name}</h3>
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
                  : 'bg-violet-600 text-white hover:bg-violet-700'
              }`}
            >
              <Save className="w-4 h-4" />
              {saved ? t('saved') : saving ? t('saving') : t('save')}
            </button>
          </div>
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

            {/* PubMed Settings - search rounds */}
            <div className="mb-4 p-4 bg-slate-50 rounded-xl border border-slate-200">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-slate-700">
                  {t('literaturePubmedSettings')}
                </label>
                <span className="text-xs text-slate-400">{t('literaturePubmedSettingsNote')}</span>
              </div>
              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-slate-600">
                  {t('literatureMaxSearchRounds')}
                </label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={literatureMaxSearchRounds}
                  onChange={(e) => {
                    setLiteratureMaxSearchRounds(parseInt(e.target.value) || 5)
                    setSaved(false)
                  }}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="text-xs text-slate-400">{t('literatureMaxSearchRoundsNote')}</p>
              </div>
            </div>
          </>
        )}

        {/* Model Selection */}
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
              className="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
              placeholder="e.g., google/gemini-3-pro-preview"
            />
          </div>
          <p className="text-xs text-slate-500 mt-2 ml-8">
            {language === 'zh' ? (
              selectedAgent === 'pi' ? '推荐: google/gemini-3-pro-preview' :
              selectedAgent === 'pathology' ? '需要具有病理图片识别能力的模型，强烈推荐: google/gemini-3-pro-preview' :
              selectedAgent === 'bioinformatics' ? '需要 coding 能力较强的模型，推荐: anthropic/claude-opus-4.6、openai/gpt-5.2-codex 或 google/gemini-3-pro-preview' :
              selectedAgent === 'biochemistry' ? '推荐: google/gemini-3-pro-preview。若上传了生化结果图片，需要选择支持 Vision 的模型' :
              selectedAgent === 'literature' ? '用于文献分析与综合的 LLM 模型，推荐: google/gemini-3-pro-preview' :
              selectedAgent === 'painter' ? '需要具有图片生成能力的模型，推荐: google/gemini-3-pro-image-preview' :
              '使用 OpenRouter 模型格式'
            ) : (
              selectedAgent === 'pi' ? 'Recommended: google/gemini-3-pro-preview' :
              selectedAgent === 'pathology' ? 'Requires pathology image recognition capability. Strongly recommended: google/gemini-3-pro-preview' :
              selectedAgent === 'bioinformatics' ? 'Requires strong coding capability. Recommended: anthropic/claude-opus-4.6, openai/gpt-5.2-codex, or google/gemini-3-pro-preview' :
              selectedAgent === 'biochemistry' ? 'Recommended: google/gemini-3-pro-preview. If biochemistry images are uploaded, a Vision-capable model is required' :
              selectedAgent === 'literature' ? 'LLM model for literature analysis and synthesis. Recommended: google/gemini-3-pro-preview' :
              selectedAgent === 'painter' ? 'Requires image generation capability. Recommended: google/gemini-3-pro-image-preview' :
              'Use OpenRouter model format'
            )}
          </p>
        </div>

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
              className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-violet-600"
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
          <textarea
            value={editedPrompt}
            onChange={(e) => {
              setEditedPrompt(e.target.value)
              setSaved(false)
            }}
            className="w-full h-full p-4 border border-slate-300 rounded-xl font-mono text-sm resize-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
            placeholder={language === 'zh' ? '输入 Agent 的系统提示词...' : 'Enter the agent system prompt...'}
          />
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
