import { useState, useEffect } from 'react'
import { usePrompts } from '../hooks/useApi'
import { useAppStore, AgentType, DEFAULT_MODELS, DEFAULT_TEMPERATURES } from '../stores/appStore'
import { useTranslation, getDefaultPrompts } from '../i18n/translations'
import { 
  Users, 
  Microscope, 
  Code2, 
  FlaskConical, 
  BookOpen,
  Save,
  RotateCcw,
  Cpu,
  Thermometer
} from 'lucide-react'

export function PromptsPanel() {
  const { currentSession, language, prompts, setPrompt, models, setModel, temperatures, setTemperature } = useAppStore()
  const { savePrompt, saveModel, saveTemperature } = usePrompts()
  const { t } = useTranslation(language)
  const [selectedAgent, setSelectedAgent] = useState<AgentType>('pi')
  const [editedPrompt, setEditedPrompt] = useState(prompts[selectedAgent])
  const [editedModel, setEditedModel] = useState(models[selectedAgent])
  const [editedTemperature, setEditedTemperature] = useState(temperatures[selectedAgent])
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  // Update edited prompt, model, and temperature when selected agent changes
  useEffect(() => {
    setEditedPrompt(prompts[selectedAgent])
    setEditedModel(models[selectedAgent])
    setEditedTemperature(temperatures[selectedAgent])
  }, [selectedAgent, prompts, models, temperatures])

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
  ]

  const handleSelectAgent = (agentId: AgentType) => {
    setSelectedAgent(agentId)
    setEditedPrompt(prompts[agentId])
    setEditedModel(models[agentId])
    setEditedTemperature(temperatures[agentId])
    setSaved(false)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      if (currentSession) {
        await savePrompt(selectedAgent, editedPrompt)
        await saveModel(selectedAgent, editedModel)
        await saveTemperature(selectedAgent, editedTemperature)
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
    setSaved(false)
  }

  const selectedConfig = agentConfig.find(a => a.id === selectedAgent)!
  const Icon = selectedConfig.icon

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
            
            return (
              <button
                key={agent.id}
                onClick={() => handleSelectAgent(agent.id)}
                className={`w-full flex items-start gap-3 p-3 rounded-xl text-left transition-all ${
                  isSelected
                    ? `bg-${agent.color}-50 border-2 border-${agent.color}-200`
                    : 'hover:bg-slate-50 border-2 border-transparent'
                }`}
              >
                <div className={`p-2 rounded-lg ${
                  isSelected ? `bg-${agent.color}-100` : 'bg-slate-100'
                }`}>
                  <AgentIcon className={`w-4 h-4 ${
                    isSelected ? `text-${agent.color}-600` : 'text-slate-500'
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
            <div className={`p-2 rounded-lg bg-${selectedConfig.color}-100`}>
              <Icon className={`w-5 h-5 text-${selectedConfig.color}-600`} />
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
              placeholder="e.g., google/gemini-2.5-pro-preview-05-06"
            />
          </div>
          <p className="text-xs text-slate-400 mt-2 ml-8">
            {language === 'zh' 
              ? '使用 OpenRouter 模型格式，如 google/gemini-2.5-pro-preview-05-06、anthropic/claude-sonnet-4 等'
              : 'Use OpenRouter model format, e.g., google/gemini-2.5-pro-preview-05-06, anthropic/claude-sonnet-4'}
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
