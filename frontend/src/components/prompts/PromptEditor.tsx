import { RotateCcw, Save, Thermometer } from 'lucide-react'

import { DEFAULT_TEMPERATURES } from '../../stores/appStore'
import { ProviderConfig } from './ProviderConfig'
import type { PromptsController } from './usePromptsController'

interface PromptEditorProps { controller: PromptsController }

export function PromptEditor({ controller }: PromptEditorProps) {
  const {
    Icon, agentConfig, colorStyles, editedPrompt, editedTemperature, handleReset, handleSave,
    handleSelectAgent, language, saveError, saved, saving, selectedAgent, selectedConfig,
    selectedStyles, setEditedPrompt, setEditedTemperature, setSaved, t,
  } = controller

  return (
    <div className="flex gap-6 h-full">
      {/* Agent List */}
      <div className="w-64 shrink-0">
        <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
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
                    : 'hover:bg-slate-50 dark:hover:bg-slate-800 border-2 border-transparent'
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
                    isSelected ? 'text-slate-900 dark:text-slate-100' : 'text-slate-700 dark:text-slate-200'
                  }`}>
                    {agent.name}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 line-clamp-2">
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
              <h3 className="font-display font-semibold text-slate-900 dark:text-slate-100">{selectedConfig.name}</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">{selectedConfig.description}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {saved && (
              <span className="text-sm text-green-700 dark:text-green-300">{t('saved')}</span>
            )}
            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-3 py-2 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg text-sm"
            >
              <RotateCcw className="w-4 h-4" />
              {t('reset')}
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all bg-accent-600 text-white hover:bg-accent-700"
            >
              <Save className="w-4 h-4" />
              {saving ? t('saving') : t('save')}
            </button>
          </div>
          {saveError && (
            <p className="text-red-600 dark:text-red-400 text-xs mt-1">{saveError}</p>
          )}
        </div>

        <ProviderConfig controller={controller} />

        {/* Temperature Control */}
        <div className="mb-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-xl">
          <div className="flex items-center gap-3">
            <Thermometer className="w-5 h-5 text-slate-500 dark:text-slate-400" />
            <label className="text-sm font-medium text-slate-700 dark:text-slate-200">
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
              className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-accent-600"
            />
            <span className="w-12 text-center text-sm font-mono text-slate-600 dark:text-slate-300 bg-white dark:bg-surface-900 px-2 py-1 rounded border border-slate-200 dark:border-slate-700">
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
            <div className="w-full h-full p-4 border border-slate-200 dark:border-slate-700 rounded-xl bg-slate-50 dark:bg-slate-800 flex items-center justify-center">
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
              className="w-full h-full p-4 border border-slate-300 dark:border-slate-600 rounded-xl font-mono text-sm resize-none focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
              placeholder={language === 'zh' ? '输入 Agent 的系统提示词...' : 'Enter the agent system prompt...'}
            />
          )}
        </div>

        <div className="mt-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-xl">
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">{t('promptTips')}</h4>
          <ul className="text-sm text-slate-500 dark:text-slate-400 space-y-1">
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
