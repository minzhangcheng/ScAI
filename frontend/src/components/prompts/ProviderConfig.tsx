import {
  AlertCircle, ChevronDown, ChevronRight, Cpu, Eye, EyeOff, Key, Search, Server,
  Sliders,
} from 'lucide-react'

import { DEFAULT_MODELS } from '../../stores/appStore'
import type { PromptsController } from './usePromptsController'

interface ProviderProps { controller: PromptsController }

function ContextWindowInput({ controller }: ProviderProps) {
  const {
    editedContextWindow, selectedAgent, setContextWindowTouched,
    setEditedContextWindow, setSaved, t,
  } = controller
  return (
    <div className="flex items-center gap-3">
      <Sliders className="w-4 h-4 text-slate-400" />
      <input
        aria-label={t('agentContextWindow')}
        type="number"
        min={8192}
        step={1024}
        value={editedContextWindow ?? ''}
        onChange={(event) => {
          const raw = event.target.value.trim()
          setEditedContextWindow(raw ? Number.parseInt(raw, 10) : undefined)
          setContextWindowTouched(Boolean(raw))
          setSaved(false)
        }}
        className="flex-1 px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
        placeholder={String(
          controller.agentServerConfigs[selectedAgent]?.effectiveContextWindow
          ?? 200000,
        )}
        title={t('agentContextWindowHint')}
      />
      <button
        type="button"
        onClick={() => {
          setEditedContextWindow(undefined)
          setContextWindowTouched(true)
          setSaved(false)
        }}
        className="px-2 py-1.5 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 rounded"
      >
        {t('reset')}
      </button>
    </div>
  )
}

function LiteratureResearchMode({ controller }: ProviderProps) {
  const {
    deepResearchModel, googleApiKey, language, literatureMode, resolvedDeepResearch,
    serverExpanded, setDeepResearchApiKey,
    setDeepResearchLlmBaseUrl, setDeepResearchModel, setGoogleApiKey, setLiteratureMode,
    setSaved, setServerExpanded, setShowDeepResearchApiKey, setShowGoogleKey,
    showDeepResearchApiKey, showGoogleKey, t,
  } = controller
  return (
    <>
            {/* Research Mode Selector */}
            <div className="mb-4 p-4 bg-blue-50/50 dark:bg-blue-950/40 rounded-xl border border-blue-100 dark:border-blue-900">
              <div className="flex items-center gap-2 mb-3">
                <Search className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <label className="text-sm font-medium text-slate-700 dark:text-slate-200">
                  {t('literatureResearchMode')}
                </label>
              </div>

              <div className="space-y-2">
                {/* Disabled */}
                <div className={`rounded-lg transition-all ${
                  literatureMode === 'disabled' ? 'bg-white dark:bg-surface-900 border border-blue-200 dark:border-blue-800 shadow-sm' : 'hover:bg-white/50 dark:hover:bg-surface-900/50'
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
                      <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{language === 'zh' ? '不启用' : 'Disabled'}</span>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{language === 'zh' ? '不使用文献 Agent 进行文献检索' : 'Do not use Literature Agent for literature search'}</p>
                    </div>
                  </label>
                </div>

                {/* Gemini Deep Research */}
                <div className={`rounded-lg transition-all ${
                  literatureMode === 'gemini_deep_research' ? 'bg-white dark:bg-surface-900 border border-blue-200 dark:border-blue-800 shadow-sm' : 'hover:bg-white/50 dark:hover:bg-surface-900/50'
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
                      <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{t('literatureGeminiDeepResearch')}</span>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{t('literatureGeminiDesc')}</p>
                      {!googleApiKey && (
                        <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                          {t('literatureGeminiNote')}
                        </p>
                      )}
                      <div className="flex items-start gap-2 mt-2 p-2 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded-lg">
                        <AlertCircle className="w-3.5 h-3.5 text-amber-500 flex-shrink-0 mt-0.5" />
                        <p className="text-xs text-amber-700 dark:text-amber-300">
                          {language === 'zh'
                            ? 'Deep Research 每次检索耗时较长（数分钟至最多60分钟），费用约 $2-5/次。'
                            : 'Deep Research takes minutes to complete (up to 60 min) and costs ~$2-5 per task.'}{' '}
                          <a href="https://ai.google.dev/gemini-api/docs/deep-research" target="_blank" rel="noopener noreferrer" className="text-amber-800 dark:text-amber-200 underline hover:text-amber-900 dark:hover:text-amber-200">
                            {language === 'zh' ? '参考 Google 文档' : 'See Google docs'}
                          </a>
                        </p>
                      </div>
                      {literatureMode === 'gemini_deep_research' && (
                        <div className="mt-2 space-y-1.5">
                          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">
                            {t('literatureGoogleApiKey')}
                          </label>
                          <div className="relative">
                            <input
                              type={showGoogleKey ? 'text' : 'password'}
                              value={googleApiKey}
                              onChange={(e) => { setGoogleApiKey(e.target.value); setSaved(false) }}
                              placeholder="AIza..."
                              className="w-full px-3 py-1.5 pr-10 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                            <button
                              type="button"
                              onClick={() => setShowGoogleKey(!showGoogleKey)}
                              className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                            >
                              {showGoogleKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                            </button>
                          </div>
                          <p className="text-xs text-slate-400">
                            <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
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
                  literatureMode === 'deep_research' ? 'bg-white dark:bg-surface-900 border border-blue-200 dark:border-blue-800 shadow-sm' : 'hover:bg-white/50 dark:hover:bg-surface-900/50'
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
                      <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{t('literatureDeepResearch')}</span>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{t('literatureDeepResearchDesc')}</p>
                      {literatureMode === 'deep_research' && (
                        <div className="mt-2 space-y-2.5">
                          <div className="space-y-1.5">
                            <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">
                              {t('literatureDeepResearchModel')}
                            </label>
                            <input
                              type="text"
                              value={deepResearchModel}
                              placeholder={t('literatureDeepResearchModelPlaceholder')}
                              onChange={(e) => { setDeepResearchModel(e.target.value); setSaved(false) }}
                              className="w-full px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                            <p className="text-xs text-slate-400">{t('literatureDeepResearchNote')}</p>
                          </div>
                          {/* Collapsible server/key for deep research */}
                          <div className="border border-slate-200 dark:border-slate-700 rounded-lg">
                            <button
                              type="button"
                              onClick={() => setServerExpanded(!serverExpanded)}
                              className="flex items-center gap-2 w-full text-left p-2"
                            >
                              {serverExpanded ? <ChevronDown className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />}
                              <Server className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
                              <span className="text-xs font-medium text-slate-700 dark:text-slate-200">
                                {t('literatureDeepResearchServer')} / {t('literatureDeepResearchApiKey')}
                              </span>
                              {!serverExpanded && (
                                <span className="text-xs text-slate-500 dark:text-slate-400 ml-auto truncate max-w-[200px]">
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
                                  className="w-full px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                />
                                <div className="relative">
                                  <input
                                    type={showDeepResearchApiKey ? 'text' : 'password'}
                                    value={resolvedDeepResearch.deepResearchApiKey}
                                    onChange={(e) => { setDeepResearchApiKey(e.target.value); setSaved(false) }}
                                    className="w-full px-3 py-1.5 pr-10 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => setShowDeepResearchApiKey(!showDeepResearchApiKey)}
                                    className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
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
                  literatureMode === 'pubmed_llm' ? 'bg-white dark:bg-surface-900 border border-blue-200 dark:border-blue-800 shadow-sm' : 'hover:bg-white/50 dark:hover:bg-surface-900/50'
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
                      <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{t('literaturePubmedLlm')}</span>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{t('literaturePubmedDesc')}</p>
                    </div>
                  </label>
                </div>
              </div>
            </div>
    </>
  )
}

function PubMedSettings({ controller }: ProviderProps) {
  const {
    apiConfig, editedApiKey, editedLlmBaseUrl, editedModel, language, literatureMaxSearchRounds,
    literatureMode, serverExpanded, setEditedApiKey, setEditedLlmBaseUrl, setEditedModel,
    setLiteratureMaxSearchRounds, setSaved, setServerExpanded, setShowAgentApiKey,
    showAgentApiKey, t,
  } = controller
  return (
    <>
            {/* PubMed Search Settings - includes LLM config, greyed out when disabled */}
            <div className={`mb-4 p-4 rounded-xl border transition-all ${
              literatureMode === 'disabled'
                ? 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 opacity-50 pointer-events-none'
                : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700'
            }`}>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-slate-700 dark:text-slate-200">
                  {t('literaturePubmedSettings')}
                </label>
                <span className="text-xs text-slate-400">{t('literaturePubmedSettingsNote')}</span>
              </div>
              <div className="space-y-3">
                {/* Max Search Rounds */}
                <div className="space-y-1.5">
                  <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">
                    {t('literatureMaxSearchRounds')}
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={literatureMaxSearchRounds}
                    disabled={literatureMode === 'disabled'}
                    onChange={(e) => {
                      setLiteratureMaxSearchRounds(parseInt(e.target.value) || 20)
                      setSaved(false)
                    }}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-slate-100 dark:disabled:bg-slate-800 disabled:text-slate-400"
                  />
                  <p className="text-xs text-slate-400">{t('literatureMaxSearchRoundsNote')}</p>
                </div>

                {/* Model */}
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                    <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">
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
                    className="w-full px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-slate-100 dark:disabled:bg-slate-800 disabled:text-slate-400"
                    placeholder="e.g., gpt-5.4-mini"
                  />
                  <p className="text-xs text-slate-400">
                    {language === 'zh'
                      ? '用于 PubMed + LLM Only 模式文献分析的模型。推荐: gemini-3.1-flash-lite、gpt-5.4-nano、gpt-5.4-mini、deepseek-v4-flash'
                      : 'Model for PubMed + LLM Only analysis. Recommended: gemini-3.1-flash-lite, gpt-5.4-nano, gpt-5.4-mini, deepseek-v4-flash'}
                  </p>
                </div>

                {/* Collapsible LLM Server / API Key */}
                <div className="border border-slate-200 dark:border-slate-700 rounded-lg">
                  <button
                    type="button"
                    onClick={() => { if (literatureMode !== 'disabled') setServerExpanded(!serverExpanded) }}
                    className="flex items-center gap-2 w-full text-left p-2"
                  >
                    {serverExpanded ? <ChevronDown className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />}
                    <Server className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
                    <span className="text-xs font-medium text-slate-700 dark:text-slate-200">
                      {t('agentLlmServer')} / {t('agentApiKey')}
                    </span>
                    {!serverExpanded && (
                      <span className="text-xs text-slate-500 dark:text-slate-400 ml-auto truncate max-w-[180px]">
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
                        className="w-full px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
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
                          className="w-full px-3 py-1.5 pr-10 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          placeholder="sk-..."
                        />
                        <button
                          type="button"
                          onClick={() => setShowAgentApiKey(!showAgentApiKey)}
                          className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                        >
                          {showAgentApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                        </button>
                      </div>
                      <ContextWindowInput controller={controller} />
                      <p className="text-xs text-slate-400">{t('agentServerHint')}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
    </>
  )
}

function StandardProvider({ controller }: ProviderProps) {
  const {
    editedApiKey, editedLlmBaseUrl, editedModel, editedVisionEnabled, language,
    selectedAgent, serverExpanded, setEditedApiKey, setEditedLlmBaseUrl, setEditedModel,
    setEditedVisionEnabled, setSaved, setServerExpanded, setShowAgentApiKey,
    showAgentApiKey, t,
  } = controller
  return (
    <>
        {/* Model + Vision Selection - hidden for literature agent (shown in PubMed settings) */}
        {selectedAgent !== 'literature' && (
        <div className="mb-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-xl">
          <div className="flex items-center gap-3">
            <Cpu className="w-5 h-5 text-slate-500 dark:text-slate-400" />
            <label className="text-sm font-medium text-slate-700 dark:text-slate-200">
              {language === 'zh' ? '模型' : 'Model'}
            </label>
            <input
              type="text"
              value={editedModel}
              onChange={(e) => {
                setEditedModel(e.target.value)
                setSaved(false)
              }}
              className="flex-1 px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
              placeholder={`e.g., ${DEFAULT_MODELS[selectedAgent]}`}
            />
          </div>

          {/* Vision Support Toggle - between model input and description */}
          {['pi', 'bioinformatics', 'biochemistry', 'painter', 'supervisor'].includes(selectedAgent) && (
            <div className="flex items-center gap-3 mt-2 ml-8">
              <Eye className="w-4 h-4 text-slate-500 dark:text-slate-400" />
              <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
                {t('visionSupport')}
              </label>
              <button
                type="button"
                onClick={() => {
                  setEditedVisionEnabled(!editedVisionEnabled)
                  setSaved(false)
                }}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                  editedVisionEnabled ? 'bg-accent-600' : 'bg-slate-300 dark:bg-slate-600'
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

          <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 ml-8">
            {language === 'zh' ? (
              selectedAgent === 'pi' ? '推荐支持 Vision（图片识别）的模型用于审核图片质量。推荐: gemini-3.1-pro-preview、gpt-5.6-sol、deepseek-v4-pro' :
              selectedAgent === 'pathology' ? '推荐具有病理图片识别能力的模型。推荐: gemini-3.1-pro-preview、gpt-5.6-sol' :
              selectedAgent === 'bioinformatics' ? '推荐 coding 能力较强且支持 Vision 的模型（用于图片修订）。推荐: claude-sonnet-4.6、claude-opus-4.6、gpt-5.6-sol、glm-5.2' :
              selectedAgent === 'biochemistry' ? '若上传了生化结果图片，推荐选择支持 Vision 的模型。推荐: gemini-3.1-pro-preview、gpt-5.6-sol' :
              selectedAgent === 'painter' ? '推荐具有图片生成能力的模型。推荐: gemini-3-pro-image、gemini-3.1-flash-lite-image、gpt-image-2' :
              selectedAgent === 'supervisor' ? '推荐低成本、快速且支持 Vision 的模型进行实时监控（用于评估图片质量）。推荐: gemini-3.1-flash-lite、gpt-5.4-nano、gpt-5.4-mini、mimo-v2.5' :
              selectedAgent === 'technique' ? '推荐低成本、快速的模型。推荐: gemini-3.1-flash-lite、gpt-5.4-nano、gpt-5.4-mini、deepseek-v4-flash' :
              '使用 OpenAI 兼容模型名称'
            ) : (
              selectedAgent === 'pi' ? 'Vision-capable model recommended for figure quality review. Recommended: gemini-3.1-pro-preview, gpt-5.6-sol, deepseek-v4-pro' :
              selectedAgent === 'pathology' ? 'Model with pathology image recognition recommended. Recommended: gemini-3.1-pro-preview, gpt-5.6-sol' :
              selectedAgent === 'bioinformatics' ? 'Model with strong coding capability and Vision support recommended (for figure revision). Recommended: claude-sonnet-4.6, claude-opus-4.6, gpt-5.6-sol, glm-5.2' :
              selectedAgent === 'biochemistry' ? 'If biochemistry images are uploaded, a Vision-capable model is recommended. Recommended: gemini-3.1-pro-preview, gpt-5.6-sol' :
              selectedAgent === 'painter' ? 'Image generation model recommended. Recommended: gemini-3-pro-image, gemini-3.1-flash-lite-image, gpt-image-2' :
              selectedAgent === 'supervisor' ? 'A cheap, fast, Vision-capable model is recommended for real-time monitoring. Recommended: gemini-3.1-flash-lite, gpt-5.4-nano, gpt-5.4-mini, mimo-v2.5' :
              selectedAgent === 'technique' ? 'A cheap, fast model is recommended. Recommended: gemini-3.1-flash-lite, gpt-5.4-nano, gpt-5.4-mini, deepseek-v4-flash' :
              'Use an OpenAI-compatible model name'
            )}
          </p>
          {['pathology', 'biochemistry', 'painter', 'supervisor'].includes(selectedAgent) && (
            <p className="text-xs text-amber-600 dark:text-amber-400 mt-1 ml-8">
              {t('agentDisableHint')}
            </p>
          )}
        </div>
        )}
        {/* Per-Agent LLM Server - collapsible, hidden for literature agent */}
        {selectedAgent !== 'literature' && (
        <div className="mb-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-xl">
          <button
            type="button"
            onClick={() => setServerExpanded(!serverExpanded)}
            className="flex items-center gap-2 w-full text-left"
          >
            {serverExpanded ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
            <Server className="w-4 h-4 text-slate-500 dark:text-slate-400" />
            <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
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
                  className="flex-1 px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
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
                    className="w-full px-3 py-1.5 pr-10 border border-slate-300 dark:border-slate-600 rounded-lg text-sm font-mono focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
                    placeholder="sk-..."
                  />
                  <button
                    type="button"
                    onClick={() => setShowAgentApiKey(!showAgentApiKey)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                  >
                    {showAgentApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>
              <ContextWindowInput controller={controller} />
              <p className="text-xs text-slate-400">
                {t('agentServerHint')}
              </p>
            </div>
          )}
        </div>
        )}
    </>
  )
}

export function ProviderConfig({ controller }: ProviderProps) {
  return controller.selectedAgent === 'literature' ? (
    <>
      <LiteratureResearchMode controller={controller} />
      <PubMedSettings controller={controller} />
    </>
  ) : <StandardProvider controller={controller} />
}
