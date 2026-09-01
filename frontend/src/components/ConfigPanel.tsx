import { useState, useEffect } from 'react'
import { useConfig, useUserConfig } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import { Key, X, Check, AlertCircle, Loader2, Server, BookOpen, Eye, EyeOff } from 'lucide-react'

interface ConfigPanelProps {
  onClose: () => void
  onSave: () => void
  onReadMode?: () => void
  isAdmin?: boolean
}

export function ConfigPanel({ onClose, onSave, onReadMode, isAdmin = true }: ConfigPanelProps) {
  const adminConfig = useConfig()
  const userConfig = useUserConfig()

  const { language, pubmedEmail, setPubmedEmail, setApiConfig } = useAppStore()
  const { t } = useTranslation(language)
  const [localError, setLocalError] = useState<string | null>(null)
  const [showApiKey, setShowApiKey] = useState(false)
  const [uncertainWarning, setUncertainWarning] = useState<string | null>(null)
  const [configLoaded, setConfigLoaded] = useState(false)
  const [configLoadError, setConfigLoadError] = useState<string | null>(null)
  const [draftApiKey, setDraftApiKey] = useState('')
  const [draftLlmBaseUrl, setDraftLlmBaseUrl] = useState('')
  const [draftProxyUrl, setDraftProxyUrl] = useState('')
  const [loadGeneration, setLoadGeneration] = useState(0)

  useEffect(() => {
    let cancelled = false
    setConfigLoaded(false)
    setConfigLoadError(null)
    const load = isAdmin ? adminConfig.getConfig : userConfig.getUserConfig
    load()
        .then((status) => {
          if (cancelled) return
          setDraftLlmBaseUrl(status.llm_base_url || 'https://openrouter.ai/api/v1')
          setDraftApiKey(status.masked_api_key || '')
          setDraftProxyUrl(status.has_proxy_url ? '•••••' : '')
          setConfigLoaded(true)
        })
        .catch((error) => {
          if (!cancelled) {
            setConfigLoadError(error instanceof Error ? error.message : 'Failed to load config')
          }
        })
    return () => { cancelled = true }
  }, [isAdmin, adminConfig.getConfig, userConfig.getUserConfig, loadGeneration])

  const loading = isAdmin ? adminConfig.loading : userConfig.loading
  const hookError = isAdmin ? adminConfig.error : userConfig.error

  const clearErrors = () => {
    setLocalError(null)
    setUncertainWarning(null)
  }

  const validateMsg = (zh: string, en: string) => language === 'zh' ? zh : en

  const handleSave = async () => {
    clearErrors()
    if (isAdmin) adminConfig.clearError()

    const apiKey = draftApiKey
    if (!apiKey) {
      setLocalError(validateMsg('请输入 API Key', 'Please enter API Key'))
      return
    }

    try {
      if (isAdmin) {
        await adminConfig.saveConfig({ apiKey: draftApiKey, llmBaseUrl: draftLlmBaseUrl, proxyUrl: draftProxyUrl })
      } else {
        await userConfig.saveUserConfig(draftApiKey, draftLlmBaseUrl, draftProxyUrl)
      }
      const isRealKey = draftApiKey && !draftApiKey.includes('******')
      setApiConfig({
        ...(isRealKey ? { apiKey: draftApiKey } : {}),
        llmBaseUrl: draftLlmBaseUrl || 'https://openrouter.ai/api/v1',
        ...(draftProxyUrl === '•••••' ? {} : { proxyUrl: draftProxyUrl }),
      })
      onSave()
    } catch {
      // Error already set in the hook
    }
  }

  type FieldDef = { value: string; onChange: (v: string) => void }

  const fields: Record<'llmBaseUrl' | 'apiKey' | 'proxyUrl', FieldDef> = {
    llmBaseUrl: {
      value: draftLlmBaseUrl,
      onChange: (v) => { setDraftLlmBaseUrl(v); clearErrors() },
    },
    apiKey: {
      value: draftApiKey,
      onChange: (v) => { setDraftApiKey(v); clearErrors() },
    },
    proxyUrl: {
      value: draftProxyUrl,
      onChange: (v) => { setDraftProxyUrl(v); clearErrors() },
    },
  }

  if (!configLoaded) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal w-full max-w-md mx-4 p-8 text-center">
          {configLoadError ? (
            <>
              <AlertCircle className="w-8 h-8 mx-auto mb-3 text-red-500" />
              <p className="text-sm text-red-600 dark:text-red-400">{configLoadError}</p>
              <div className="mt-4 flex justify-center gap-3">
                <button onClick={onClose} className="px-4 py-2 text-sm bg-surface-100 dark:bg-surface-800 rounded-lg">
                  {language === 'zh' ? '关闭' : 'Close'}
                </button>
                <button
                  onClick={() => setLoadGeneration(value => value + 1)}
                  className="px-4 py-2 text-sm text-white bg-accent-600 hover:bg-accent-700 rounded-lg"
                >
                  {language === 'zh' ? '重试' : 'Retry'}
                </button>
              </div>
            </>
          ) : (
            <>
              <Loader2 className="w-8 h-8 mx-auto mb-3 animate-spin text-accent-600" />
              <p className="text-sm text-surface-500 dark:text-surface-400">
                {language === 'zh' ? '正在读取已有配置...' : 'Loading saved configuration...'}
              </p>
            </>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal w-full max-w-lg mx-4 overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="bg-gradient-to-r from-accent-600 to-accent-700 px-6 py-5 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-white/20 rounded-lg">
                <Key className="w-5 h-5" />
              </div>
              <div>
                <h2 className="font-display font-semibold text-lg">{t('apiConfig')}</h2>
                <p className="text-accent-200 text-sm">{t('apiConfigDesc')}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5">
          {/* LLM Server URL */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
              <Server className="w-4 h-4 inline mr-1" />
              {language === 'zh' ? '默认 LLM 服务器' : 'Default LLM Server'} <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              inputMode="url"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              value={fields.llmBaseUrl.value}
              onChange={(e) => fields.llmBaseUrl.onChange(e.target.value)}
              placeholder="https://openrouter.ai/api/v1"
              className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 bg-white dark:bg-surface-950 text-slate-900 dark:text-slate-100 rounded-xl focus:ring-2 focus:ring-accent-500 focus:border-accent-500 transition-all font-mono text-sm"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
              {language === 'zh'
                ? '支持任何 OpenAI 兼容的 API 端点'
                : 'Supports any OpenAI-compatible API endpoint'}
            </p>
            <p className="text-xs text-blue-600 mt-1">
              💡 {language === 'zh'
                ? '建议使用 OpenRouter 以支持更多模型'
                : 'Recommend using OpenRouter for more model options'}
            </p>
          </div>

          {/* API Key */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
              API Key <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={fields.apiKey.value}
                onChange={(e) => fields.apiKey.onChange(e.target.value)}
                placeholder="sk-or-v1-..."
                className="w-full px-4 py-3 pr-12 border border-slate-300 dark:border-slate-600 bg-white dark:bg-surface-950 text-slate-900 dark:text-slate-100 rounded-xl focus:ring-2 focus:ring-accent-500 focus:border-accent-500 transition-all"
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                title={showApiKey ? (language === 'zh' ? '隐藏' : 'Hide') : (language === 'zh' ? '显示' : 'Show')}
              >
                {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
              {language === 'zh' ? '用于 LLM 服务的 API 密钥' : 'API key for LLM service'} ·
              <a href="https://openrouter.ai/keys" target="_blank" className="text-accent-600 hover:underline">
                {t('getKey')}
              </a>
            </p>
          </div>

          {/* PubMed Email */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
              {t('literaturePubmedEmailLabel')} <span className="text-red-500">*</span>
            </label>
            <input
              type="email"
              value={pubmedEmail}
              onChange={(e) => setPubmedEmail(e.target.value)}
              placeholder="your@email.com"
              className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 bg-white dark:bg-surface-950 text-slate-900 dark:text-slate-100 rounded-xl focus:ring-2 focus:ring-accent-500 focus:border-accent-500 transition-all text-sm"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
              {t('literaturePubmedEmailNote')}
            </p>
          </div>

          {/* Proxy Server */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
              {language === 'zh' ? '代理服务器' : 'Proxy Server'}
              <span className="text-slate-400 font-normal ml-2">
                ({language === 'zh' ? '可选，供内网用户使用' : 'Optional, for intranet users'})
              </span>
            </label>
            <input
              type="text"
              value={fields.proxyUrl.value}
              onChange={(e) => fields.proxyUrl.onChange(e.target.value)}
              placeholder="http://proxy.example.com:8080"
              className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 bg-white dark:bg-surface-950 text-slate-900 dark:text-slate-100 rounded-xl focus:ring-2 focus:ring-accent-500 focus:border-accent-500 transition-all"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
              {language === 'zh'
                ? '设置HTTP/HTTPS代理，用于API请求'
                : 'Set HTTP/HTTPS proxy for API requests'}
            </p>
          </div>

          {/* Uncertain warning (orange) */}
          {uncertainWarning && (
            <div className="flex items-center gap-2 p-3 bg-orange-50 border border-orange-200 rounded-xl text-orange-700 text-sm">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {uncertainWarning}
            </div>
          )}

          {/* Error */}
          {(hookError || localError) && (
            <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {hookError || localError}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-50 dark:bg-surface-950 border-t border-slate-200 dark:border-slate-700 flex justify-between">
          <div>
            {onReadMode && (
              <button
                onClick={() => {
                  onReadMode()
                }}
                className="flex items-center gap-2 px-5 py-2.5 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-xl font-medium transition-colors"
              >
                <BookOpen className="w-4 h-4" />
                {t('readMode')}
              </button>
            )}
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleSave}
              disabled={loading}
              className="px-5 py-2.5 bg-accent-600 hover:bg-accent-700 text-white rounded-xl font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Check className="w-4 h-4" />
              )}
              {t('saveConfig')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
