import { useState } from 'react'
import { useConfig } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import { Key, X, Check, AlertCircle, Loader2, Server } from 'lucide-react'

interface ConfigPanelProps {
  onClose: () => void
  onSave: () => void
  allowSkip?: boolean
  onSkip?: () => void
}

export function ConfigPanel({ onClose, onSave, allowSkip = false, onSkip }: ConfigPanelProps) {
  const { apiConfig, setApiConfig, saveConfig, loading, error, clearError } = useConfig()
  const { language } = useAppStore()
  const { t } = useTranslation(language)
  const [localError, setLocalError] = useState<string | null>(null)

  // Clear errors when user modifies config
  const handleConfigChange = (updates: Partial<typeof apiConfig>) => {
    setApiConfig(updates)
    setLocalError(null)
    clearError()
  }

  const handleSave = async () => {
    setLocalError(null)
    clearError()

    if (!apiConfig.apiKey) {
      setLocalError(language === 'zh' ? '请输入 API Key' : 'Please enter API Key')
      return
    }

    try {
      await saveConfig()
      onSave()
    } catch (e) {
      // Error is already set in the hook
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden animate-slide-in">
        {/* Header */}
        <div className="bg-gradient-to-r from-violet-600 to-indigo-600 px-6 py-5 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-white/20 rounded-lg">
                <Key className="w-5 h-5" />
              </div>
              <div>
                <h2 className="font-semibold text-lg">{t('apiConfig')}</h2>
                <p className="text-violet-200 text-sm">{t('apiConfigDesc')}</p>
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
            <label className="block text-sm font-medium text-slate-700 mb-2">
              <Server className="w-4 h-4 inline mr-1" />
              {language === 'zh' ? 'LLM 服务器' : 'LLM Server'}
            </label>
            <input
              type="text"
              value={apiConfig.llmBaseUrl}
              onChange={(e) => handleConfigChange({ llmBaseUrl: e.target.value })}
              placeholder="https://openrouter.ai/api/v1"
              className="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-all font-mono text-sm"
            />
            <p className="text-xs text-slate-500 mt-1.5">
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
            <label className="block text-sm font-medium text-slate-700 mb-2">
              API Key <span className="text-red-500">*</span>
            </label>
            <input
              type="password"
              value={apiConfig.apiKey}
              onChange={(e) => handleConfigChange({ apiKey: e.target.value })}
              placeholder="sk-or-v1-..."
              className="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-all"
            />
            <p className="text-xs text-slate-500 mt-1.5">
              {language === 'zh' ? '用于 LLM 服务的 API 密钥' : 'API key for LLM service'} · 
              <a href="https://openrouter.ai/keys" target="_blank" className="text-violet-600 hover:underline">
                {t('getKey')}
              </a>
            </p>
          </div>

          {/* Google */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              {t('googleKey')}
              <span className="text-slate-400 font-normal ml-2">
                ({language === 'zh' ? '可选' : 'Optional'})
              </span>
            </label>
            <input
              type="password"
              value={apiConfig.googleApiKey}
              onChange={(e) => handleConfigChange({ googleApiKey: e.target.value })}
              placeholder="AIza..."
              className="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-all"
            />
            <p className="text-xs text-slate-500 mt-1.5">
              {t('googleDesc')} · 
              <a href="https://aistudio.google.com/app/apikey" target="_blank" className="text-violet-600 hover:underline">
                {t('getKey')}
              </a>
            </p>
          </div>

          {/* Proxy Server */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              {language === 'zh' ? '代理服务器' : 'Proxy Server'}
              <span className="text-slate-400 font-normal ml-2">
                ({language === 'zh' ? '可选，供内网用户使用' : 'Optional, for intranet users'})
              </span>
            </label>
            <input
              type="text"
              value={apiConfig.proxyUrl || ''}
              onChange={(e) => handleConfigChange({ proxyUrl: e.target.value })}
              placeholder="http://proxy.example.com:8080"
              className="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-violet-500 focus:border-violet-500 transition-all"
            />
            <p className="text-xs text-slate-500 mt-1.5">
              {language === 'zh' 
                ? '设置HTTP/HTTPS代理，用于API请求' 
                : 'Set HTTP/HTTPS proxy for API requests'}
            </p>
          </div>

          {/* Error */}
          {(error || localError) && (
            <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {error || localError}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex justify-between">
          <div>
            {allowSkip && (
              <button
                onClick={() => {
                  onSkip?.()
                  onClose()
                }}
                className="px-5 py-2.5 text-slate-500 hover:text-slate-700 hover:bg-slate-200 rounded-xl font-medium transition-colors"
              >
                {language === 'zh' ? '稍后配置' : 'Skip for now'}
              </button>
            )}
          </div>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-5 py-2.5 text-slate-700 hover:bg-slate-200 rounded-xl font-medium transition-colors"
            >
              {t('cancel')}
            </button>
            <button
              onClick={handleSave}
              disabled={loading}
              className="px-5 py-2.5 bg-violet-600 hover:bg-violet-700 text-white rounded-xl font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
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
