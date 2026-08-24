import { AlertCircle } from 'lucide-react'
import { useTranslation } from '../i18n/translations'

interface ValidationDialogProps {
  message: string
  uncertain: boolean
  onIgnore: () => void
  onReconfigure: () => void
  language: 'zh' | 'en'
}

export function ValidationDialog({ message, uncertain, onIgnore, onReconfigure, language }: ValidationDialogProps) {
  const { t } = useTranslation(language)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        <div className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-red-100 dark:bg-red-900/60 rounded-full">
              <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              {uncertain ? t('apiKeyValidationUncertain') : t('apiKeyInvalid')}
            </h3>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-300 mb-2">
            {message}
          </p>
          <p className="text-xs text-slate-400 dark:text-slate-500">
            {uncertain
              ? t('apiKeyValidationUncertainDesc')
              : t('apiKeyInvalidDesc')}
          </p>
        </div>
        <div className="px-6 py-4 bg-slate-50 dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 flex justify-end gap-3">
          <button
            onClick={onIgnore}
            className="px-4 py-2 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg font-medium transition-colors text-sm"
          >
            {t('ignoreError')}
          </button>
          <button
            onClick={onReconfigure}
            className="px-4 py-2 bg-accent-600 hover:bg-accent-700 text-white rounded-lg font-medium transition-colors text-sm"
          >
            {t('reconfigureApiKey')}
          </button>
        </div>
      </div>
    </div>
  )
}
