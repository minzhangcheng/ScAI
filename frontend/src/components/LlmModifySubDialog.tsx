import { AlertCircle, Play, Settings, Clock } from 'lucide-react'
import { useTranslation } from '../i18n/translations'

interface LlmModifySubDialogProps {
  onResume: () => void
  onReturnToSettings: () => void
  onReturnToTimeout: () => void
  language: 'zh' | 'en'
}

export default function LlmModifySubDialog({
  onResume,
  onReturnToSettings,
  onReturnToTimeout,
  language,
}: LlmModifySubDialogProps) {
  const { t } = useTranslation(language)

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={t('configModified')}
    >
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-2xl w-full max-w-sm mx-4">
        {/* Header */}
        <div className="flex items-center gap-3 px-6 py-4 border-b border-gray-100 dark:border-gray-700">
          <div className="w-10 h-10 rounded-full bg-amber-100 dark:bg-amber-900/60 flex items-center justify-center">
            <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400" aria-hidden="true" />
          </div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {t('configModified')}
          </h2>
        </div>

        {/* Info */}
        <div className="px-6 py-4">
          <p className="text-sm text-gray-600 dark:text-gray-300">
            {t('configModifiedDesc')}
          </p>
        </div>

        {/* Buttons */}
        <div className="px-6 pb-4 space-y-2">
          <button
            onClick={onResume}
            className="w-full flex items-center gap-3 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
          >
            <Play className="w-4 h-4" aria-hidden="true" />
            {t('resumeAnalysis')}
          </button>

          <button
            onClick={onReturnToSettings}
            className="w-full flex items-center gap-3 px-4 py-3 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-xl font-medium transition-colors"
          >
            <Settings className="w-4 h-4" aria-hidden="true" />
            {t('returnToSettings')}
          </button>

          <button
            onClick={onReturnToTimeout}
            className="w-full flex items-center gap-3 px-4 py-3 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-xl font-medium transition-colors"
          >
            <Clock className="w-4 h-4" aria-hidden="true" />
            {t('backToTimeoutDialog')}
          </button>
        </div>
      </div>
    </div>
  )
}
