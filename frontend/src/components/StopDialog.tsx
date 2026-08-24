import { Square, CheckCircle2, Loader2 } from 'lucide-react'
import { useTranslation } from '../i18n/translations'

interface StopDialogProps {
  onStopImmediate: () => void
  onStopSummarize: () => void
  onClose: () => void
  stopping: boolean
  language: 'zh' | 'en'
}

export function StopDialog({ onStopImmediate, onStopSummarize, onClose, stopping, language }: StopDialogProps) {
  // t() is available for future i18n when translation keys are added
  const { t: _t } = useTranslation(language)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-2xl w-full max-w-md mx-4">
        <div className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-red-100 rounded-full">
              <Square className="w-5 h-5 text-red-600" />
            </div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              {language === 'zh' ? '停止分析' : 'Stop Analysis'}
            </h3>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-300 mb-4">
            {language === 'zh' ? '请选择停止方式：' : 'Choose how to stop:'}
          </p>
        </div>
        <div className="px-6 pb-6 space-y-2">
          <button
            onClick={onStopImmediate}
            disabled={stopping}
            className="w-full flex items-center gap-3 px-4 py-3 bg-red-50 hover:bg-red-100 border border-red-200 rounded-xl text-left transition-colors disabled:opacity-50"
          >
            <Square className="w-5 h-5 text-red-600 flex-shrink-0" />
            <div>
              <div className="font-medium text-red-800">
                {language === 'zh' ? '立即终止' : 'Terminate Immediately'}
              </div>
              <div className="text-xs text-red-600">
                {language === 'zh'
                  ? '完全终止分析，保存状态，稍后可恢复继续'
                  : 'Stop analysis, save state, resume later'}
              </div>
            </div>
          </button>
          <button
            onClick={onStopSummarize}
            disabled={stopping}
            className="w-full flex items-center gap-3 px-4 py-3 bg-amber-50 hover:bg-amber-100 border border-amber-200 rounded-xl text-left transition-colors disabled:opacity-50"
          >
            <CheckCircle2 className="w-5 h-5 text-amber-600 flex-shrink-0" />
            <div>
              <div className="font-medium text-amber-800">
                {language === 'zh' ? '终止分析并总结结果' : 'Terminate and Summarize'}
              </div>
              <div className="text-xs text-amber-600">
                {language === 'zh'
                  ? '停止分析轮次，根据已有结果生成报告'
                  : 'Stop rounds, generate report from current results'}
              </div>
            </div>
          </button>
          <button
            onClick={onClose}
            disabled={stopping}
            className="w-full px-4 py-3 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl font-medium transition-colors text-center disabled:opacity-50"
          >
            {language === 'zh' ? '按错了，继续' : 'Cancel, Continue'}
          </button>
          {stopping && (
            <div className="flex items-center gap-2 text-sm text-blue-600 pt-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              {language === 'zh' ? '正在停止...' : 'Stopping...'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
