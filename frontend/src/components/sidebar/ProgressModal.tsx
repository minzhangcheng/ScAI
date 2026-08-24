// Export/import progress modal. Split out of Sidebar.tsx; JSX copied verbatim.
import { Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import { useTranslation } from '../../i18n/translations'

export interface ProgressInfo {
  action: string
  detail: string
  percent?: number
  closable?: boolean
  success?: boolean
  onConfirm?: () => void
  onCancel?: () => void
  cancelLabel?: string
}

interface ProgressModalProps {
  progressInfo: ProgressInfo | null
  setProgressInfo: (info: ProgressInfo | null) => void
  t: ReturnType<typeof useTranslation>['t']
  language: string
}

export function ProgressModal({ progressInfo, setProgressInfo, t, language }: ProgressModalProps) {
  if (!progressInfo) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal p-6 w-80 text-center space-y-4 animate-scale-in">
        {/* 加载动画 - 进行中时显示 */}
        {!progressInfo.closable && (
          <Loader2 className="w-8 h-8 animate-spin text-accent-600 dark:text-accent-400 mx-auto" />
        )}

        {/* 成功图标 */}
        {progressInfo.closable && progressInfo.success && (
          <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400 mx-auto" />
        )}

        {/* 失败/完成图标（导出完成或导入失败） */}
        {progressInfo.closable && !progressInfo.success && (
          <AlertCircle className="w-8 h-8 text-red-500 mx-auto" />
        )}

        {/* 文本信息 */}
        <div>
          <p className="text-sm font-semibold text-surface-800 dark:text-surface-100">{progressInfo.action}</p>
          <p className="text-xs text-surface-500 dark:text-surface-400 mt-1">{progressInfo.detail}</p>
        </div>

        {/* 进度条（如需要） */}
        {progressInfo.percent != null && (
          <div className="w-full bg-surface-200 dark:bg-surface-700 rounded-full h-2 overflow-hidden">
            <div
              className="bg-accent-600 h-full rounded-full transition-all duration-300"
              style={{ width: `${progressInfo.percent}%` }}
            />
          </div>
        )}

        {/* 确定/关闭按钮 - 仅在可关闭时显示 */}
        {progressInfo.closable && (
          <button
            onClick={() => {
              if (progressInfo.onConfirm) {
                progressInfo.onConfirm()
              } else {
                setProgressInfo(null)
              }
            }}
            className="w-full px-4 py-2 bg-accent-600 text-white text-sm rounded-lg hover:bg-accent-700 transition-colors font-medium"
          >
            {t('confirm')}
          </button>
        )}

        {/* 取消按钮 - 进行中且支持取消时显示 */}
        {!progressInfo.closable && progressInfo.onCancel && (
          <button
            onClick={() => { progressInfo.onCancel && progressInfo.onCancel() }}
            className="w-full px-4 py-2 bg-surface-100 text-surface-700 text-sm rounded-lg hover:bg-surface-200 transition-colors font-medium"
          >
            {progressInfo.cancelLabel || (language === 'zh' ? '取消' : 'Cancel')}
          </button>
        )}
      </div>
    </div>
  )
}
