import { createPortal } from 'react-dom'
import { X, AlertTriangle } from 'lucide-react'

interface ConfirmDialogProps {
  open: boolean
  title: string
  message: string
  confirmLabel?: string
  variant?: 'danger' | 'default'
  titleIcon?: React.ReactNode
  confirmIcon?: React.ReactNode
  onConfirm: () => void
  onCancel: () => void
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = 'Confirm',
  variant = 'default',
  titleIcon,
  confirmIcon,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  if (!open) return null

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal w-full max-w-sm mx-4 overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200 dark:border-surface-700">
          <div className="flex items-center gap-2">
            {titleIcon}
            <h2 className="text-lg font-display font-semibold text-surface-900 dark:text-surface-100">{title}</h2>
          </div>
          <button
            onClick={onCancel}
            className="p-1 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-surface-400" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-5">
          {variant === 'danger' && (
            <div className="flex gap-3 mb-4">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
              <p className="text-sm text-surface-700 dark:text-surface-200 leading-relaxed">{message}</p>
            </div>
          )}
          {variant === 'default' && (
            <p className="text-sm text-surface-700 dark:text-surface-200 leading-relaxed">{message}</p>
          )}
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-800">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium text-surface-600 dark:text-surface-300 hover:bg-surface-100 dark:hover:bg-surface-700 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`px-4 py-2 text-sm font-medium text-white rounded-lg transition-colors flex items-center gap-2 ${
              variant === 'danger'
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-accent-600 hover:bg-accent-700'
            }`}
          >
            {confirmIcon}
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}
