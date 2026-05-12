import { X } from 'lucide-react'

interface ErrorBannerProps {
  message: string
  onDismiss: () => void
}

export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 flex items-center gap-2">
      <span className="flex-shrink-0">⚠</span>
      <span className="flex-1">{message}</span>
      <button
        onClick={onDismiss}
        className="flex-shrink-0 text-red-500 hover:text-red-700"
        aria-label="Dismiss"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}