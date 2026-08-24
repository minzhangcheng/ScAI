import { ReactNode } from 'react'
import { ScAIIcon } from './ScAIIcon'

export function AuthLayout({ children, subtitle, onSubmit }: {
  children: ReactNode;
  subtitle: string;
  onSubmit?: (e: React.FormEvent) => void;
}) {
  return (
    <div className="min-h-screen bg-surface-50 dark:bg-surface-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <ScAIIcon className="w-16 h-16 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-surface-800 dark:text-surface-100">ScAI</h1>
          <p className="text-surface-500 dark:text-surface-400 mt-1">{subtitle}</p>
        </div>
        <form onSubmit={onSubmit} className="bg-white dark:bg-surface-900 rounded-xl shadow-sm border border-surface-200 dark:border-surface-700 p-6 space-y-4">
          {children}
        </form>
      </div>
    </div>
  )
}
