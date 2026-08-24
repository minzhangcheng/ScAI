import { ScAIIcon } from './ScAIIcon'

export function LoadingScreen() {
  return (
    <div className="min-h-screen bg-surface-900 dark:bg-surface-950 flex flex-col items-center justify-center gap-4">
      <ScAIIcon className="w-16 h-16 animate-pulse" />
      <p className="text-surface-400 dark:text-surface-300 text-sm font-medium tracking-wide">Loading ScAI...</p>
    </div>
  )
}
