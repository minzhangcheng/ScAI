import { Dna } from 'lucide-react'

export function LoadingScreen() {
  return (
    <div className="min-h-screen bg-surface-900 flex flex-col items-center justify-center gap-4">
      <div className="p-4 bg-gradient-to-br from-accent-500 to-accent-700 rounded-2xl shadow-glow-accent">
        <Dna className="w-8 h-8 text-white animate-spin" />
      </div>
      <p className="text-surface-400 text-sm font-medium tracking-wide">Loading ScAI...</p>
    </div>
  )
}