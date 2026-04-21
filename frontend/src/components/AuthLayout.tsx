import { ReactNode } from 'react'
import { Dna } from 'lucide-react'

export function AuthLayout({ children, subtitle, onSubmit }: {
  children: ReactNode;
  subtitle: string;
  onSubmit?: (e: React.FormEvent) => void;
}) {
  return (
    <div className="min-h-screen bg-surface-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex p-3 bg-gradient-to-br from-accent-500 to-accent-700 rounded-2xl shadow-glow-accent mb-4">
            <Dna className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-surface-800">ScAI</h1>
          <p className="text-surface-500 mt-1">{subtitle}</p>
        </div>
        <form onSubmit={onSubmit} className="bg-white rounded-xl shadow-sm border border-surface-200 p-6 space-y-4">
          {children}
        </form>
      </div>
    </div>
  )
}
