import { Users, Microscope, Code2, FlaskConical, BookOpen, Palette, Eye, Wrench, Bot } from 'lucide-react'
import type { AgentType } from '../stores/appStore'
import type { TranslationKey } from '../types'

export const AGENT_OPTIONS: ReadonlyArray<{ value: AgentType; labelKey: TranslationKey }> = [
  { value: 'pi', labelKey: 'chatTargetPi' },
  { value: 'pathology', labelKey: 'chatTargetPathology' },
  { value: 'bioinformatics', labelKey: 'chatTargetBioinformatics' },
  { value: 'biochemistry', labelKey: 'chatTargetBiochemistry' },
]

export const agentIcons: Record<AgentType, typeof Users> = {
  pi: Users,
  pathology: Microscope,
  bioinformatics: Code2,
  biochemistry: FlaskConical,
  literature: BookOpen,
  painter: Palette,
  supervisor: Eye,
  technique: Wrench,
}

export const agentBgColors: Record<AgentType, string> = {
  pi: 'bg-violet-50 dark:bg-violet-950/40 border-violet-200 dark:border-violet-800',
  pathology: 'bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-800',
  bioinformatics: 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800',
  biochemistry: 'bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800',
  literature: 'bg-blue-50 dark:bg-blue-950/40 border-blue-200 dark:border-blue-800',
  painter: 'bg-pink-50 dark:bg-pink-950/40 border-pink-200 dark:border-pink-800',
  supervisor: 'bg-cyan-50 dark:bg-cyan-950/40 border-cyan-200 dark:border-cyan-800',
  technique: 'bg-teal-50 dark:bg-teal-950/40 border-teal-200 dark:border-teal-800',
}

export const agentGradients: Record<AgentType, string> = {
  pi: 'from-violet-500 to-violet-600',
  pathology: 'from-rose-500 to-rose-600',
  bioinformatics: 'from-emerald-500 to-emerald-600',
  biochemistry: 'from-amber-500 to-amber-600',
  literature: 'from-blue-500 to-blue-600',
  painter: 'from-pink-500 to-pink-600',
  supervisor: 'from-cyan-500 to-cyan-600',
  technique: 'from-teal-500 to-teal-600',
}

export const FALLBACK_AGENT_ICON = Bot
