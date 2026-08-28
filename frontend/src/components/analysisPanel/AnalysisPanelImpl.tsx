import { Users } from 'lucide-react'

import { AnalysisPanelView } from './AnalysisPanelView'
import { useAnalysisPanelController } from './useAnalysisPanelController'

export function AnalysisPanelImpl() {
  const controller = useAnalysisPanelController()
  const { currentSession, t } = controller

  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 dark:bg-slate-800 rounded-full mb-6">
          <Users className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="font-display text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
          {t('startNewAnalysis')}
        </h2>
        <p className="text-slate-500 dark:text-slate-400 max-w-md">
          {t('welcomeDesc')}
        </p>
      </div>
    )
  }

  return <AnalysisPanelView controller={controller} />
}
