import type { AnalysisPanelController } from './useAnalysisPanelController'
import { AnalysisControls } from './AnalysisControls'
import { AnalysisDialogs } from './AnalysisDialogs'
import { AnalysisStatusView } from './AnalysisStatusView'

interface AnalysisPanelViewProps {
  controller: AnalysisPanelController
}

export function AnalysisPanelView({ controller }: AnalysisPanelViewProps) {
  return (
    <div className="flex flex-col h-full">
      <AnalysisControls controller={controller} />
      <AnalysisStatusView controller={controller} />
      <AnalysisDialogs controller={controller} />
    </div>
  )
}
