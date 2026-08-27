import { AnalysisPanelImpl } from './analysisPanel/AnalysisPanelImpl'

export function buildDirectionsPayload(
  directions: Array<{ direction: string; notes: string; isCustom?: boolean }>,
) {
  return directions.map(direction => ({
    direction: direction.direction,
    notes: direction.notes ?? '',
    is_custom: !!direction.isCustom,
  }))
}

export function AnalysisPanel() {
  return <AnalysisPanelImpl />
}
