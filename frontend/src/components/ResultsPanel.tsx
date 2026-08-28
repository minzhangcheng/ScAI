import { ResultsView } from './results/ResultsView'
import { useResultsController } from './results/useResultsController'

export { mergeFigureMetadata } from './results/useResultsController'

export function ResultsPanel() {
  return <ResultsView controller={useResultsController()} />
}
