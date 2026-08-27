import { ExplorationReportView } from './exploration/ExplorationReportView'
import {
  useExplorationReportController,
  type ExplorationReportDialogProps,
} from './exploration/useExplorationReportController'

export type {
  DirectionCandidate,
  DirectionLiteratureEvidence,
  DirectionSearchQuery,
  ExplorationReport,
  PubMedEvidenceItem,
  SelectedDirection,
} from './exploration/useExplorationReportController'

export function ExplorationReportDialog(props: ExplorationReportDialogProps) {
  return <ExplorationReportView controller={useExplorationReportController(props)} />
}
