import { ReportView } from './report/ReportView'
import { useReportController } from './report/useReportController'

export {
  partialReportNotice,
  reportBranchQuery,
  reportCompletionContent,
} from './report/useReportController'
export type { DetailedReport } from './report/useReportController'

export function ReportPanel() {
  return <ReportView controller={useReportController()} />
}
