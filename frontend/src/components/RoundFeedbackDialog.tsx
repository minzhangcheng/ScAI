import { RoundFeedbackView } from './roundFeedback/RoundFeedbackView'
import {
  useRoundFeedbackController,
  type RoundFeedbackDialogProps,
} from './roundFeedback/useRoundFeedbackController'

export type { UploadedFileWithAgent } from './roundFeedback/useRoundFeedbackController'

export function RoundFeedbackDialog(props: RoundFeedbackDialogProps) {
  return <RoundFeedbackView controller={useRoundFeedbackController(props)} />
}
