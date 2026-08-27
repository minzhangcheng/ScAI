import { AnnotationDialogView } from './annotation/AnnotationDialogView'
import {
  useAnnotationDialogController,
  type AnnotationConfirmationDialogProps,
} from './annotation/useAnnotationDialogController'

export type { ClusterAnnotation } from './annotation/useAnnotationDialogController'

export function AnnotationConfirmationDialog(props: AnnotationConfirmationDialogProps) {
  return <AnnotationDialogView controller={useAnnotationDialogController(props)} />
}
