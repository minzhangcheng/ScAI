import { PlanReviewView } from './planReview/PlanReviewView'
import {
  usePlanReviewController,
  type PlanReviewDialogProps,
} from './planReview/usePlanReviewController'

export type { AnalysisPlan } from './planReview/usePlanReviewController'

export function PlanReviewDialog(props: PlanReviewDialogProps) {
  return <PlanReviewView controller={usePlanReviewController(props)} />
}
