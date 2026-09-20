import { useCallback, useState } from 'react'

import { useCountdownTimer } from '../../hooks/useCountdownTimer'
import type { Language } from '../../i18n/translations'

export interface ExplorationReport {
  cluster_overview: string
  data_quality_warnings: string[]
  strongest_heterogeneity: string[]
  key_pathway_changes: string[]
  abnormal_markers: string[]
  low_confidence_annotations: string[]
  pi_direction_assessment: string
  direction_literature_evidence?: DirectionLiteratureEvidence[]
  research_direction_candidates: DirectionCandidate[]
}

export interface PubMedEvidenceItem {
  pmid: string
  title: string
  year?: string
  url?: string
  relevance: string
  supports: string
  source?: string
  species_context?: string
  tissue_context?: string
  why_it_matters: string
}

export interface DirectionSearchQuery {
  category: string
  query: string
  result_count?: number
  relevant_count?: number
  degraded?: boolean
}

export interface DirectionLiteratureEvidence {
  direction_id: string
  direction: string
  status: string
  search_tier?: string
  confidence?: string
  negative_validation_confidence?: string
  direct_evidence_count?: number
  analogous_evidence_count?: number
  contradictory_evidence_count?: number
  methodological_precedent_count?: number
  direct_evidence?: PubMedEvidenceItem[]
  analogous_evidence?: PubMedEvidenceItem[]
  contradictory_evidence?: PubMedEvidenceItem[]
  methodological_precedent?: PubMedEvidenceItem[]
  species_tissue_qualification?: string
  support_level?: string
  contradiction_level?: string
  prior_art_level?: string
  novelty_status?: string
  similar_published_directions?: Array<{ pmid?: string; title?: string; url?: string }>
  knowledge_gap_summary?: string
  fallback_tier?: string
  scheduler_limitation?: string
  queries?: DirectionSearchQuery[]
  summary?: string
  limitations?: string[]
  timed_out?: boolean
  timeout_seconds?: number | null
  policy_version?: string
  plan_status?: string
  assessment_status?: string
  assessment_uncertainty?: string[]
  assessment_limitations?: string[]
  task_executions?: Array<{ task_id?: string; status?: string; returned_count?: number; compiled_query?: string }>
  article_inventory?: Array<{ pmid?: string; title?: string; source?: string; task_ids?: string[] }>
  extractions?: Array<{ pmid?: string; finding?: string; source_passage?: string; applicability_limits?: string[] }>
  unexecuted_task_ids?: string[]
}

export interface DirectionCandidate {
  direction_id?: string
  direction: string
  rationale: string
  expected_innovation?: string
  expected_novelty?: string
  data_reliability?: string
  literature_novelty?: string
  functional_impact?: string
  overall_priority?: string
  evidence_ref?: string
  biology_expectedness?: string
  data_novelty?: string
  mechanistic_importance?: string
  data_sufficiency?: string
  methodological_gap?: string
  risk_flags?: string[]
  prerequisite_dependencies?: string[]
  species_tissue_qualification?: string
  core_observation?: string
  main_hypothesis?: string
  alternative_hypotheses?: string[]
  artifact_risks?: string[]
  supporting_data?: Array<{ source?: string; evidence?: string; strength?: string }>
  contradictory_or_missing_evidence?: string[]
  contradicts_stated_premise?: boolean
  validation_plan?: {
    next_analysis?: string[]
    wet_lab_validation?: string[]
    decision_criteria?: string[]
  }
  biological_plausibility?: string
  validation_feasibility?: string
  artifact_risk?: string
  recommendation?: string
  pi_review_summary?: string
}

export interface SelectedDirection {
  direction: string
  notes: string
  isCustom: boolean
}

export interface ExplorationReportDialogProps {
  report: ExplorationReport
  roundSummary: string
  explorationReportUrl?: string
  detailedReportUrl?: string
  language: Language
  countdownSeconds: number  // -1=manual, 0=no wait, >0=countdown
  onChooseDirections: (directions: SelectedDirection[]) => Promise<void>
  onLetPiDecide: () => Promise<void>
  onRequestSupplementary: (requestText: string) => Promise<void>
  onClose: () => void
  isSupplementaryRunning?: boolean
}

export function useExplorationReportController({
  report,
  roundSummary,
  explorationReportUrl,
  detailedReportUrl,
  language,
  countdownSeconds,
  onChooseDirections,
  onLetPiDecide,
  onRequestSupplementary,
  onClose,
  isSupplementaryRunning = false,
}: ExplorationReportDialogProps) {
  const isZh = language === 'zh'
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set())
  const [candidateNotes, setCandidateNotes] = useState<Record<number, string>>({})
  const [customDirections, setCustomDirections] = useState<Array<{ direction: string; notes: string }>>([])
  const [submitting, setSubmitting] = useState(false)
  const [showFullReport, setShowFullReport] = useState(false)
  const [showSupplementaryInput, setShowSupplementaryInput] = useState(false)
  const [expandedEvidence, setExpandedEvidence] = useState<Set<number>>(new Set())
  const toggleEvidence = (index: number) => {
    setExpandedEvidence(prev => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }
  const handlePiDecide = useCallback(async () => {
    setSubmitting(true)
    try {
      await onLetPiDecide()
    } finally {
      setSubmitting(false)
    }
  }, [onLetPiDecide])
  const { timeRemaining, hasCountdown } = useCountdownTimer({
    countdownSeconds,
    onExpire: handlePiDecide,
  })
  const toggleCandidate = (index: number) => {
    setSelectedIndices(prev => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }
  const updateCandidateNote = (index: number, note: string) => {
    setCandidateNotes(prev => ({ ...prev, [index]: note }))
  }
  const addCustomDirection = () => {
    setCustomDirections(prev => [...prev, { direction: '', notes: '' }])
  }
  const updateCustomDirection = (index: number, field: 'direction' | 'notes', value: string) => {
    setCustomDirections(prev => prev.map((d, i) => i === index ? { ...d, [field]: value } : d))
  }
  const removeCustomDirection = (index: number) => {
    setCustomDirections(prev => prev.filter((_, i) => i !== index))
  }
  const totalSelected = selectedIndices.size + customDirections.filter(d => d.direction.trim()).length
  const canSubmit = totalSelected > 0 && !submitting
  const handleSubmit = useCallback(async () => {
    setSubmitting(true)
    try {
      const directions: SelectedDirection[] = []
      for (const idx of selectedIndices) {
        const c = report.research_direction_candidates[idx]
        if (c) {
          directions.push({
            direction: c.direction,
            notes: candidateNotes[idx] || '',
            isCustom: false,
          })
        }
      }
      for (const d of customDirections) {
        if (d.direction.trim()) {
          directions.push({
            direction: d.direction.trim(),
            notes: d.notes.trim(),
            isCustom: true,
          })
        }
      }
      if (directions.length > 0) {
        await onChooseDirections(directions)
      }
    } finally {
      setSubmitting(false)
    }
  }, [selectedIndices, candidateNotes, customDirections, report, onChooseDirections])
  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m}:${sec.toString().padStart(2, '0')}`
  }
  return {
    addCustomDirection, canSubmit, candidateNotes, customDirections, detailedReportUrl,
    expandedEvidence, explorationReportUrl, formatTime, handlePiDecide, handleSubmit,
    hasCountdown, isSupplementaryRunning, isZh, language, onClose, onRequestSupplementary,
    removeCustomDirection, report, roundSummary, selectedIndices, setShowFullReport,
    setShowSupplementaryInput, showFullReport, showSupplementaryInput, submitting,
    timeRemaining, toggleCandidate, toggleEvidence, totalSelected, updateCandidateNote,
    updateCustomDirection,
  }
}

export type ExplorationReportController = ReturnType<typeof useExplorationReportController>
