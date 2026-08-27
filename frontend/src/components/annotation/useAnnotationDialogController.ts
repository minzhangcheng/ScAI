import { useCallback, useEffect, useRef, useState } from 'react'

import type { Language } from '../../i18n/translations'

export interface ClusterAnnotation {
  cluster_id: string
  cell_type: string
  subtypes: string[]
  key_markers: string[]
  confidence: string  // "high" / "medium" / "low"
  evidence: string
  alternative_cell_types: string[]
  low_confidence_reason: string
  user_confirmed: boolean
  user_override: string | null
  group_proportions?: Record<string, number>
  requires_reclustering?: boolean;
  reclustering_reason?: string;
  suggested_reclustering_scope?: string | null;
}

export interface AnnotationConfirmationDialogProps {
  annotations: ClusterAnnotation[]
  tissue: string
  organism: string
  tissueSummary: string
  reportHtmlUrl?: string
  detailedReportUrl?: string
  validationStatus?: string;
  validationSummary?: string;
  validationTextSummary?: string;
  validationResult?: { artifacts?: Array<{ type: string; name: string; path: string }> } & Record<string, any>;
  reclusteringRecommendations?: Array<{
    cluster_id: string; requires_reclustering: boolean;
    reclustering_reason: string; suggested_reclustering_scope?: string | null;
  }>;
  proportionGroupLabels?: string[]
  proportionDimension?: string
  language: Language
  countdownSeconds: number  // countdown for high-confidence auto-confirm
  onConfirm: (annotations: Array<{ cluster_id: string; user_override?: string; user_confirmed: boolean }>) => Promise<void>
  onRequestSupplementary: (requestText: string) => Promise<void>
  onClose: () => void
  isSupplementaryRunning?: boolean
}

export function useAnnotationDialogController({
  annotations: initialAnnotations,
  tissue,
  organism,
  tissueSummary,
  reportHtmlUrl,
  detailedReportUrl,
  validationSummary,
  validationTextSummary,
  validationResult,
  reclusteringRecommendations,
  proportionGroupLabels = [],
  proportionDimension = 'group',
  language,
  countdownSeconds,
  onConfirm,
  onRequestSupplementary,
  onClose,
  isSupplementaryRunning = false,
}: AnnotationConfirmationDialogProps) {
  const [annotations, setAnnotations] = useState(initialAnnotations)
  const [showSupplementaryInput, setShowSupplementaryInput] = useState(false)
  const [editingCluster, setEditingCluster] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [countdown, setCountdown] = useState(countdownSeconds > 0 ? countdownSeconds : 0)
  const [submitting, setSubmitting] = useState(false)
  const [expandedClusters, setExpandedClusters] = useState<Set<string>>(new Set())
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const annotationKey = JSON.stringify({
    labels: proportionGroupLabels,
    dimension: proportionDimension,
    annotations: initialAnnotations.map(a => [
      a.cluster_id, a.confidence, a.cell_type, a.user_override ?? null,
      a.group_proportions ?? {},
      a.requires_reclustering ?? false, a.reclustering_reason ?? '',
      a.suggested_reclustering_scope ?? null,
    ]),
  })
  useEffect(() => {
    setAnnotations(initialAnnotations)
  }, [annotationKey])
  const lowConfidenceClusters = annotations.filter(a => a.confidence === 'low')
  const allLowConfidenceConfirmed = lowConfidenceClusters.every(a => a.user_confirmed)
  const hasLowConfidence = lowConfidenceClusters.length > 0
  const handleConfirmAll = useCallback(async () => {
    setSubmitting(true)
    try {
      const payload = annotations.map(a => ({
        cluster_id: a.cluster_id,
        user_override: a.user_override || undefined,
        user_confirmed: true
      }))
      await onConfirm(payload)
    } finally {
      setSubmitting(false)
    }
  }, [annotations, onConfirm])
  const confirmAllRef = useRef(handleConfirmAll)
  useEffect(() => { confirmAllRef.current = handleConfirmAll }, [handleConfirmAll])
  useEffect(() => {
    if (countdownSeconds <= 0 || hasLowConfidence) return
    intervalRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          if (intervalRef.current) clearInterval(intervalRef.current)
          confirmAllRef.current()
          return 0
        }
        return prev - 1
      })
    }, 1000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [countdownSeconds, hasLowConfidence])
  const stopCountdown = () => {
    setCountdown(0)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }
  const handleEdit = (clusterId: string, currentType: string) => {
    stopCountdown()
    setEditingCluster(clusterId)
    setEditValue(currentType)
  }
  const handleSaveEdit = (clusterId: string) => {
    setAnnotations(prev => prev.map(a =>
      a.cluster_id === clusterId
        ? { ...a, user_override: editValue, user_confirmed: true, cell_type: editValue }
        : a
    ))
    setEditingCluster(null)
  }
  const handleConfirmCluster = (clusterId: string) => {
    stopCountdown()
    setAnnotations(prev => prev.map(a =>
      a.cluster_id === clusterId ? { ...a, user_confirmed: true } : a
    ))
  }
  const toggleExpanded = (clusterId: string) => {
    setExpandedClusters(prev => {
      const next = new Set(prev)
      if (next.has(clusterId)) {
        next.delete(clusterId)
      } else {
        next.add(clusterId)
      }
      return next
    })
  }
  return {
    allLowConfidenceConfirmed, annotations, countdown, detailedReportUrl, editingCluster,
    editValue, expandedClusters, handleConfirmAll, handleConfirmCluster, handleEdit,
    handleSaveEdit, hasLowConfidence, initialAnnotations, isSupplementaryRunning, language,
    lowConfidenceClusters, onClose, onRequestSupplementary, organism, proportionDimension,
    proportionGroupLabels, reclusteringRecommendations, reportHtmlUrl, setEditValue,
    setShowSupplementaryInput, showSupplementaryInput, submitting, tissue, tissueSummary,
    toggleExpanded, validationResult, validationSummary, validationTextSummary,
  }
}

export type AnnotationDialogController = ReturnType<typeof useAnnotationDialogController>
