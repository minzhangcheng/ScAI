import { useState, useEffect, useRef, useCallback } from 'react'
import {
  X,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Edit3,
  Check,
  FileDown,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { Language } from '../i18n/translations'

interface ClusterAnnotation {
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
}

interface AnnotationConfirmationDialogProps {
  annotations: ClusterAnnotation[]
  tissue: string
  organism: string
  tissueSummary: string
  reportHtmlUrl?: string
  language: Language
  countdownSeconds: number  // countdown for high-confidence auto-confirm
  onConfirm: (annotations: Array<{ cluster_id: string; user_override?: string; user_confirmed: boolean }>) => Promise<void>
  onClose: () => void
}

export function AnnotationConfirmationDialog({
  annotations: initialAnnotations,
  tissue,
  organism,
  tissueSummary,
  reportHtmlUrl,
  language,
  countdownSeconds,
  onConfirm,
  onClose
}: AnnotationConfirmationDialogProps) {
  const [annotations, setAnnotations] = useState(initialAnnotations)
  const [editingCluster, setEditingCluster] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [countdown, setCountdown] = useState(countdownSeconds > 0 ? countdownSeconds : 0)
  const [submitting, setSubmitting] = useState(false)
  const [expandedClusters, setExpandedClusters] = useState<Set<string>>(new Set())
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

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

  // Ref to always hold the latest handleConfirmAll for timer callback
  const confirmAllRef = useRef(handleConfirmAll)
  useEffect(() => { confirmAllRef.current = handleConfirmAll }, [handleConfirmAll])

  // Countdown timer for auto-confirm (only when no low-confidence clusters need manual review)
  useEffect(() => {
    if (countdownSeconds <= 0 || hasLowConfidence) return

    intervalRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          if (intervalRef.current) clearInterval(intervalRef.current)
          // Auto-confirm all (use ref to avoid stale closure)
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
    // Stop countdown when user starts manual editing to prevent auto-submit
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
    // Stop countdown when user manually confirms a cluster to prevent race
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

  const confidenceColor = (conf: string) => {
    switch (conf) {
      case 'high': return 'bg-green-100 text-green-800'
      case 'medium': return 'bg-amber-100 text-amber-800'
      case 'low': return 'bg-red-100 text-red-800'
      default: return 'bg-slate-100 text-slate-800'
    }
  }

  const confidenceIcon = (conf: string) => {
    switch (conf) {
      case 'high': return <CheckCircle2 className="w-4 h-4 text-green-600" />
      case 'medium': return <Clock className="w-4 h-4 text-amber-600" />
      case 'low': return <AlertTriangle className="w-4 h-4 text-red-600" />
      default: return null
    }
  }

  const hasDetails = (ann: ClusterAnnotation) =>
    (ann.alternative_cell_types && ann.alternative_cell_types.length > 0) ||
    (ann.low_confidence_reason && ann.low_confidence_reason.length > 0)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-xl max-w-[90vw] w-[1000px] mx-4 max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-800">
                {language === 'zh' ? '细胞注释确认' : 'Cell Annotation Confirmation'}
              </h2>
              <p className="text-sm text-slate-500 mt-1">
                {organism} — {tissue}
                {tissueSummary && ` | ${tissueSummary.substring(0, 100)}`}
              </p>
            </div>
            {reportHtmlUrl && (
              <a
                href={reportHtmlUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-violet-50 text-violet-700 hover:bg-violet-100 rounded-lg text-sm font-medium transition-colors border border-violet-200"
              >
                <FileDown className="w-4 h-4" />
                {language === 'zh' ? '查看分析报告' : 'View Analysis Report'}
              </a>
            )}
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-lg">
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Annotations panel (full width now) */}
          <div className="flex-1 flex flex-col overflow-hidden relative">
            <div className="flex-1 overflow-y-auto px-6 py-4">
              {/* Warning banner for low confidence */}
              {hasLowConfidence && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                  <div className="text-sm text-red-700">
                    {language === 'zh'
                      ? `${lowConfidenceClusters.length} 个亚群的注释置信度较低，需要手动确认后才能继续分析。`
                      : `${lowConfidenceClusters.length} cluster(s) have low confidence annotations and require manual confirmation.`}
                  </div>
                </div>
              )}

              {/* Countdown for auto-confirm (only when no low confidence) */}
              {!hasLowConfidence && countdown > 0 && (
                <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg flex items-center gap-2">
                  <Clock className="w-5 h-5 text-blue-500" />
                  <span className="text-sm text-blue-700">
                    {language === 'zh'
                      ? `所有注释置信度较高，${countdown} 秒后自动确认...`
                      : `All annotations have high/medium confidence. Auto-confirming in ${countdown}s...`}
                  </span>
                </div>
              )}

              {/* Annotations table */}
              <table className="w-full text-sm">
            <thead className="sticky top-0 bg-white">
              <tr className="border-b text-left text-slate-500">
                <th className="py-2 pr-3 w-20">Cluster</th>
                <th className="py-2 pr-3 w-40">
                  {language === 'zh' ? '细胞类型' : 'Cell Type'}
                </th>
                <th className="py-2 pr-3 w-24">
                  {language === 'zh' ? '置信度' : 'Confidence'}
                </th>
                <th className="py-2 pr-3">
                  {language === 'zh' ? '标记基因' : 'Key Markers'}
                </th>
                <th className="py-2 w-24 text-right">
                  {language === 'zh' ? '操作' : 'Action'}
                </th>
              </tr>
            </thead>
            <tbody>
              {annotations.map(ann => {
                const isExpanded = expandedClusters.has(ann.cluster_id)
                const showExpandButton = hasDetails(ann)

                return (
                <tr
                  key={ann.cluster_id}
                  className={`border-b last:border-0 ${
                    ann.confidence === 'low' && !ann.user_confirmed
                      ? 'bg-red-50'
                      : ann.user_confirmed
                      ? 'bg-green-50'
                      : ''
                  }`}
                >
                  <td className="py-3 pr-3 font-mono text-slate-700">{ann.cluster_id}</td>
                  <td className="py-3 pr-3">
                    {editingCluster === ann.cluster_id ? (
                      <div className="flex items-center gap-1">
                        <input
                          type="text"
                          value={editValue}
                          onChange={e => setEditValue(e.target.value)}
                          className="border rounded px-2 py-1 text-sm w-full"
                          autoFocus
                          onKeyDown={e => e.key === 'Enter' && handleSaveEdit(ann.cluster_id)}
                        />
                        <button
                          onClick={() => handleSaveEdit(ann.cluster_id)}
                          className="p-1 text-green-600 hover:bg-green-100 rounded"
                        >
                          <Check className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <div>
                        <div className="flex items-center gap-2">
                          <span className={ann.user_override ? 'font-medium text-violet-700' : 'text-slate-800'}>
                            {ann.user_override || ann.cell_type}
                          </span>
                          {ann.user_override && (
                            <span className="text-xs text-slate-400 line-through">{initialAnnotations.find(a => a.cluster_id === ann.cluster_id)?.cell_type}</span>
                          )}
                          {showExpandButton && (
                            <button
                              onClick={() => toggleExpanded(ann.cluster_id)}
                              className="p-0.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded"
                              title={isExpanded ? 'Collapse' : 'Show details'}
                            >
                              {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                            </button>
                          )}
                        </div>
                        {/* Expanded details for low/medium confidence */}
                        {isExpanded && (
                          <div className="mt-2 space-y-1.5">
                            {ann.alternative_cell_types && ann.alternative_cell_types.length > 0 && (
                              <div className="px-2 py-1.5 bg-amber-50 border border-amber-200 rounded text-xs text-amber-800">
                                <span className="font-medium">{language === 'zh' ? '可能的其他类型: ' : 'Alternatives: '}</span>
                                {ann.alternative_cell_types.join(', ')}
                              </div>
                            )}
                            {ann.low_confidence_reason && (
                              <div className="px-2 py-1.5 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                                <span className="font-medium">{language === 'zh' ? '原因: ' : 'Reason: '}</span>
                                {ann.low_confidence_reason}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </td>
                  <td className="py-3 pr-3">
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${confidenceColor(ann.confidence)}`}>
                      {confidenceIcon(ann.confidence)}
                      {ann.confidence}
                    </span>
                  </td>
                  <td className="py-3 pr-3 text-slate-600 text-xs">
                    {ann.key_markers.slice(0, 10).join(', ')}
                    {ann.key_markers.length > 10 && ` (+${ann.key_markers.length - 10})`}
                  </td>
                  <td className="py-3 text-right">
                    <div className="flex items-center justify-end gap-1">
                      {!ann.user_confirmed && (
                        <button
                          onClick={() => handleConfirmCluster(ann.cluster_id)}
                          className="p-1.5 text-green-600 hover:bg-green-100 rounded-lg"
                          title={language === 'zh' ? '确认' : 'Confirm'}
                        >
                          <Check className="w-4 h-4" />
                        </button>
                      )}
                      <button
                        onClick={() => handleEdit(ann.cluster_id, ann.user_override || ann.cell_type)}
                        className="p-1.5 text-slate-500 hover:bg-slate-100 rounded-lg"
                        title={language === 'zh' ? '修改' : 'Edit'}
                      >
                        <Edit3 className="w-4 h-4" />
                      </button>
                      {ann.user_confirmed && (
                        <CheckCircle2 className="w-4 h-4 text-green-500" />
                      )}
                    </div>
                  </td>
                </tr>
              )})}
            </tbody>
          </table>
            </div>
            {/* Footer */}
            <div className="flex items-center justify-between px-6 py-4 border-t bg-slate-50">
              <div className="text-sm text-slate-500">
                {annotations.filter(a => a.user_confirmed).length} / {annotations.length}{' '}
                {language === 'zh' ? '已确认' : 'confirmed'}
              </div>
              <button
                onClick={handleConfirmAll}
                disabled={submitting || (hasLowConfidence && !allLowConfidenceConfirmed)}
                className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  submitting || (hasLowConfidence && !allLowConfidenceConfirmed)
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                    : 'bg-violet-600 text-white hover:bg-violet-700 shadow-md'
                }`}
              >
                {submitting
                  ? (language === 'zh' ? '提交中...' : 'Submitting...')
                  : hasLowConfidence && !allLowConfidenceConfirmed
                  ? (language === 'zh' ? '请先确认低置信度亚群' : 'Confirm low-confidence clusters first')
                  : (language === 'zh' ? '确认全部注释' : 'Confirm All Annotations')}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
