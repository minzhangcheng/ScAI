import {
  AlertTriangle, Check, CheckCircle2, ChevronDown, ChevronUp, Clock, Edit3, FileDown, X,
} from 'lucide-react'

import type { Language } from '../../i18n/translations'
import { SupplementaryAnalysisModal } from '../SupplementaryAnalysisModal'
import type {
  AnnotationDialogController,
  ClusterAnnotation,
} from './useAnnotationDialogController'

interface AnnotationDialogViewProps {
  controller: AnnotationDialogController
}

function confidenceColor(confidence: string) {
  switch (confidence) {
    case 'high': return 'bg-green-100 text-green-800'
    case 'medium': return 'bg-amber-100 text-amber-800'
    case 'low': return 'bg-red-100 text-red-800'
    default: return 'bg-slate-100 text-slate-800'
  }
}

function confidenceIcon(confidence: string) {
  switch (confidence) {
    case 'high': return <CheckCircle2 className="w-4 h-4 text-green-600" />
    case 'medium': return <Clock className="w-4 h-4 text-amber-600" />
    case 'low': return <AlertTriangle className="w-4 h-4 text-red-600" />
    default: return null
  }
}

function hasDetails(annotation: ClusterAnnotation) {
  return Boolean(
    annotation.alternative_cell_types?.length
    || annotation.low_confidence_reason?.length,
  )
}

function proportionHeader(labels: string[], dimension: string, language: Language) {
  let prefix: string
  if (language === 'zh') {
    prefix = dimension === 'sample' ? '样本比例'
      : dimension === 'batch' ? '批次比例' : '细胞比例'
  } else {
    prefix = dimension === 'sample' ? 'Sample Proportion'
      : dimension === 'batch' ? 'Batch Proportion' : 'Cell Proportion'
  }
  const displayLabels = labels.map(label => (
    dimension === 'overall' && label === '整体'
      ? (language === 'zh' ? '整体' : 'Overall')
      : label
  ))
  return displayLabels.length > 0 ? prefix + ' (' + displayLabels.join('/') + ')' : prefix
}

function renderProportions(annotation: ClusterAnnotation, labels: string[]) {
  if (!labels.length) return '—'
  return labels.map(label => {
    const value = annotation.group_proportions?.[label]
    return typeof value === 'number' ? value.toFixed(1) + '%' : '—'
  }).join(' / ')
}

export function AnnotationDialogView({ controller }: AnnotationDialogViewProps) {
  const {
    allLowConfidenceConfirmed, annotations, countdown, detailedReportUrl, editingCluster,
    editValue, expandedClusters, handleConfirmAll, handleConfirmCluster, handleEdit,
    handleSaveEdit, hasLowConfidence, initialAnnotations, isSupplementaryRunning, language,
    lowConfidenceClusters, onClose, onRequestSupplementary, organism, proportionDimension,
    proportionGroupLabels, reclusteringRecommendations, reportHtmlUrl, setEditValue,
    setShowSupplementaryInput, showSupplementaryInput, submitting, tissue, tissueSummary,
    toggleExpanded, validationResult, validationSummary, validationTextSummary,
  } = controller; return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal max-w-[95vw] w-[1120px] mx-4 max-h-[85vh] flex flex-col animate-scale-in">
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-4">
            <div>
              <h2 className="text-lg font-display font-semibold text-slate-800 dark:text-slate-100">
                {language === 'zh' ? '细胞注释确认' : 'Cell Annotation Confirmation'}
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                {organism} — {tissue}
                {tissueSummary && ` | ${tissueSummary.substring(0, 100)}`}
              </p>
            </div>
            {reportHtmlUrl && (
              <a
                href={reportHtmlUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-accent-50 text-accent-700 hover:bg-accent-100 rounded-lg text-sm font-medium transition-colors border border-accent-200"
              >
                <FileDown className="w-4 h-4" />
                {language === 'zh' ? '查看分析报告' : 'View Analysis Report'}
              </a>
            )}
          </div>
          <button
            onClick={onClose}
            aria-label={language === 'zh' ? '关闭' : 'Close'}
            className="p-2 hover:bg-slate-100 rounded-lg"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 flex flex-col overflow-hidden relative">
            <div className="flex-1 overflow-y-auto px-6 py-4">
              {(validationSummary || validationTextSummary ||
                (reclusteringRecommendations && reclusteringRecommendations.length > 0) ||
                (validationResult?.artifacts && validationResult.artifacts.length > 0)) && (
                <div className="mb-4 rounded border border-blue-200 bg-blue-50 p-3 text-sm">
                  <div className="font-semibold text-blue-800 mb-1">Validation evidence</div>
                  {validationSummary && <p className="text-gray-700 mb-2">{validationSummary}</p>}
                  {validationTextSummary && (
                    <pre className="whitespace-pre-wrap text-xs text-gray-600 mb-2">{validationTextSummary}</pre>
                  )}
                  {validationResult?.artifacts && validationResult.artifacts.length > 0 && (
                    <div className="mb-2">
                      <div className="text-xs font-semibold text-gray-700">Artifacts</div>
                      <ul className="list-disc pl-5 text-xs text-blue-700">
                        {validationResult.artifacts.map((art) => (
                          <li key={art.path}>
                            <a href={`/api/file?path=${encodeURIComponent(art.path)}`}
                               target="_blank" rel="noreferrer" className="underline">
                              {art.name} ({art.type})
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {reclusteringRecommendations && reclusteringRecommendations.length > 0 && (
                    <div className="mt-2 rounded border border-amber-300 bg-amber-50 p-2">
                      <div className="font-semibold text-amber-800">Reclustering recommended</div>
                      <ul className="list-disc pl-5 text-xs text-amber-900">
                        {reclusteringRecommendations.map((r) => (
                          <li key={r.cluster_id}>
                            Cluster {r.cluster_id}: {r.reclustering_reason}
                            {r.suggested_reclustering_scope ? ` (${r.suggested_reclustering_scope})` : ''}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
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
                <th className="py-2 pr-3 w-44">
                  {proportionHeader(proportionGroupLabels, proportionDimension, language)}
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
                          <span className={ann.user_override ? 'font-medium text-accent-700' : 'text-slate-800'}>
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
                  <td className="py-3 pr-3 text-slate-600 text-xs whitespace-nowrap">
                    {renderProportions(ann, proportionGroupLabels)}
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
            <div className="flex items-center justify-between px-6 py-4 border-t bg-slate-50">
              <div className="flex items-center gap-3">
                <div className="text-sm text-slate-500">
                  {annotations.filter(a => a.user_confirmed).length} / {annotations.length}{' '}
                  {language === 'zh' ? '已确认' : 'confirmed'}
                </div>
                {detailedReportUrl && (
                  <a
                    href={detailedReportUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-accent-600 hover:text-accent-800 flex items-center gap-1"
                  >
                    <FileDown className="w-3.5 h-3.5" />
                    {language === 'zh' ? '查看详细报告' : 'View Detailed Report'}
                  </a>
                )}
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowSupplementaryInput(true)}
                  disabled={submitting || isSupplementaryRunning}
                  className="px-4 py-2.5 rounded-xl text-sm font-medium border border-amber-300 text-amber-700 hover:bg-amber-50 transition-all disabled:opacity-50"
                >
                  {isSupplementaryRunning
                    ? (language === 'zh' ? '补充分析中...' : 'Analyzing...')
                    : (language === 'zh' ? '请求补充分析' : 'Request More Analysis')}
                </button>
                <button
                  onClick={handleConfirmAll}
                  disabled={submitting || isSupplementaryRunning || (hasLowConfidence && !allLowConfidenceConfirmed)}
                  className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    submitting || isSupplementaryRunning || (hasLowConfidence && !allLowConfidenceConfirmed)
                      ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                      : 'bg-accent-600 text-white hover:bg-accent-700 shadow-md'
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
      {showSupplementaryInput && (
        <SupplementaryAnalysisModal
          language={language}
          placeholder={language === 'zh'
            ? '例如：请进一步分析cluster 3和cluster 5的差异基因，并搜索相关文献'
            : 'e.g., Please analyze DEGs between cluster 3 and 5, and search relevant literature'}
          onSubmit={onRequestSupplementary}
          onClose={() => setShowSupplementaryInput(false)}
        />
      )}
    </div>
  )
}
