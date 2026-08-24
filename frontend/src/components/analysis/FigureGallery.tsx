import { useState } from 'react'
import { useAppStore, PublicationFigure, FigureReviewStatus } from '../../stores/appStore'
import { useTranslation } from '../../i18n/translations'
import { fetchAndDownload } from '../../utils/download'
import {
  CheckCircle2,
  Clock,
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Download,
  X,
  ZoomIn,
  FileImage
} from 'lucide-react'

const API_BASE = '/api'

// How many figures the gallery shows before "show all". A completed run can
// produce dozens; leading with a readable handful keeps the page browsable.
const PREVIEW_COUNT = 8

interface FigureGalleryProps {
  figures: PublicationFigure[]
  /** How many leading entries the PI curated for the brief report, if any. */
  curatedCount?: number
}

export function FigureGallery({ figures, curatedCount }: FigureGalleryProps) {
  const { currentSession, language } = useAppStore()
  const { t } = useTranslation(language)
  const [selectedFigure, setSelectedFigure] = useState<PublicationFigure | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  if (figures.length === 0) {
    return null
  }

  const previewCount = Math.max(PREVIEW_COUNT, curatedCount ?? 0)
  const visibleFigures = showAll ? figures : figures.slice(0, previewCount)
  const hiddenCount = figures.length - visibleFigures.length

  const getStatusIcon = (status: FigureReviewStatus, diagnostic?: boolean) => {
    if (diagnostic && status === 'pending') {
      return <FileImage className="w-4 h-4 text-sky-600" />
    }
    switch (status) {
      case 'approved':
        return <CheckCircle2 className="w-4 h-4 text-green-500" />
      case 'accepted_with_reservations':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />
      case 'pending':
        return <Clock className="w-4 h-4 text-amber-500" />
      case 'revision_requested':
        return <AlertCircle className="w-4 h-4 text-red-500" />
    }
  }

  const getStatusLabel = (status: FigureReviewStatus, diagnostic?: boolean) => {
    if (diagnostic && status === 'pending') {
      return language === 'zh' ? '平台图' : 'Platform figure'
    }
    switch (status) {
      case 'approved':
        return t('figureApproved')
      case 'accepted_with_reservations':
        return language === 'zh' ? '有保留通过' : 'Accepted with reservations'
      case 'pending':
        return t('figurePending')
      case 'revision_requested':
        return t('figureRevisionRequested')
    }
  }

  const getStatusColor = (status: FigureReviewStatus, diagnostic?: boolean) => {
    if (diagnostic && status === 'pending') {
      return 'bg-sky-100 dark:bg-sky-900/60 text-sky-700 dark:text-sky-300 border-sky-200 dark:border-sky-800'
    }
    switch (status) {
      case 'approved':
        return 'bg-green-100 dark:bg-green-900/60 text-green-700 dark:text-green-300 border-green-200 dark:border-green-800'
      case 'accepted_with_reservations':
        return 'bg-yellow-100 dark:bg-yellow-900/60 text-yellow-700 dark:text-yellow-300 border-yellow-200 dark:border-yellow-800'
      case 'pending':
        return 'bg-amber-100 dark:bg-amber-900/60 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800'
      case 'revision_requested':
        return 'bg-red-100 dark:bg-red-900/60 text-red-700 dark:text-red-300 border-red-200 dark:border-red-800'
    }
  }

  const downloadPDF = async (figure: PublicationFigure) => {
    if (!currentSession) return
    try {
      await fetchAndDownload(
        `${API_BASE}/session/${currentSession.id}/figure/${figure.id}/pdf`,
        `${figure.name}.pdf`
      )
    } catch (e) {
      console.error('Failed to download PDF:', e)
    }
  }

  const downloadJPG = async (figure: PublicationFigure) => {
    if (!currentSession) return
    try {
      await fetchAndDownload(
        `${API_BASE}/session/${currentSession.id}/figure/${figure.id}/png`,
        `${figure.name}.jpg`
      )
    } catch (e) {
      console.error('Failed to download JPG:', e)
    }
  }

  return (
    <div className="mt-6">
      <button
        onClick={() => setCollapsed(c => !c)}
        className="w-full text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
      >
        {collapsed ? (
          <ChevronRight className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
        <FileImage className="w-4 h-4" />
        {t('figureGallery')} ({figures.length})
        {!collapsed && hiddenCount > 0 && (
          <span className="normal-case tracking-normal text-slate-400">
            {language === 'zh'
              ? `· 显示 ${visibleFigures.length} / ${figures.length}`
              : `· showing ${visibleFigures.length} of ${figures.length}`}
          </span>
        )}
      </button>

      {!collapsed && (
      <>
      {/* Figure Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {visibleFigures.map((figure) => (
          <div
            key={figure.id}
            className="bg-white dark:bg-surface-900 border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden hover:shadow-md transition-shadow"
          >
            {/* Image Preview */}
            <div
              className="publication-canvas relative aspect-square cursor-pointer group"
              onClick={() => setSelectedFigure(figure)}
            >
              {figure.pngBase64 ? (
                <img
                  src={`data:image/jpeg;base64,${figure.pngBase64}`}
                  alt={figure.name}
                  className="w-full h-full object-contain"
                />
              ) : currentSession?.id ? (
                <img
                  key={`api-${figure.id}-${figure.reviewStatus}-${figure.revisionCount}`}
                  src={`${API_BASE}/session/${currentSession.id}/figure/${figure.id}/png`}
                  alt={figure.name}
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    const target = e.currentTarget
                    target.style.display = 'none'
                    if (target.nextElementSibling) {
                      ;(target.nextElementSibling as HTMLElement).style.display = 'flex'
                    }
                  }}
                />
              ) : null}
              {/* Placeholder shown on API load error */}
              {!figure.pngBase64 && (
                <div className="w-full h-full items-center justify-center text-slate-400" style={{ display: 'none' }}>
                  <FileImage className="w-12 h-12" />
                </div>
              )}

              {/* Zoom overlay */}
              <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <ZoomIn className="w-8 h-8 text-white" />
              </div>

              {/* Status Badge */}
              <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(figure.reviewStatus, figure.diagnostic)}`}>
                <span className="flex items-center gap-1">
                  {getStatusIcon(figure.reviewStatus, figure.diagnostic)}
                  {getStatusLabel(figure.reviewStatus, figure.diagnostic)}
                </span>
              </div>

              {/* Revision Count */}
              {figure.revisionCount > 0 && (
                <div className="absolute top-2 left-2 px-2 py-1 bg-slate-800/70 text-white rounded-full text-xs">
                  v{figure.revisionCount + 1}
                </div>
              )}
            </div>

            {/* Info */}
            <div className="p-3">
              <h4 className="font-medium text-slate-900 dark:text-slate-100 text-sm truncate" title={figure.name}>
                {figure.name}
              </h4>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2" title={figure.description}>
                {figure.description}
              </p>
              <div className="flex flex-wrap items-center gap-1 mt-2">
                <span className="text-xs text-slate-400 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">
                  {figure.figureType}
                </span>
                {figure.round ? (
                  <span className="text-xs text-slate-400 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">
                    {language === 'zh' ? `第 ${figure.round} 轮` : `Round ${figure.round}`}
                  </span>
                ) : null}
                {figure.diagnostic && (
                  <span className="text-xs text-slate-500 dark:text-slate-400 bg-slate-200 dark:bg-slate-700 px-2 py-0.5 rounded">
                    {language === 'zh' ? '诊断图' : 'Diagnostic'}
                  </span>
                )}
                {figure.planStepId !== undefined && (
                  <span
                    className="text-xs text-violet-600 dark:text-violet-400 bg-violet-50 dark:bg-violet-950/40 px-2 py-0.5 rounded"
                    title={figure.planId
                      ? `${language === 'zh' ? '计划' : 'Plan'} ${figure.planId}`
                      : undefined}
                  >
                    {language === 'zh'
                      ? `计划步骤 ${figure.planStepId}`
                      : `Plan step ${figure.planStepId}`}
                  </span>
                )}
                {figure.purpose && figure.purpose !== 'result' && (
                  <span className="text-xs text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">
                    {figure.purpose}
                  </span>
                )}
                {/* The status badge alone would read "Final" over a figure the
                    revision loop never made legible. The renderer's measurement
                    travels with it. */}
                {figure.layoutBlocking && (
                  <span
                    className="text-xs text-amber-700 dark:text-amber-300 bg-amber-100 dark:bg-amber-900/60 px-2 py-0.5 rounded"
                    title={language === 'zh' ? '渲染引擎测得文字被遮挡' : 'The renderer measured overlapping text'}
                  >
                    {language === 'zh' ? '排版待修' : 'Layout defect'}
                  </span>
                )}
              </div>

              {/* Download Buttons */}
              <div className="flex items-center gap-2 mt-3">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    downloadJPG(figure)
                  }}
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded transition-colors"
                >
                  <Download className="w-3 h-3" />
                  JPG
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    downloadPDF(figure)
                  }}
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-violet-100 dark:bg-violet-900/60 hover:bg-violet-200 dark:hover:bg-violet-800/70 text-violet-700 dark:text-violet-300 rounded transition-colors"
                >
                  <Download className="w-3 h-3" />
                  PDF
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {(hiddenCount > 0 || showAll) && (
        <div className="mt-3 flex justify-center">
          <button
            onClick={() => setShowAll(s => !s)}
            className="flex items-center gap-1 px-4 py-1.5 text-sm text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors"
          >
            {showAll ? (
              <>
                <ChevronDown className="w-4 h-4 rotate-180" />
                {language === 'zh' ? '收起' : 'Show less'}
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                {language === 'zh' ? `显示全部 (${figures.length})` : `Show all (${figures.length})`}
              </>
            )}
          </button>
        </div>
      )}
      </>
      )}

      {/* Lightbox Modal */}
      {selectedFigure && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setSelectedFigure(null)}
        >
          <div
            className="bg-white dark:bg-surface-900 rounded-xl max-w-4xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
              <div>
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">{selectedFigure.name}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">{selectedFigure.description}</p>
              </div>
              <button
                onClick={() => setSelectedFigure(null)}
                className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="publication-canvas p-4 max-h-[60vh] overflow-auto">
              {selectedFigure.pngBase64 ? (
                <img
                  src={`data:image/jpeg;base64,${selectedFigure.pngBase64}`}
                  alt={selectedFigure.name}
                  className="max-w-full h-auto mx-auto"
                />
              ) : currentSession?.id ? (
                <img
                  src={`${API_BASE}/session/${currentSession.id}/figure/${selectedFigure.id}/png`}
                  alt={selectedFigure.name}
                  className="max-w-full h-auto mx-auto"
                  onError={(e) => {
                    const target = e.currentTarget
                    target.style.display = 'none'
                    if (target.nextElementSibling) {
                      ;(target.nextElementSibling as HTMLElement).style.display = 'flex'
                    }
                  }}
                />
              ) : null}
              {/* Placeholder shown on API load error */}
              {!selectedFigure.pngBase64 && (
                <div className="items-center justify-center h-64 text-slate-400" style={{ display: 'none' }}>
                  <FileImage className="w-16 h-16" />
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-slate-200 dark:border-slate-700 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getStatusColor(selectedFigure.reviewStatus, selectedFigure.diagnostic)}`}>
                  {getStatusIcon(selectedFigure.reviewStatus, selectedFigure.diagnostic)}
                  <span className="ml-1">{getStatusLabel(selectedFigure.reviewStatus, selectedFigure.diagnostic)}</span>
                </span>
                {selectedFigure.piFeedback && (
                  <span className="text-sm text-slate-500 dark:text-slate-400 italic">
                    {language === 'zh' ? 'PI反馈: ' : 'PI Feedback: '}
                    {selectedFigure.piFeedback}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => downloadJPG(selectedFigure)}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-lg transition-colors"
                >
                  <Download className="w-4 h-4" />
                  JPG
                </button>
                <button
                  onClick={() => downloadPDF(selectedFigure)}
                  className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-lg transition-colors"
                >
                  <Download className="w-4 h-4" />
                  {t('downloadPDF')}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
