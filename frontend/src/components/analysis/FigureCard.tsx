import {
  AlertCircle, AlertTriangle, CheckCircle2, Clock, Download, FileImage, ZoomIn,
} from 'lucide-react'

import type { FigureReviewStatus, PublicationFigure } from '../../stores/appStore'
import { normalizeImageBase64 } from '../../utils/image'

const API_BASE = '/api'

export function getStatusIcon(status: FigureReviewStatus, diagnostic?: boolean, reviewUnavailable?: boolean) {
    if (reviewUnavailable && status === 'pending') {
      return <AlertTriangle className="w-4 h-4 text-slate-500" />
    }
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

export function getStatusLabel(status: FigureReviewStatus, diagnostic: boolean | undefined, language: string, t: (key: any) => string, reviewUnavailable?: boolean) {
    if (reviewUnavailable && status === 'pending') {
      return t('figureReviewUnavailable')
    }
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

export function getStatusColor(status: FigureReviewStatus, diagnostic?: boolean, reviewUnavailable?: boolean) {
    if (reviewUnavailable && status === 'pending') {
      return 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700'
    }
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

interface FigureCardProps {
  branchQuery?: string
  currentSessionId?: string
  downloadJPG: (figure: PublicationFigure) => Promise<void>
  downloadPDF: (figure: PublicationFigure) => Promise<void>
  figure: PublicationFigure
  language: string
  onSelect: (figure: PublicationFigure) => void
  t: (key: any) => string
}

export function FigureCard({
  branchQuery = '', currentSessionId, downloadJPG, downloadPDF, figure, language, onSelect, t,
}: FigureCardProps) {
  const pngBase64 = normalizeImageBase64(figure.pngBase64)
  return (
          <div
            key={figure.id}
            className="bg-white dark:bg-surface-900 border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden hover:shadow-md transition-shadow"
          >
            {/* Image Preview */}
            <div
              className="publication-canvas relative aspect-square cursor-pointer group"
              onClick={() => onSelect(figure)}
            >
              {pngBase64 ? (
                <img
                  src={`data:image/jpeg;base64,${pngBase64}`}
                  alt={figure.name}
                  className="w-full h-full object-contain"
                />
              ) : currentSessionId ? (
                <img
                  key={`api-${figure.id}-${figure.reviewStatus}-${figure.revisionCount}`}
                  src={`${API_BASE}/session/${currentSessionId}/figure/${figure.id}/png${branchQuery}`}
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
              {!pngBase64 && (
                <div className="w-full h-full items-center justify-center text-slate-400" style={{ display: 'none' }}>
                  <FileImage className="w-12 h-12" />
                </div>
              )}

              {/* Zoom overlay */}
              <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <ZoomIn className="w-8 h-8 text-white" />
              </div>

              {/* Status Badge */}
              <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(figure.reviewStatus, figure.diagnostic, figure.reviewUnavailable)}`}>
                <span className="flex items-center gap-1">
                  {getStatusIcon(figure.reviewStatus, figure.diagnostic, figure.reviewUnavailable)}
                  {getStatusLabel(figure.reviewStatus, figure.diagnostic, language, t, figure.reviewUnavailable)}
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

  )
}
