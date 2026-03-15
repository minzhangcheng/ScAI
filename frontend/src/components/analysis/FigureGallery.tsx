import { useState } from 'react'
import { useAppStore, PublicationFigure, FigureReviewStatus } from '../../stores/appStore'
import { useTranslation } from '../../i18n/translations'
import {
  CheckCircle2,
  Clock,
  AlertCircle,
  AlertTriangle,
  Download,
  X,
  ZoomIn,
  FileImage
} from 'lucide-react'

const API_BASE = '/api'

interface FigureGalleryProps {
  figures: PublicationFigure[]
}

export function FigureGallery({ figures }: FigureGalleryProps) {
  const { currentSession, language } = useAppStore()
  const { t } = useTranslation(language)
  const [selectedFigure, setSelectedFigure] = useState<PublicationFigure | null>(null)

  if (figures.length === 0) {
    return null
  }

  const isSessionCompleted = currentSession?.status === 'completed'

  const getStatusIcon = (status: FigureReviewStatus) => {
    // When session is completed, revision_requested figures are the final accepted version
    if (isSessionCompleted && status === 'revision_requested') {
      return <CheckCircle2 className="w-4 h-4 text-green-500" />
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

  const getStatusLabel = (status: FigureReviewStatus) => {
    if (isSessionCompleted && status === 'revision_requested') {
      return language === 'zh' ? '最终版' : 'Final'
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

  const getStatusColor = (status: FigureReviewStatus) => {
    if (isSessionCompleted && status === 'revision_requested') {
      return 'bg-green-100 text-green-700 border-green-200'
    }
    switch (status) {
      case 'approved':
        return 'bg-green-100 text-green-700 border-green-200'
      case 'accepted_with_reservations':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200'
      case 'pending':
        return 'bg-amber-100 text-amber-700 border-amber-200'
      case 'revision_requested':
        return 'bg-red-100 text-red-700 border-red-200'
    }
  }

  const downloadPDF = async (figure: PublicationFigure) => {
    if (!currentSession) return

    try {
      const response = await fetch(
        `${API_BASE}/session/${currentSession.id}/figure/${figure.id}/pdf`
      )
      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${figure.name}.pdf`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }
    } catch (e) {
      console.error('Failed to download PDF:', e)
    }
  }

  const downloadPNG = async (figure: PublicationFigure) => {
    if (!currentSession) return

    try {
      const response = await fetch(
        `${API_BASE}/session/${currentSession.id}/figure/${figure.id}/png`
      )
      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${figure.name}.png`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }
    } catch (e) {
      console.error('Failed to download PNG:', e)
    }
  }

  return (
    <div className="mt-6">
      <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wide mb-3 flex items-center gap-2">
        <FileImage className="w-4 h-4" />
        {t('figureGallery')} ({figures.length})
      </h3>

      {/* Figure Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {figures.map((figure) => (
          <div
            key={figure.id}
            className="bg-white border border-slate-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow"
          >
            {/* Image Preview */}
            <div
              className="relative aspect-square bg-slate-100 cursor-pointer group"
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
              <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(figure.reviewStatus)}`}>
                <span className="flex items-center gap-1">
                  {getStatusIcon(figure.reviewStatus)}
                  {getStatusLabel(figure.reviewStatus)}
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
              <h4 className="font-medium text-slate-900 text-sm truncate" title={figure.name}>
                {figure.name}
              </h4>
              <p className="text-xs text-slate-500 mt-1 line-clamp-2" title={figure.description}>
                {figure.description}
              </p>
              <div className="flex items-center gap-1 mt-2">
                <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                  {figure.figureType}
                </span>
              </div>

              {/* Download Buttons */}
              <div className="flex items-center gap-2 mt-3">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    downloadPNG(figure)
                  }}
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 rounded transition-colors"
                >
                  <Download className="w-3 h-3" />
                  PNG
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    downloadPDF(figure)
                  }}
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-violet-100 hover:bg-violet-200 text-violet-700 rounded transition-colors"
                >
                  <Download className="w-3 h-3" />
                  PDF
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox Modal */}
      {selectedFigure && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setSelectedFigure(null)}
        >
          <div
            className="bg-white rounded-xl max-w-4xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-200">
              <div>
                <h3 className="font-semibold text-slate-900">{selectedFigure.name}</h3>
                <p className="text-sm text-slate-500">{selectedFigure.description}</p>
              </div>
              <button
                onClick={() => setSelectedFigure(null)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-4 max-h-[60vh] overflow-auto bg-slate-50">
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
            <div className="p-4 border-t border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getStatusColor(selectedFigure.reviewStatus)}`}>
                  {getStatusIcon(selectedFigure.reviewStatus)}
                  <span className="ml-1">{getStatusLabel(selectedFigure.reviewStatus)}</span>
                </span>
                {selectedFigure.piFeedback && (
                  <span className="text-sm text-slate-500 italic">
                    {language === 'zh' ? 'PI反馈: ' : 'PI Feedback: '}
                    {selectedFigure.piFeedback}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => downloadPNG(selectedFigure)}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg transition-colors"
                >
                  <Download className="w-4 h-4" />
                  PNG
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
