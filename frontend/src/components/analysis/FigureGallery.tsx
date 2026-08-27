import { ChevronDown, ChevronRight, Download, FileImage, X } from 'lucide-react'

import { FigureCard, getStatusColor, getStatusIcon, getStatusLabel } from './FigureCard'
import { useFigureGallery, type FigureGalleryProps } from './useFigureGallery'

const API_BASE = '/api'

export function FigureGallery(props: FigureGalleryProps) {
  const controller = useFigureGallery(props)
  const { figures } = props
  if (figures.length === 0) return null
  const {
    collapsed, currentSession, downloadJPG, downloadPDF, hiddenCount, language,
    selectedFigure, setCollapsed, setSelectedFigure, setShowAll, showAll, t, visibleFigures,
  } = controller
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

        {visibleFigures.map(figure => (
          <FigureCard
            key={figure.id}
            currentSessionId={currentSession?.id}
            downloadJPG={downloadJPG}
            downloadPDF={downloadPDF}
            figure={figure}
            language={language}
            onSelect={setSelectedFigure}
            t={t}
          />
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
                  <span className="ml-1">{getStatusLabel(selectedFigure.reviewStatus, selectedFigure.diagnostic, language, t)}</span>
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
