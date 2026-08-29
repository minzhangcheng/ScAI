import { useState } from 'react'
import { useAppStore, GraphicAbstract as GraphicAbstractType } from '../../stores/appStore'
import { useTranslation } from '../../i18n/translations'
import {
  Download,
  ZoomIn,
  X,
  Sparkles
} from 'lucide-react'
import { base64ToDataUri, detectBase64MimeType, normalizeImageBase64 } from '../../utils/image'
import { downloadBlob, fetchAndDownload } from '../../utils/download'
import { withReportBranch } from '../../utils/reportBranch'

interface GraphicAbstractProps {
  branchId?: string | null
  graphicAbstract: GraphicAbstractType
  sessionId?: string
}

export function GraphicAbstract({ branchId, graphicAbstract, sessionId }: GraphicAbstractProps) {
  const { language } = useAppStore()
  const { t } = useTranslation(language)
  const [showModal, setShowModal] = useState(false)

  const isTextMode = !!graphicAbstract?.isTextMode || !!graphicAbstract?.textDiagram
  const imageB64 = normalizeImageBase64(graphicAbstract?.imageBase64)
  const imageUrl = sessionId
    ? withReportBranch(`/api/session/${sessionId}/report/figure/graphic_abstract.jpg`, branchId)
    : ''
  const imageSource = imageB64 ? base64ToDataUri(imageB64) : imageUrl
  const textDiagram = graphicAbstract?.textDiagram || ''

  if (!imageSource && !textDiagram) {
    return null
  }

  const downloadImage = async () => {
    if (!imageB64) {
      if (imageUrl) {
        try {
          await fetchAndDownload(imageUrl, 'graphic_abstract.jpg')
        } catch (error) {
          console.error('Failed to download graphic abstract:', error)
        }
      }
      return
    }
    const mimeType = detectBase64MimeType(imageB64)
    const ext = mimeType === 'image/jpeg' ? 'jpg' : (mimeType.split('/')[1] ?? 'png')
    const byteString = atob(imageB64)
    const bytes = new Uint8Array(byteString.length)
    for (let i = 0; i < byteString.length; i++) bytes[i] = byteString.charCodeAt(i)
    downloadBlob(new Blob([bytes], { type: mimeType }), `graphic_abstract.${ext}`)
  }

  const downloadTextDiagram = () => {
    if (!textDiagram) return
    downloadBlob(new Blob([textDiagram], { type: 'text/plain;charset=utf-8' }), 'graphic_abstract.txt')
  }

  return (
    <div className="mt-6">
      {/* Header */}
      <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-amber-500" />
        {isTextMode ? t('graphicAbstractTextMode') : t('graphicAbstract')}
      </h3>

      {/* Card */}
      <div className="bg-white dark:bg-surface-900 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
        {/* Preview */}
        {isTextMode ? (
          <div
            className="relative bg-slate-50 dark:bg-slate-800 cursor-pointer group p-4"
            onClick={() => setShowModal(true)}
          >
            <pre className="font-mono text-xs text-slate-800 dark:text-slate-100 whitespace-pre overflow-auto max-h-96 leading-5">
              {textDiagram}
            </pre>
            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
              <div className="flex items-center gap-2 text-white bg-black/50 px-4 py-2 rounded-full">
                <ZoomIn className="w-5 h-5" />
                <span className="text-sm font-medium">
                  {t('clickToEnlarge')}
                </span>
              </div>
            </div>
          </div>
        ) : (
        <div
          className="publication-canvas relative cursor-pointer group"
          onClick={() => setShowModal(true)}
        >
          <img
            src={imageSource}
            alt="Graphic Abstract"
            className="w-full h-auto max-h-96 object-contain"
          />

          {/* Zoom overlay */}
          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
            <div className="flex items-center gap-2 text-white bg-black/50 px-4 py-2 rounded-full">
              <ZoomIn className="w-5 h-5" />
              <span className="text-sm font-medium">
                {t('clickToEnlarge')}
              </span>
            </div>
          </div>
        </div>
        )}

        {/* Info & Actions */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-700">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <p className="text-sm text-slate-600 dark:text-slate-300">
                <span className="font-medium text-slate-700 dark:text-slate-200">
                  {t('graphicAbstractPrompt')}
                </span>
                {graphicAbstract.prompt}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                {t('generatedAt')}
                {new Date(graphicAbstract.createdAt).toLocaleString()}
              </p>
            </div>

            <button
              onClick={isTextMode ? downloadTextDiagram : () => void downloadImage()}
              className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-lg transition-colors shrink-0"
            >
              <Download className="w-4 h-4" />
              {isTextMode ? t('downloadText') : t('downloadHD')}
            </button>
          </div>
        </div>
      </div>

      {/* Lightbox Modal */}
      {showModal && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setShowModal(false)}
        >
          <div
            className="relative max-w-5xl max-h-[90vh]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={() => setShowModal(false)}
              className="absolute -top-12 right-0 p-2 text-white/70 hover:text-white transition-colors"
            >
              <X className="w-8 h-8" />
            </button>

            {/* Content */}
            {isTextMode ? (
              <pre className="font-mono text-sm text-slate-100 bg-slate-900 rounded-lg shadow-2xl p-6 max-w-full max-h-[85vh] overflow-auto whitespace-pre leading-6">
                {textDiagram}
              </pre>
            ) : (
              <img
                src={imageSource}
                alt="Graphic Abstract"
                className="publication-canvas max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl"
              />
            )}

            {/* Bottom bar */}
            <div className="absolute -bottom-14 left-0 right-0 flex items-center justify-center gap-4">
              <button
                onClick={isTextMode ? downloadTextDiagram : () => void downloadImage()}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" />
                {isTextMode
                  ? t('downloadText')
                  : t('downloadHD')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
