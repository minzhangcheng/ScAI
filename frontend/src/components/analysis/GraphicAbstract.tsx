import { useState } from 'react'
import { useAppStore, GraphicAbstract as GraphicAbstractType } from '../../stores/appStore'
import { useTranslation } from '../../i18n/translations'
import {
  Download,
  ZoomIn,
  X,
  Sparkles
} from 'lucide-react'
import { base64ToDataUri, detectBase64MimeType } from '../../utils/image'

interface GraphicAbstractProps {
  graphicAbstract: GraphicAbstractType
}

export function GraphicAbstract({ graphicAbstract }: GraphicAbstractProps) {
  const { language } = useAppStore()
  const { t } = useTranslation(language)
  const [showModal, setShowModal] = useState(false)

  if (!graphicAbstract?.imageBase64) {
    return null
  }

  const downloadImage = () => {
    const link = document.createElement('a')
    const mimeType = detectBase64MimeType(graphicAbstract.imageBase64)
    const ext = mimeType === 'image/jpeg' ? 'jpg' : mimeType.split('/')[1]
    link.href = base64ToDataUri(graphicAbstract.imageBase64)
    link.download = `graphic_abstract.${ext}`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="mt-6">
      {/* Header */}
      <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wide mb-3 flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-amber-500" />
        {t('graphicAbstract')}
      </h3>

      {/* Card */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        {/* Image Preview */}
        <div
          className="relative bg-slate-50 cursor-pointer group"
          onClick={() => setShowModal(true)}
        >
          <img
            src={base64ToDataUri(graphicAbstract.imageBase64)}
            alt="Graphic Abstract"
            className="w-full h-auto max-h-96 object-contain"
          />

          {/* Zoom overlay */}
          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
            <div className="flex items-center gap-2 text-white bg-black/50 px-4 py-2 rounded-full">
              <ZoomIn className="w-5 h-5" />
              <span className="text-sm font-medium">
                {language === 'zh' ? '点击放大' : 'Click to enlarge'}
              </span>
            </div>
          </div>
        </div>

        {/* Info & Actions */}
        <div className="p-4 border-t border-slate-200">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <p className="text-sm text-slate-600">
                <span className="font-medium text-slate-700">
                  {language === 'zh' ? '生成提示: ' : 'Prompt: '}
                </span>
                {graphicAbstract.prompt}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                {language === 'zh' ? '生成于 ' : 'Generated at '}
                {new Date(graphicAbstract.createdAt).toLocaleString()}
              </p>
            </div>

            <button
              onClick={downloadImage}
              className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-lg transition-colors shrink-0"
            >
              <Download className="w-4 h-4" />
              {language === 'zh' ? '下载' : 'Download'}
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

            {/* Image */}
            <img
              src={base64ToDataUri(graphicAbstract.imageBase64)}
              alt="Graphic Abstract"
              className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl"
            />

            {/* Bottom bar */}
            <div className="absolute -bottom-14 left-0 right-0 flex items-center justify-center gap-4">
              <button
                onClick={downloadImage}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" />
                {language === 'zh' ? '下载高清图' : 'Download HD'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
