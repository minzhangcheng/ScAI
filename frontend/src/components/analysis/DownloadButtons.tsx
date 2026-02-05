import { useState } from 'react'
import { useAppStore, PublicationFigure, PublicationTable } from '../../stores/appStore'
import { useTranslation } from '../../i18n/translations'
import {
  Download,
  Package,
  FileImage,
  Table2,
  Loader2,
  CheckCircle2
} from 'lucide-react'

const API_BASE = '/api'

interface DownloadButtonsProps {
  figures: PublicationFigure[]
  tables?: PublicationTable[]
}

export function DownloadButtons({ figures, tables = [] }: DownloadButtonsProps) {
  const { currentSession, language } = useAppStore()
  const { t } = useTranslation(language)
  const [downloading, setDownloading] = useState<string | null>(null)
  const [downloadComplete, setDownloadComplete] = useState<string | null>(null)

  if (figures.length === 0 && tables.length === 0) {
    return null
  }

  const showSuccess = (type: string) => {
    setDownloadComplete(type)
    setTimeout(() => setDownloadComplete(null), 2000)
  }

  const downloadAllFigures = async () => {
    if (!currentSession) return

    setDownloading('figures')
    try {
      const response = await fetch(
        `${API_BASE}/session/${currentSession.id}/report/download-all-figures`,
        { method: 'POST' }
      )

      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `figures_${currentSession.id}.zip`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        showSuccess('figures')
      }
    } catch (e) {
      console.error('Failed to download figures:', e)
    } finally {
      setDownloading(null)
    }
  }

  const downloadAllTables = async () => {
    if (!currentSession) return

    setDownloading('tables')
    try {
      const response = await fetch(
        `${API_BASE}/session/${currentSession.id}/report/download-all-tables`,
        { method: 'POST' }
      )

      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `tables_${currentSession.id}.zip`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        showSuccess('tables')
      }
    } catch (e) {
      console.error('Failed to download tables:', e)
    } finally {
      setDownloading(null)
    }
  }

  const downloadAll = async () => {
    if (!currentSession) return

    setDownloading('all')
    try {
      // Download figures
      const figuresResponse = await fetch(
        `${API_BASE}/session/${currentSession.id}/report/download-all-figures`,
        { method: 'POST' }
      )

      if (figuresResponse.ok) {
        const blob = await figuresResponse.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `figures_${currentSession.id}.zip`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }

      // Download tables if available
      if (tables.length > 0) {
        const tablesResponse = await fetch(
          `${API_BASE}/session/${currentSession.id}/report/download-all-tables`,
          { method: 'POST' }
        )

        if (tablesResponse.ok) {
          const blob = await tablesResponse.blob()
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `tables_${currentSession.id}.zip`
          document.body.appendChild(a)
          a.click()
          document.body.removeChild(a)
          URL.revokeObjectURL(url)
        }
      }

      showSuccess('all')
    } catch (e) {
      console.error('Failed to download all:', e)
    } finally {
      setDownloading(null)
    }
  }

  const getButtonContent = (type: string, label: string, icon: React.ReactNode) => {
    if (downloading === type) {
      return (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          {language === 'zh' ? '下载中...' : 'Downloading...'}
        </>
      )
    }
    if (downloadComplete === type) {
      return (
        <>
          <CheckCircle2 className="w-4 h-4 text-green-500" />
          {language === 'zh' ? '完成!' : 'Done!'}
        </>
      )
    }
    return (
      <>
        {icon}
        {label}
      </>
    )
  }

  return (
    <div className="mt-6">
      <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wide mb-3 flex items-center gap-2">
        <Download className="w-4 h-4" />
        {language === 'zh' ? '批量下载' : 'Batch Download'}
      </h3>

      <div className="flex flex-wrap gap-3">
        {/* Download All Figures */}
        {figures.length > 0 && (
          <button
            onClick={downloadAllFigures}
            disabled={downloading !== null}
            className="flex items-center gap-2 px-4 py-2.5 bg-white border border-slate-200 hover:bg-slate-50 disabled:opacity-50 text-slate-700 rounded-lg transition-colors"
          >
            {getButtonContent(
              'figures',
              `${t('downloadAllFigures')} (${figures.length})`,
              <FileImage className="w-4 h-4 text-violet-500" />
            )}
          </button>
        )}

        {/* Download All Tables */}
        {tables.length > 0 && (
          <button
            onClick={downloadAllTables}
            disabled={downloading !== null}
            className="flex items-center gap-2 px-4 py-2.5 bg-white border border-slate-200 hover:bg-slate-50 disabled:opacity-50 text-slate-700 rounded-lg transition-colors"
          >
            {getButtonContent(
              'tables',
              `${language === 'zh' ? '下载所有表格' : 'Download All Tables'} (${tables.length})`,
              <Table2 className="w-4 h-4 text-green-500" />
            )}
          </button>
        )}

        {/* Download Everything */}
        {(figures.length > 0 || tables.length > 0) && (
          <button
            onClick={downloadAll}
            disabled={downloading !== null}
            className="flex items-center gap-2 px-4 py-2.5 bg-violet-600 hover:bg-violet-700 disabled:opacity-50 text-white rounded-lg transition-colors"
          >
            {getButtonContent(
              'all',
              language === 'zh' ? '下载全部' : 'Download All',
              <Package className="w-4 h-4" />
            )}
          </button>
        )}
      </div>

      {/* Summary */}
      <p className="text-xs text-slate-400 mt-2">
        {language === 'zh'
          ? `共 ${figures.length} 张图片${tables.length > 0 ? `、${tables.length} 个表格` : ''}，PDF 格式适合发表使用`
          : `Total ${figures.length} figures${tables.length > 0 ? `, ${tables.length} tables` : ''}. PDF format suitable for publication.`}
      </p>
    </div>
  )
}
