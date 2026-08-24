import { useState } from 'react'
import { useAppStore, PublicationFigure, PublicationTable } from '../../stores/appStore'
import { useTranslation } from '../../i18n/translations'
import { fetchAndDownload, FetchError } from '../../utils/download'
import { useSelfClearingError } from '../../hooks/useSelfClearingError'
import { ErrorBanner } from '../shared/ErrorBanner'
import {
  Download,
  Package,
  FileImage,
  Table2,
  Loader2,
  CheckCircle2
} from 'lucide-react'

import { API_BASE } from '../../constants/api'

interface DownloadButtonsProps {
  figures: PublicationFigure[]
  tables?: PublicationTable[]
}

export function DownloadButtons({ figures, tables = [] }: DownloadButtonsProps) {
  const { currentSession, language } = useAppStore()
  const { t } = useTranslation(language)
  const [downloading, setDownloading] = useState<string | null>(null)
  const [downloadComplete, setDownloadComplete] = useState<string | null>(null)
  const [downloadError, setDownloadError] = useSelfClearingError()

  if (figures.length === 0 && tables.length === 0) {
    return null
  }

  const showSuccess = (type: string) => {
    setDownloadComplete(type)
    setTimeout(() => setDownloadComplete(null), 2000)
  }

  const downloadType = async (
    type: string,
    url: string,
    filename: string,
    options?: RequestInit,
  ) => {
    if (!currentSession) return
    setDownloadError(null)
    setDownloading(type)
    try {
      await fetchAndDownload(url, filename, options)
      showSuccess(type)
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : `Failed to download ${type}`)
    } finally {
      setDownloading(null)
    }
  }

  const downloadAllFigures = () => {
    if (!currentSession) return
    return downloadType(
      'figures',
      `${API_BASE}/session/${currentSession.id}/report/download-all-figures`,
      `figures_${currentSession.id}.zip`,
      { method: 'POST' },
    )
  }

  const downloadAllTables = () => {
    if (!currentSession) return
    return downloadType(
      'tables',
      `${API_BASE}/session/${currentSession.id}/report/download-all-tables`,
      `tables_${currentSession.id}.zip`,
      { method: 'POST' },
    )
  }

  const downloadAll = async () => {
    if (!currentSession) return
    setDownloadError(null)
    setDownloading('all')
    try {
      const downloads: Promise<void>[] = []
      if (figures.length > 0) {
        downloads.push(
          fetchAndDownload(
            `${API_BASE}/session/${currentSession.id}/report/download-all-figures`,
            `figures_${currentSession.id}.zip`,
            { method: 'POST' },
          ),
        )
      }
      // Always attempt the tables zip — it includes score_definitions.csv even
      // when there are no publication tables. Ignore 404 (no tables directory).
      downloads.push(
        fetchAndDownload(
          `${API_BASE}/session/${currentSession.id}/report/download-all-tables`,
          `tables_${currentSession.id}.zip`,
          { method: 'POST' },
        ).catch((e) => {
          if (e instanceof FetchError && e.status === 404) return
          throw e
        }),
      )
      await Promise.all(downloads)
      showSuccess('all')
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : 'Failed to download')
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
      <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
        <Download className="w-4 h-4" />
        {language === 'zh' ? '批量下载' : 'Batch Download'}
      </h3>

      <div className="flex flex-wrap gap-3">
        {/* Download All Figures */}
        {figures.length > 0 && (
          <button
            onClick={downloadAllFigures}
            disabled={downloading !== null}
            className="flex items-center gap-2 px-4 py-2.5 bg-white dark:bg-surface-900 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 text-slate-700 dark:text-slate-200 rounded-lg transition-colors"
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
            className="flex items-center gap-2 px-4 py-2.5 bg-white dark:bg-surface-900 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 text-slate-700 dark:text-slate-200 rounded-lg transition-colors"
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
      {downloadError && (
        <ErrorBanner message={downloadError} onDismiss={() => setDownloadError(null)} />
      )}
    </div>
  )
}
