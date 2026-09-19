import { useState } from 'react'
import { FileDown } from 'lucide-react'

import type { Language } from '../../i18n/translations'
import { fetchAndDownload } from '../../utils/download'

interface StageReviewReportDownloadProps {
  url?: string
  language: Language
  stage: 1 | 2
}

export function StageReviewReportDownload({
  url, language, stage,
}: StageReviewReportDownloadProps) {
  const [download, setDownload] = useState<{ url: string; pending: boolean; error: string }>({
    url: '', pending: false, error: '',
  })
  const kind = stage === 1 ? 'annotation' : 'exploration'
  const isLegacyReport = new RegExp(`/${kind}-report/html(?:[?#]|$)`).test(url ?? '')
  const isDownloading = download.url === url && download.pending
  const downloadError = download.url === url ? download.error : ''
  const downloadReport = async () => {
    if (!url || isDownloading) return
    const reportUrl = url
    setDownload({ url: reportUrl, pending: true, error: '' })
    try {
      await fetchAndDownload(
        reportUrl,
        isLegacyReport
          ? `legacy_${kind}_report.html`
          : `round_1_stage_${stage}_review_report.html`,
        { preferResponseFilename: true },
      )
      setDownload(current => current.url === reportUrl
        ? { url: reportUrl, pending: false, error: '' } : current)
    } catch {
      setDownload(current => current.url === reportUrl
        ? { url: reportUrl, pending: false, error: language === 'zh'
          ? '报告下载失败，请重试。' : 'Report download failed. Please retry.' } : current)
    }
  }
  const reportLabel = isLegacyReport
    ? (language === 'zh'
      ? `下载旧版${stage === 1 ? '注释' : '探索'}报告`
      : `Download Legacy ${stage === 1 ? 'Annotation' : 'Exploration'} Report`)
    : (language === 'zh' ? `下载 Stage ${stage} 审查报告` : `Download Stage ${stage} Review Report`)

  return (
    <div>
      {url ? (
        <button
          type="button"
          onClick={downloadReport}
          disabled={isDownloading}
          className="flex items-center gap-2 px-4 py-2 bg-accent-50 text-accent-700 hover:bg-accent-100 rounded-lg text-sm font-medium disabled:opacity-50"
        >
          <FileDown className="w-4 h-4" />
          {isDownloading ? (language === 'zh' ? '正在下载…' : 'Downloading…') : reportLabel}
        </button>
      ) : (
        <p className="text-sm text-slate-500">
          {language === 'zh'
            ? `Stage ${stage} 审查报告暂不可下载`
            : `Stage ${stage} review report is currently unavailable for download`}
        </p>
      )}
      {downloadError && <p role="alert" className="mt-1 text-sm text-red-600">{downloadError}</p>}
    </div>
  )
}
