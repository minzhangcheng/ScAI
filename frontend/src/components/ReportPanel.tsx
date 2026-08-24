import { useState, useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import { devLog } from '../utils/logger'
import { fetchAndDownload } from '../utils/download'
import { WS_CONNECT_TIMEOUT_MS, apiFetch, useConfig, useUserConfig } from '../hooks/useApi'
import { useSelfClearingError } from '../hooks/useSelfClearingError'
import { ErrorBanner } from './shared/ErrorBanner'
import { getToken } from '../utils/authStorage'
import { WS_AUTH_SUBPROTOCOL } from '../constants/auth'
import { ValidationDialog } from './ValidationDialog'
import {
  FileText,
  Image,
  Table,
  Loader2,
  AlertCircle,
  RefreshCw,
  ExternalLink,
  Package
} from 'lucide-react'

import { API_BASE } from '../constants/api'

type ReportDownloadEndpoint = 'download-all-figures' | 'download-all-tables' | 'download-full'

interface ReportFigure {
  id: string
  name: string
  png_path: string
  pdf_path: string | null
  section: string
  analysis_type: string
}

interface ReportTable {
  id: string
  name: string
  csv_path: string
  section: string
  columns: string[]
  preview_data: any[]
  analysis_type: string
}

interface ReportSection {
  id: string
  title: string
  description: string
  analysis_type: 'general' | 'specific'
  content: string
  figures: ReportFigure[]
  tables: ReportTable[]
}

interface ReportAppendix {
  project_name?: string
  project_description?: string
  created_at?: string
  agent_models?: Record<string, string>
  agent_temperatures?: Record<string, number>
  treatment_groups?: Array<{ name: string; description: string; sample_count: number }>
  data_files?: Array<{ filename: string; section: string; data_type: string }>
  human_messages?: string[]
}

export interface DetailedReport {
  id: string
  session_id: string
  title: string
  abstract: string
  sections: ReportSection[]
  conclusions: string
  figures: ReportFigure[]
  tables: ReportTable[]
  graphic_abstract_path?: string
  generated_at: string
  status: 'pending' | 'generating' | 'completed' | 'partial' | 'failed'
  progress: number
  current_step: string
  html_path?: string
  appendix?: ReportAppendix
  partial_reason?: 'degraded_fallback' | 'figure_coverage' | 'approved_partial' | string | null
  figureless_sections?: string[]
  unreported_figures?: string[]
}

interface ReportPreview {
  session_id: string
  html_url: string
  download_url: string
  report: DetailedReport | null
  html_content: string | null
  figures: { name: string; png_url: string; pdf_url: string | null }[]
  tables: { name: string; csv_url: string }[]
  has_html: boolean
}

export function reportCompletionContent(report: DetailedReport, language: string) {
  const isPartial = report.status === 'partial'
  return {
    status: report.status,
    message: language === 'zh'
      ? (isPartial ? '报告已生成（部分报告）' : '报告生成完成')
      : (isPartial ? 'Report generated (partial)' : 'Report generation completed'),
  }
}

export function partialReportNotice(
  report: DetailedReport,
  language: string,
): string | null {
  if (report.status !== 'partial') return null

  const figurelessSections = report.figureless_sections ?? []
  const unreportedFigures = report.unreported_figures ?? []
  const coverageDetails: string[] = []
  if (figurelessSections.length > 0) {
    coverageDetails.push(
      language === 'zh'
        ? `缺少结果图的章节：${figurelessSections.join(', ')}`
        : `sections without result figures: ${figurelessSections.join(', ')}`,
    )
  }
  if (unreportedFigures.length > 0) {
    coverageDetails.push(
      language === 'zh'
        ? `未纳入报告的图片：${unreportedFigures.join(', ')}`
        : `figures not included in the report: ${unreportedFigures.join(', ')}`,
    )
  }

  if (report.partial_reason === 'degraded_fallback') {
    const base = language === 'zh'
      ? '报告模型失败后已生成降级版报告。'
      : 'A degraded fallback report was generated after the report model failed.'
    return coverageDetails.length > 0
      ? `${base} ${coverageDetails.join('；')}`
      : base
  }

  if (
    report.partial_reason === 'figure_coverage'
    || coverageDetails.length > 0
  ) {
    const base = language === 'zh'
      ? '报告已生成，但图片覆盖不完整。'
      : 'The report was generated with incomplete figure coverage.'
    const details = coverageDetails.length > 0
      ? coverageDetails.join('；')
      : report.current_step
    return details ? `${base} ${details}` : base
  }

  if (report.partial_reason === 'approved_partial') {
    return language === 'zh'
      ? '报告已按批准的部分结果范围生成。'
      : 'The report was generated for the approved partial result scope.'
  }

  return language === 'zh'
    ? '报告已生成，但被标记为部分报告。'
    : 'The report was generated but is marked as partial.'
}

export function reportBranchQuery(branchId?: string): string {
  return branchId ? `?branch_id=${encodeURIComponent(branchId)}` : ''
}

export function ReportPanel() {
  const { currentSession, language, addMessage, setReportGenerated, authUser, showConfigPanel } = useAppStore()
  const { t } = useTranslation(language)
  const adminConfig = useConfig()
  const userConfig = useUserConfig()

  const [report, setReport] = useState<DetailedReport | null>(null)
  const [preview, setPreview] = useState<ReportPreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reportArchives, setReportArchives] = useState<Array<{version: number; filename: string; created_at: string; size_bytes: number; download_url: string}>>([])
  const [archiveWarning, setArchiveWarning] = useState<string | null>(null)
  const [downloadError, setDownloadError] = useSelfClearingError()
  const wsRef = useRef<WebSocket | null>(null)
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const branchQuery = reportBranchQuery(currentSession?.restoreBranchId)

  // API validation state
  const [validating, setValidating] = useState(false)
  const [showValidationDialog, setShowValidationDialog] = useState(false)
  const [validationMessage, setValidationMessage] = useState('')
  const [validationUncertain, setValidationUncertain] = useState(false)

  // Load existing report on mount and when analysis completes
  useEffect(() => {
    if (currentSession?.id && currentSession?.status === 'completed') {
      loadReport()
    }
    return () => {
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current)
      }
    }
  }, [currentSession?.id, currentSession?.status, currentSession?.restoreBranchId])

  // Retry loading if report is null after initial load (handles race condition)
  useEffect(() => {
    if (currentSession?.status === 'completed' && !report && !loading && !generating) {
      retryTimerRef.current = setTimeout(() => {
        devLog('ReportPanel: Retrying report load...')
        loadReport()
      }, 3000)
      return () => {
        if (retryTimerRef.current) clearTimeout(retryTimerRef.current)
      }
    }
  }, [currentSession?.status, report, loading, generating])

  // WebSocket for real-time progress
  useEffect(() => {
    if (!generating || !currentSession?.id) return

    let isMounted = true
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/report/${currentSession.id}`
    const token = getToken()
    const wsProtocols = token ? [WS_AUTH_SUBPROTOCOL, token] : undefined
    const websocket = new WebSocket(wsUrl, wsProtocols)

    // Connection timeout
    const connectTimeout = setTimeout(() => {
      if (websocket.readyState !== WebSocket.OPEN) {
        websocket.close()
        if (isMounted) {
          setError('WebSocket connection timeout')
          setGenerating(false)
        }
      }
    }, WS_CONNECT_TIMEOUT_MS)

    websocket.onmessage = (event) => {
      if (!isMounted) return
      let data
      try {
        data = JSON.parse(event.data)
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
        return
      }
      if (data.type === 'progress') {
        setReport(prev => {
          if (prev && prev.progress === data.progress && prev.current_step === data.step) return prev
          return prev ? { ...prev, progress: data.progress, current_step: data.step, status: data.status } : null
        })
        // Also add to message stream (object format so MessageStream can detect progress=100 for checkmark icon)
        addMessage({
          id: `report-progress-${Date.now()}`,
          type: 'progress',
          agent: 'pi',
          content: {
            step: `[Report] ${data.step}`,
            progress: data.progress,
            status: data.status
          },
          timestamp: new Date().toISOString()
        })
      } else if (data.type === 'complete') {
        const completedReport = data.report as DetailedReport
        setReport(completedReport)
        setGenerating(false)
        setReportGenerated(true)
        // Load preview after generation
        Promise.all([loadPreview(), fetchArchives()])
        // Add completion message
        const completion = reportCompletionContent(completedReport, language)
        addMessage({
          id: `report-complete-${Date.now()}`,
          type: 'completed',
          agent: 'pi',
          content: completion,
          timestamp: new Date().toISOString()
        })
      } else if (data.type === 'error') {
        setError(data.message)
        setGenerating(false)
        // Add error message
        addMessage({
          id: `report-error-${Date.now()}`,
          type: 'error',
          agent: 'pi',
          content: `[Report Error] ${data.message}`,
          timestamp: new Date().toISOString()
        })
      } else if (data.type === 'archive_failed') {
        console.warn('Report archive failed:', data.message)
        setArchiveWarning(data.message || 'Failed to archive previous report')
      }
    }

    websocket.onopen = () => {
      clearTimeout(connectTimeout)
      if (!isMounted) return
      // Connection established
    }

    websocket.onerror = () => {
      clearTimeout(connectTimeout)
      if (!isMounted) return
      setError('WebSocket connection failed')
      setGenerating(false)
    }

    wsRef.current = websocket

    return () => {
      isMounted = false
      clearTimeout(connectTimeout)
      if (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING) {
        websocket.close()
      }
      wsRef.current = null
    }
  }, [generating, currentSession?.id])

  const loadReport = async () => {
    if (!currentSession?.id) return

    setLoading(true)
    setError(null)

    try {
      const data = await apiFetch<any>(`/session/${currentSession.id}/detailed-report${branchQuery}`)
      if (data?.report) {
        setReport(data.report)
        // A partial report is a finished, downloadable artifact with an
        // explicit coverage warning, not an in-progress generation.
        if (data.report.status === 'completed' || data.report.status === 'partial') {
          setReportGenerated(true)
        }
        // Also load preview
        await Promise.all([loadPreview(), fetchArchives()])
      }
    } catch (e) {
      console.error('Failed to load report:', e)
    } finally {
      setLoading(false)
    }
  }

  const loadPreview = async () => {
    if (!currentSession?.id) return

    try {
      const data = await apiFetch<any>(`/session/${currentSession.id}/report/preview${branchQuery}`)
      if (data) {
        setPreview(data)
      }
    } catch (e) {
      console.error('Failed to load preview:', e)
    }
  }

  const fetchArchives = useCallback(async () => {
    if (!currentSession?.id) return
    try {
      const data = await apiFetch<any>(`/session/${currentSession.id}/report/archives${branchQuery}`)
      if (data) {
        setReportArchives(data)
      }
    } catch {
      setReportArchives([])
    }
  }, [currentSession?.id, currentSession?.restoreBranchId])

  const doGenerateReport = async () => {
    if (!currentSession?.id) return

    setGenerating(true)
    setError(null)

    // Add start message to stream
    addMessage({
      id: `report-start-${Date.now()}`,
      type: 'status',
      agent: 'pi',
      content: language === 'zh' ? '开始生成详细报告...' : 'Starting detailed report generation...',
      timestamp: new Date().toISOString()
    })

    // Initialize report with pending status
    setReport({
      id: '',
      session_id: currentSession.id,
      title: '',
      abstract: '',
      sections: [],
      conclusions: '',
      figures: [],
      tables: [],
      generated_at: '',
      status: 'generating',
      progress: 0,
      current_step: language === 'zh' ? '初始化报告生成...' : 'Initializing report generation...'
    })

    try {
      await apiFetch(`/session/${currentSession.id}/generate-detailed-report`, {
        method: 'POST'
      })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      if (msg.includes('already in progress') || msg.includes('409')) {
        setError(language === 'zh' ? '报告生成已在进行中' : 'Report generation already in progress')
        setGenerating(false)
        return
      }
      setError(msg)
      setGenerating(false)
    }
  }

  const generateReport = async () => {
    if (!currentSession?.id) return

    setValidating(true)
    try {
      const isAdmin = authUser?.role === 'admin'
      let validation: { valid: boolean; message?: string; uncertain?: boolean }

      if (isAdmin) {
        const { apiConfig, apiConfigured } = useAppStore.getState()
        if (!apiConfig.apiKey && apiConfigured) {
          // No local key but server is configured: validate the saved global key
          // instead of sending an empty key.
          validation = await adminConfig.validateSavedConfig()
        } else {
          validation = await adminConfig.validateApiKey()
        }
      } else {
        validation = await userConfig.validateSavedConfig()
      }

      setValidating(false)

      if (!validation.valid && !validation.uncertain) {
        setValidationMessage(validation.message || t('apiKeyInvalid'))
        setValidationUncertain(validation.uncertain || false)
        setShowValidationDialog(true)
        return
      }
    } catch {
      setValidating(false)
    }

    await doGenerateReport()
  }

  const safeDownload = async (url: string, filename: string, options?: RequestInit) => {
    setDownloadError(null)
    try {
      await fetchAndDownload(url, filename, options)
    } catch (e: unknown) {
      setDownloadError(e instanceof Error ? e.message : 'Download failed')
    }
  }

  const downloadReportArtifact = async (endpoint: ReportDownloadEndpoint, label: string) => {
    if (!report || !currentSession) return
    await safeDownload(
      `${API_BASE}/session/${currentSession.id}/report/${endpoint}${branchQuery}`,
      `${currentSession.name}_${label}.zip`,
      { method: 'POST' }
    )
  }

  const openHtmlInNewTab = () => {
    if (!currentSession?.id) return
    window.open(`${API_BASE}/session/${currentSession.id}/report/html${branchQuery}`, '_blank')
  }

  if (!currentSession) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>{language === 'zh' ? '请先创建或选择一个分析会话' : 'Please create or select an analysis session first'}</p>
        </div>
      </div>
    )
  }

  if (currentSession.status !== 'completed') {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <Loader2 className="w-16 h-16 mx-auto mb-4 animate-spin text-accent-400" />
          <p className="text-lg font-medium mb-2">
            {language === 'zh' ? '分析进行中...' : 'Analysis in progress...'}
          </p>
          <p className="text-sm">
            {language === 'zh'
              ? '分析完成后将自动生成详细报告'
              : 'Detailed report will be generated automatically when analysis completes'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Validating spinner overlay */}
      {validating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-lg p-6 flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-accent-600" />
            <span className="text-slate-700 font-medium">
              {t('validatingApiKey')}
            </span>
          </div>
        </div>
      )}

      {/* Validation failure dialog */}
      {showValidationDialog && (
        <ValidationDialog
          message={validationMessage}
          uncertain={validationUncertain}
          onIgnore={() => { setShowValidationDialog(false); doGenerateReport() }}
          onReconfigure={() => { setShowValidationDialog(false); if (showConfigPanel) showConfigPanel() }}
          language={language}
        />
      )}

      {/* Header */}
      <div className="p-4 border-b border-slate-200 bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-accent-600" />
            <h2 className="font-display text-xl font-semibold text-slate-900">
              {language === 'zh' ? '详细研究报告' : 'Detailed Research Report'}
            </h2>
          </div>

          <div className="flex items-center gap-2">
            {report && (report.status === 'completed' || report.status === 'partial') && (
              <>
                {preview?.has_html && (
                  <button
                    onClick={openHtmlInNewTab}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-accent-50 text-accent-700 rounded-lg hover:bg-accent-100"
                  >
                    <ExternalLink className="w-4 h-4" />
                    {language === 'zh' ? '新窗口打开' : 'Open in Tab'}
                  </button>
                )}

                <button
                  onClick={() => downloadReportArtifact('download-all-figures', 'figures')}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-pink-50 text-pink-700 rounded-lg hover:bg-pink-100"
                >
                  <Image className="w-4 h-4" />
                  {language === 'zh' ? '下载图片' : 'Figures'}
                </button>
                <button
                  onClick={() => downloadReportArtifact('download-all-tables', 'tables')}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-emerald-50 text-emerald-700 rounded-lg hover:bg-emerald-100"
                >
                  <Table className="w-4 h-4" />
                  {language === 'zh' ? '下载表格' : 'Tables'}
                </button>
                <button
                  onClick={() => downloadReportArtifact('download-full', 'full_report')}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-accent-50 text-accent-700 rounded-lg hover:bg-accent-100"
                >
                  <Package className="w-4 h-4" />
                  {language === 'zh' ? '打包下载' : 'Download All'}
                </button>
              </>
            )}

            {/* Only show regenerate button when report exists */}
            {report && (report.status === 'completed' || report.status === 'partial') && (
              <button
                onClick={generateReport}
                disabled={generating}
                className="flex items-center gap-1.5 px-4 py-2 text-sm bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 disabled:opacity-50"
              >
                {generating ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4" />
                )}
                {generating
                  ? (language === 'zh' ? '生成中...' : 'Generating...')
                  : (language === 'zh' ? '重新生成' : 'Regenerate')}
              </button>
            )}
          </div>
        </div>

        {downloadError && (
          <ErrorBanner message={downloadError} onDismiss={() => setDownloadError(null)} />
        )}

        {archiveWarning && (
          <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-800">
            {language === 'zh' ? '归档历史报告失败：' : 'Previous report archive failed: '}{archiveWarning}
          </div>
        )}

        {report?.status === 'partial' && (
          <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-800">
            {partialReportNotice(report, language)}
          </div>
        )}

        {reportArchives.length > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-200">
            <h4 className="text-sm font-medium text-slate-600 mb-2">
              {language === 'zh' ? '历史报告' : 'Previous Reports'}
            </h4>
            <div className="space-y-1">
              {reportArchives.map(archive => (
                <button
                  key={archive.filename}
                  onClick={() => {
                    safeDownload(`${API_BASE}/${archive.download_url}`, archive.filename)
                  }}
                  className="block text-sm text-blue-600 hover:text-blue-800 hover:underline text-left cursor-pointer"
                >
                  v{archive.version} — {new Date(archive.created_at).toLocaleString()} ({(archive.size_bytes / 1024 / 1024).toFixed(1)} MB)
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Progress bar */}
        {generating && report && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm text-slate-600 mb-2">
              <span>{report.current_step}</span>
              <span>{Math.round(report.progress)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-accent-600 transition-all duration-300"
                style={{ width: `${report.progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col min-h-0">
        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <Loader2 className="w-8 h-8 animate-spin text-accent-600" />
          </div>
        ) : error ? (
          <div className="flex-1 overflow-auto p-4">
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                <span>{error}</span>
              </div>
            </div>
          </div>
        ) : !report || report.status === 'pending' ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
            <FileText className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg mb-2">
              {language === 'zh' ? '详细报告尚未生成' : 'Detailed report not yet generated'}
            </p>
            <p className="text-sm mb-4">
              {language === 'zh'
                ? '报告可能正在生成中，或者生成过程中出现了问题'
                : 'Report may be generating, or there was an issue during generation'}
            </p>
            <button
              onClick={generateReport}
              disabled={generating}
              className="flex items-center gap-1.5 px-4 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50"
            >
              {generating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              {generating
                ? (language === 'zh' ? '生成中...' : 'Generating...')
                : (language === 'zh' ? '生成报告' : 'Generate Report')}
            </button>
          </div>
        ) : report.status === 'generating' ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
            <Loader2 className="w-16 h-16 mb-4 animate-spin text-accent-500" />
            <p className="text-lg mb-2">
              {language === 'zh' ? '正在生成详细报告...' : 'Generating detailed report...'}
            </p>
            <p className="text-sm text-accent-600">
              {report.current_step || (language === 'zh' ? '准备中...' : 'Preparing...')}
            </p>
            {report.progress > 0 && (
              <div className="w-64 mt-4">
                <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-accent-600 transition-all duration-300"
                    style={{ width: `${report.progress}%` }}
                  />
                </div>
                <p className="text-xs text-center mt-1">{Math.round(report.progress)}%</p>
              </div>
            )}
          </div>
        ) : preview?.has_html ? (
          // HTML Report - use src instead of srcDoc so relative paths work
          <iframe
            src={preview.html_url}
            className="publication-canvas w-full flex-1 border-0 min-h-0"
            title="Report"
          />
        ) : (
          // Report record exists but the rendered HTML is not available
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
            <FileText className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg mb-2">
              {language === 'zh' ? 'HTML 报告暂不可用' : 'HTML report not available'}
            </p>
            <p className="text-sm mb-4">
              {language === 'zh'
                ? '报告记录已存在，但渲染后的 HTML 文件缺失，请尝试重新生成。'
                : 'A report record exists, but the rendered HTML file is missing. Try regenerating.'}
            </p>
            <button
              onClick={generateReport}
              disabled={generating}
              className="flex items-center gap-1.5 px-4 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50"
            >
              {generating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              {generating
                ? (language === 'zh' ? '生成中...' : 'Generating...')
                : (language === 'zh' ? '重新生成' : 'Regenerate')}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
