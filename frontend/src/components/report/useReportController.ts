import { useCallback, useEffect, useRef, useState } from 'react'

import { WS_AUTH_SUBPROTOCOL } from '../../constants/auth'
import { API_BASE } from '../../constants/api'
import { WS_CONNECT_TIMEOUT_MS, apiFetch, useConfig, useUserConfig } from '../../hooks/useApi'
import { useSelfClearingError } from '../../hooks/useSelfClearingError'
import { useAppStore } from '../../stores/appStore'
import { getToken } from '../../utils/authStorage'
import { fetchAndDownload } from '../../utils/download'
import { devLog } from '../../utils/logger'
import { reportBranchQuery } from '../../utils/reportBranch'
import { useTranslation } from '../../i18n/translations'

export { reportBranchQuery } from '../../utils/reportBranch'

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

export function useReportController() {
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
  const [validating, setValidating] = useState(false)
  const [showValidationDialog, setShowValidationDialog] = useState(false)
  const [validationMessage, setValidationMessage] = useState('')
  const [validationUncertain, setValidationUncertain] = useState(false)
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
  useEffect(() => {
    if (!generating || !currentSession?.id) return
    let isMounted = true
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/report/${currentSession.id}`
    const token = getToken()
    const wsProtocols = token ? [WS_AUTH_SUBPROTOCOL, token] : undefined
    const websocket = new WebSocket(wsUrl, wsProtocols)
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
        Promise.all([loadPreview(), fetchArchives()])
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
        if (data.report.status === 'completed' || data.report.status === 'partial') {
          setReportGenerated(true)
        }
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
    addMessage({
      id: `report-start-${Date.now()}`,
      type: 'status',
      agent: 'pi',
      content: language === 'zh' ? '开始生成详细报告...' : 'Starting detailed report generation...',
      timestamp: new Date().toISOString()
    })
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
  return {
    archiveWarning, currentSession, doGenerateReport, downloadError, downloadReportArtifact,
    error, generating, language, loading, openHtmlInNewTab, preview, report, reportArchives,
    safeDownload, setDownloadError, setShowValidationDialog, showConfigPanel,
    showValidationDialog, t, validating, validationMessage, validationUncertain, generateReport,
  }
}

export type ReportController = ReturnType<typeof useReportController>
