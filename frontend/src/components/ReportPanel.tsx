import { useState, useEffect } from 'react'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import {
  FileText,
  Download,
  RefreshCw,
  Image,
  Table,
  FileDown,
  ChevronDown,
  ChevronRight,
  Loader2,
  CheckCircle,
  AlertCircle,
  Play,
  Eye,
  Code,
  ExternalLink,
  Package
} from 'lucide-react'

const API_BASE = '/api'

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

interface DetailedReport {
  id: string
  session_id: string
  title: string
  abstract: string
  sections: ReportSection[]
  conclusions: string
  figures: ReportFigure[]
  tables: ReportTable[]
  generated_at: string
  status: 'pending' | 'generating' | 'completed' | 'failed'
  progress: number
  current_step: string
  html_path?: string
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

export function ReportPanel() {
  const { currentSession, language, addMessage, setReportGenerated } = useAppStore()
  const { t } = useTranslation(language)

  const [report, setReport] = useState<DetailedReport | null>(null)
  const [preview, setPreview] = useState<ReportPreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [viewMode, setViewMode] = useState<'preview' | 'detail'>('preview')

  // Load existing report on mount
  useEffect(() => {
    if (currentSession?.id) {
      loadReport()
    }
  }, [currentSession?.id])

  // WebSocket for real-time progress
  useEffect(() => {
    if (!generating || !currentSession?.id) return

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/report/${currentSession.id}`
    const websocket = new WebSocket(wsUrl)

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'progress') {
        setReport(prev => prev ? {
          ...prev,
          progress: data.progress,
          current_step: data.step,
          status: data.status
        } : null)
        // Also add to message stream
        addMessage({
          id: `report-progress-${Date.now()}`,
          type: 'progress',
          agent: 'pi',
          content: `[Report] ${data.step} (${Math.round(data.progress)}%)`,
          timestamp: new Date().toISOString()
        })
      } else if (data.type === 'complete') {
        setReport(data.report)
        setGenerating(false)
        setReportGenerated(true)
        // Load preview after generation
        loadPreview()
        // Add completion message
        addMessage({
          id: `report-complete-${Date.now()}`,
          type: 'completed',
          agent: 'pi',
          content: {
            status: 'completed',
            message: language === 'zh' ? '报告生成完成' : 'Report generation completed'
          },
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
      }
    }

    websocket.onerror = () => {
      setError('WebSocket connection failed')
      setGenerating(false)
    }

    setWs(websocket)

    return () => {
      websocket.close()
    }
  }, [generating, currentSession?.id])

  const loadReport = async () => {
    if (!currentSession?.id) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/session/${currentSession.id}/detailed-report`)
      if (response.ok) {
        const data = await response.json()
        if (data.report) {
          setReport(data.report)
          // Mark report as generated if it's completed
          if (data.report.status === 'completed') {
            setReportGenerated(true)
          }
          // Expand first section by default
          if (data.report.sections?.length > 0) {
            setExpandedSections(new Set([data.report.sections[0].id]))
          }
          // Also load preview
          await loadPreview()
        }
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
      const response = await fetch(`${API_BASE}/session/${currentSession.id}/report/preview`)
      if (response.ok) {
        const data = await response.json()
        setPreview(data)
      }
    } catch (e) {
      console.error('Failed to load preview:', e)
    }
  }

  const generateReport = async () => {
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
      const response = await fetch(`${API_BASE}/session/${currentSession.id}/generate-detailed-report`, {
        method: 'POST'
      })

      if (!response.ok) {
        throw new Error('Failed to start report generation')
      }
    } catch (e: any) {
      setError(e.message)
      setGenerating(false)
    }
  }

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(sectionId)) {
        next.delete(sectionId)
      } else {
        next.add(sectionId)
      }
      return next
    })
  }

  const downloadFile = async (path: string, filename: string) => {
    try {
      const response = await fetch(`${API_BASE}/download?path=${encodeURIComponent(path)}`)
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error('Download failed:', e)
    }
  }

  const downloadFromUrl = async (url: string, filename: string) => {
    try {
      const response = await fetch(url)
      const blob = await response.blob()
      const blobUrl = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = blobUrl
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(blobUrl)
    } catch (e) {
      console.error('Download failed:', e)
    }
  }

  const downloadAllFigures = async () => {
    if (!report || !currentSession) return

    try {
      const response = await fetch(`${API_BASE}/session/${currentSession.id}/report/download-all-figures`, {
        method: 'POST'
      })
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${currentSession.name}_figures.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error('Download failed:', e)
    }
  }

  const downloadAllTables = async () => {
    if (!report || !currentSession) return

    try {
      const response = await fetch(`${API_BASE}/session/${currentSession.id}/report/download-all-tables`, {
        method: 'POST'
      })
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${currentSession.name}_tables.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error('Download failed:', e)
    }
  }

  const downloadFullReport = async () => {
    if (!report || !currentSession) return

    try {
      const response = await fetch(`${API_BASE}/session/${currentSession.id}/report/download-full`, {
        method: 'POST'
      })
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${currentSession.name}_full_report.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error('Download failed:', e)
    }
  }

  const openHtmlInNewTab = () => {
    if (!currentSession?.id) return
    window.open(`${API_BASE}/session/${currentSession.id}/report/html`, '_blank')
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
          <AlertCircle className="w-16 h-16 mx-auto mb-4 opacity-50 text-amber-500" />
          <p className="text-lg font-medium mb-2">
            {language === 'zh' ? '分析尚未完成' : 'Analysis not completed'}
          </p>
          <p className="text-sm">
            {language === 'zh'
              ? '请等待分析完成后再生成详细报告'
              : 'Please wait for the analysis to complete before generating a detailed report'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-200 bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-indigo-600" />
            <h2 className="text-xl font-semibold text-slate-900">
              {language === 'zh' ? '详细研究报告' : 'Detailed Research Report'}
            </h2>
          </div>

          <div className="flex items-center gap-2">
            {report && report.status === 'completed' && (
              <>
                {/* View mode toggle */}
                <div className="flex items-center bg-slate-100 rounded-lg p-1 mr-2">
                  <button
                    onClick={() => setViewMode('preview')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md transition-colors ${
                      viewMode === 'preview'
                        ? 'bg-white text-indigo-700 shadow-sm'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <Eye className="w-4 h-4" />
                    {language === 'zh' ? '预览' : 'Preview'}
                  </button>
                  <button
                    onClick={() => setViewMode('detail')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md transition-colors ${
                      viewMode === 'detail'
                        ? 'bg-white text-indigo-700 shadow-sm'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <Code className="w-4 h-4" />
                    {language === 'zh' ? '详情' : 'Detail'}
                  </button>
                </div>

                {preview?.has_html && (
                  <button
                    onClick={openHtmlInNewTab}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-violet-50 text-violet-700 rounded-lg hover:bg-violet-100"
                  >
                    <ExternalLink className="w-4 h-4" />
                    {language === 'zh' ? '新窗口打开' : 'Open in Tab'}
                  </button>
                )}

                <button
                  onClick={downloadAllFigures}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-pink-50 text-pink-700 rounded-lg hover:bg-pink-100"
                >
                  <Image className="w-4 h-4" />
                  {language === 'zh' ? '下载图片' : 'Figures'}
                </button>
                <button
                  onClick={downloadAllTables}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-emerald-50 text-emerald-700 rounded-lg hover:bg-emerald-100"
                >
                  <Table className="w-4 h-4" />
                  {language === 'zh' ? '下载表格' : 'Tables'}
                </button>
                <button
                  onClick={downloadFullReport}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100"
                >
                  <Package className="w-4 h-4" />
                  {language === 'zh' ? '打包下载' : 'Download All'}
                </button>
              </>
            )}

            <button
              onClick={generateReport}
              disabled={generating}
              className="flex items-center gap-1.5 px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
            >
              {generating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {generating
                ? (language === 'zh' ? '生成中...' : 'Generating...')
                : (language === 'zh' ? '生成报告' : 'Generate')}
            </button>
          </div>
        </div>

        {/* Progress bar */}
        {generating && report && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm text-slate-600 mb-2">
              <span>{report.current_step}</span>
              <span>{Math.round(report.progress)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-600 transition-all duration-300"
                style={{ width: `${report.progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
          </div>
        ) : error ? (
          <div className="m-4 bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              <span>{error}</span>
            </div>
          </div>
        ) : !report || report.status === 'pending' ? (
          <div className="flex flex-col items-center justify-center h-64 text-slate-500">
            <FileText className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg mb-2">
              {language === 'zh' ? '尚未生成详细报告' : 'No detailed report yet'}
            </p>
            <p className="text-sm">
              {language === 'zh'
                ? '点击上方"生成报告"按钮开始生成'
                : 'Click "Generate" button above to start'}
            </p>
          </div>
        ) : viewMode === 'preview' && preview?.has_html ? (
          // HTML Preview Mode
          <div className="h-full">
            <iframe
              srcDoc={preview.html_content || ''}
              className="w-full h-full border-0"
              title="Report Preview"
              sandbox="allow-same-origin"
            />
          </div>
        ) : (
          // Detail Mode
          <div className="p-4">
            <div className="max-w-4xl mx-auto space-y-6">
              {/* Title */}
              <div className="bg-gradient-to-r from-indigo-50 to-violet-50 rounded-2xl p-6 border border-indigo-100">
                <h1 className="text-2xl font-bold text-slate-900 mb-2">{report.title}</h1>
                <p className="text-sm text-slate-500">
                  {language === 'zh' ? '生成时间：' : 'Generated: '}
                  {new Date(report.generated_at).toLocaleString()}
                </p>
              </div>

              {/* Abstract */}
              {report.abstract && (
                <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                  <h2 className="text-lg font-semibold text-slate-900 mb-3 flex items-center gap-2">
                    <span className="w-1 h-6 bg-indigo-500 rounded-full" />
                    {language === 'zh' ? '摘要' : 'Abstract'}
                  </h2>
                  <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{report.abstract}</p>
                </div>
              )}

              {/* Sections */}
              {report.sections?.map((section) => (
                <div key={section.id} className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                  {/* Section Header */}
                  <button
                    onClick={() => toggleSection(section.id)}
                    className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-50 transition-colors"
                  >
                    <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                      <span className={`w-1 h-6 rounded-full ${section.analysis_type === 'general' ? 'bg-blue-500' : 'bg-amber-500'}`} />
                      {section.title}
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        section.analysis_type === 'general'
                          ? 'bg-blue-100 text-blue-700'
                          : 'bg-amber-100 text-amber-700'
                      }`}>
                        {section.analysis_type === 'general'
                          ? (language === 'zh' ? '通用分析' : 'General')
                          : (language === 'zh' ? '特定分析' : 'Specific')}
                      </span>
                    </h2>
                    {expandedSections.has(section.id) ? (
                      <ChevronDown className="w-5 h-5 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-slate-400" />
                    )}
                  </button>

                  {/* Section Content */}
                  {expandedSections.has(section.id) && (
                    <div className="px-6 pb-6 space-y-6">
                      {/* Text Content */}
                      {section.content && (
                        <div className="text-slate-700 leading-relaxed whitespace-pre-wrap">
                          {section.content}
                        </div>
                      )}

                      {/* Figures */}
                      {section.figures?.map((figure) => (
                        <div key={figure.id} className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                          <div className="flex items-start justify-between mb-3">
                            <h3 className="font-medium text-slate-900 flex items-center gap-2">
                              <Image className="w-4 h-4 text-pink-500" />
                              {figure.name}
                            </h3>
                            <div className="flex gap-2">
                              <button
                                onClick={() => downloadFile(figure.png_path, `${figure.name}.png`)}
                                className="text-xs px-2 py-1 bg-pink-100 text-pink-700 rounded hover:bg-pink-200"
                              >
                                PNG
                              </button>
                              {figure.pdf_path && (
                                <button
                                  onClick={() => downloadFile(figure.pdf_path!, `${figure.name}.pdf`)}
                                  className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
                                >
                                  PDF
                                </button>
                              )}
                            </div>
                          </div>

                          {/* Image Preview */}
                          <div className="mb-4 bg-white rounded-lg p-2 border border-slate-200">
                            <img
                              src={`${API_BASE}/file?path=${encodeURIComponent(figure.png_path)}`}
                              alt={figure.name}
                              className="max-w-full h-auto mx-auto cursor-pointer"
                              onClick={() => figure.pdf_path && downloadFile(figure.pdf_path, `${figure.name}.pdf`)}
                              title={figure.pdf_path ? (language === 'zh' ? '点击下载高清PDF' : 'Click to download high-res PDF') : ''}
                            />
                          </div>
                        </div>
                      ))}

                      {/* Tables */}
                      {section.tables?.map((table) => (
                        <div key={table.id} className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                          <div className="flex items-start justify-between mb-3">
                            <h3 className="font-medium text-slate-900 flex items-center gap-2">
                              <Table className="w-4 h-4 text-emerald-500" />
                              {table.name}
                            </h3>
                            <button
                              onClick={() => downloadFile(table.csv_path, `${table.name}.csv`)}
                              className="text-xs px-2 py-1 bg-emerald-100 text-emerald-700 rounded hover:bg-emerald-200"
                            >
                              CSV
                            </button>
                          </div>

                          {/* Table Preview */}
                          <div className="mb-4 overflow-x-auto">
                            <table className="min-w-full bg-white rounded-lg border border-slate-200 text-sm">
                              <thead className="bg-slate-100">
                                <tr>
                                  {table.columns?.map((col, i) => (
                                    <th key={i} className="px-3 py-2 text-left font-medium text-slate-700 border-b">
                                      {col}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {table.preview_data?.slice(0, 10).map((row, i) => (
                                  <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                                    {table.columns?.map((col, j) => (
                                      <td key={j} className="px-3 py-2 text-slate-600 border-b border-slate-100">
                                        {row[col] ?? '-'}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {table.preview_data && table.preview_data.length > 10 && (
                              <p className="text-xs text-slate-500 mt-2 text-center">
                                {language === 'zh'
                                  ? `显示前10行，共${table.preview_data.length}行`
                                  : `Showing first 10 of ${table.preview_data.length} rows`}
                              </p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {/* Conclusions */}
              {report.conclusions && (
                <div className="bg-gradient-to-r from-violet-50 to-indigo-50 rounded-xl p-6 border border-violet-100">
                  <h2 className="text-lg font-semibold text-slate-900 mb-3 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    {language === 'zh' ? '结论' : 'Conclusions'}
                  </h2>
                  <div className="text-slate-700 leading-relaxed whitespace-pre-wrap">
                    {report.conclusions}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
