import { useState, useEffect, useCallback } from 'react'
import { useAppStore, PublicationFigure, PublicationTable, GraphicAbstract as GraphicAbstractType } from '../stores/appStore'
import { useSession } from '../hooks/useApi'
import { useTranslation } from '../i18n/translations'
import { FigureGallery, IntegratedReport, GraphicAbstract, DownloadButtons } from './analysis'
import {
  FileText,
  Download,
  ChevronDown,
  ChevronRight,
  Microscope,
  Code2,
  FlaskConical,
  BookOpen
} from 'lucide-react'

export function ResultsPanel() {
  const { currentSession, messages, language } = useAppStore()
  const { getReport } = useSession()
  const { t } = useTranslation(language)
  const [report, setReport] = useState<any>(null)
  const [expandedRounds, setExpandedRounds] = useState<Set<number>>(new Set())
  const [_loading, setLoading] = useState(false)

  // Publication outputs state
  const [publicationFigures, setPublicationFigures] = useState<PublicationFigure[]>([])
  const [publicationTables, setPublicationTables] = useState<PublicationTable[]>([])
  const [integratedReport, setIntegratedReport] = useState<string | null>(null)
  const [graphicAbstract, setGraphicAbstract] = useState<GraphicAbstractType | null>(null)

  const loadReport = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getReport()
      setReport(data)
    } catch (e) {
      console.error('Failed to load report:', e)
    } finally {
      setLoading(false)
    }
  }, [getReport])

  useEffect(() => {
    if (currentSession?.status === 'completed') {
      loadReport()
    }
  }, [currentSession?.status, loadReport])

  // Extract publication outputs from session and messages
  useEffect(() => {
    if (currentSession?.publicationFigures) {
      setPublicationFigures(currentSession.publicationFigures)
    }
    if (currentSession?.publicationTables) {
      setPublicationTables(currentSession.publicationTables)
    }
    if (currentSession?.integratedReport) {
      setIntegratedReport(currentSession.integratedReport)
    }
    if (currentSession?.graphicAbstract) {
      setGraphicAbstract(currentSession.graphicAbstract)
    }

    // Helper: transform figure from snake_case to camelCase
    const transformFigure = (f: any): PublicationFigure => ({
      id: f.id,
      name: f.name,
      description: f.description,
      figureType: f.figure_type || f.figureType || '',
      pngPath: f.png_path || f.pngPath || '',
      pdfPath: f.pdf_path || f.pdfPath || '',
      pngBase64: f.png_base64 || f.pngBase64,
      reviewStatus: f.review_status || f.reviewStatus || 'pending',
      piFeedback: f.pi_feedback || f.piFeedback,
      revisionCount: f.revision_count ?? f.revisionCount ?? 0,
      createdAt: f.created_at || f.createdAt || '',
    })

    // Process messages for real-time updates
    for (const msg of messages) {
      if (msg.type === 'figure_review' || msg.type === 'figure_approved') {
        const content = msg.content as any
        if (content?.figure) {
          const fig = transformFigure(content.figure)
          setPublicationFigures(prev => {
            const existing = prev.find((f: PublicationFigure) => f.id === fig.id)
            if (existing) {
              return prev.map((f: PublicationFigure) => f.id === fig.id ? fig : f)
            }
            return [...prev, fig]
          })
        }
      }
      if (msg.type === 'integrated_report') {
        const content = msg.content as any
        if (content?.report) {
          setIntegratedReport(content.report)
        }
        if (content?.figures) {
          setPublicationFigures(content.figures.map(transformFigure))
        }
        if (content?.tables) {
          setPublicationTables(content.tables)
        }
      }
      if (msg.type === 'graphic_abstract') {
        const content = msg.content as any
        if (content?.image_base64 || content?.imageBase64) {
          setGraphicAbstract({
            id: content.id || 'ga-1',
            prompt: content.prompt || '',
            imageBase64: content.image_base64 || content.imageBase64,
            createdAt: content.created_at || content.createdAt || new Date().toISOString()
          })
        }
      }
    }
  }, [currentSession, messages])

  const toggleRound = (round: number) => {
    const newExpanded = new Set(expandedRounds)
    if (newExpanded.has(round)) {
      newExpanded.delete(round)
    } else {
      newExpanded.add(round)
    }
    setExpandedRounds(newExpanded)
  }

  const downloadReport = () => {
    if (!report?.final_summary) return
    
    const content = `# ${currentSession?.name}\n\n${report.final_summary}`
    const blob = new Blob([content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${currentSession?.name || 'report'}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 rounded-full mb-6">
          <FileText className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="text-xl font-semibold text-slate-900 mb-2">
          {t('analysisResults')}
        </h2>
        <p className="text-slate-500 max-w-md">
          {t('waitingResultsDesc')}
        </p>
      </div>
    )
  }

  if (currentSession.status !== 'completed' && (!currentSession.roundResults || currentSession.roundResults.length === 0)) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 rounded-full mb-6">
          <FileText className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="text-xl font-semibold text-slate-900 mb-2">
          {t('waitingResults')}
        </h2>
        <p className="text-slate-500 max-w-md">
          {t('waitingResultsDesc')}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">{t('analysisResults')}</h2>
          <p className="text-sm text-slate-500 mt-1">
            {t('completedRounds', { n: currentSession.roundResults?.length || 0 })}
          </p>
        </div>
        
        {currentSession.finalSummary && (
          <button
            onClick={downloadReport}
            className="flex items-center gap-2 px-4 py-2 bg-violet-600 text-white rounded-xl font-medium hover:bg-violet-700 transition-colors"
          >
            <Download className="w-4 h-4" />
            {t('downloadReport')}
          </button>
        )}
      </div>

      {/* Publication Outputs */}
      {(currentSession.status === 'completed' || publicationFigures.length > 0) && (
        <div className="space-y-4">
          <FigureGallery figures={publicationFigures} />

          {integratedReport && (
            <IntegratedReport report={integratedReport} />
          )}

          {graphicAbstract && (
            <GraphicAbstract graphicAbstract={graphicAbstract} />
          )}

          {publicationFigures.length > 0 && (
            <DownloadButtons
              figures={publicationFigures}
              tables={publicationTables}
            />
          )}
        </div>
      )}

      {/* Round Results */}
      <div>
        <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wide mb-3">
          {t('roundDetails')}
        </h3>
        <div className="space-y-3">
          {(currentSession.roundResults || []).map((round) => {
            const isExpanded = expandedRounds.has(round.round)
            
            return (
              <div
                key={round.round}
                className="border border-slate-200 rounded-xl overflow-hidden bg-white"
              >
                {/* Round Header */}
                <button
                  onClick={() => toggleRound(round.round)}
                  className="w-full flex items-center justify-between p-4 hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {isExpanded ? (
                      <ChevronDown className="w-5 h-5 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-slate-400" />
                    )}
                    <span className="font-semibold text-slate-900">
                      Round {round.round}
                    </span>
                    <span className="text-sm text-slate-500">
                      {(round.pathologyFindings?.length || 0) + (round.bioinformaticsFindings?.length || 0) + (round.biochemistryFindings?.length || 0) + (round.literatureFindings?.length || 0)} {t('findings')}
                    </span>
                  </div>
                </button>

                {/* Round Content */}
                {isExpanded && (
                  <div className="px-4 pb-4 space-y-4">
                    {/* PI Summary */}
                    {round.summary && (
                      <div className="p-3 bg-slate-50 rounded-lg">
                        <p className="text-sm text-slate-700">{round.summary}</p>
                      </div>
                    )}

                    {/* Findings by Agent */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Pathology */}
                      {round.pathologyFindings?.length > 0 && (
                        <div className="p-3 border border-red-100 bg-red-50/50 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Microscope className="w-4 h-4 text-red-600" />
                            <span className="text-sm font-medium text-red-900">{t('pathologyFindings')}</span>
                          </div>
                          <ul className="text-sm text-red-800 space-y-1">
                            {round.pathologyFindings.map((f, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-red-400 mt-1">•</span>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Bioinformatics */}
                      {round.bioinformaticsFindings?.length > 0 && (
                        <div className="p-3 border border-green-100 bg-green-50/50 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Code2 className="w-4 h-4 text-green-600" />
                            <span className="text-sm font-medium text-green-900">{t('bioinformaticsFindings')}</span>
                          </div>
                          <ul className="text-sm text-green-800 space-y-1">
                            {round.bioinformaticsFindings.map((f, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-green-400 mt-1">•</span>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Biochemistry */}
                      {round.biochemistryFindings?.length > 0 && (
                        <div className="p-3 border border-amber-100 bg-amber-50/50 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <FlaskConical className="w-4 h-4 text-amber-600" />
                            <span className="text-sm font-medium text-amber-900">{t('biochemistryFindings')}</span>
                          </div>
                          <ul className="text-sm text-amber-800 space-y-1">
                            {round.biochemistryFindings.map((f, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-amber-400 mt-1">•</span>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Literature */}
                      {round.literatureFindings?.length > 0 && (
                        <div className="p-3 border border-blue-100 bg-blue-50/50 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <BookOpen className="w-4 h-4 text-blue-600" />
                            <span className="text-sm font-medium text-blue-900">{t('literatureFindings')}</span>
                          </div>
                          <ul className="text-sm text-blue-800 space-y-1">
                            {round.literatureFindings.map((f, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-blue-400 mt-1">•</span>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
