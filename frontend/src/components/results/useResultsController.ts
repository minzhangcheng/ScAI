import { useCallback, useEffect, useState } from 'react'

import { useSession } from '../../hooks/useApi'
import { useTranslation } from '../../i18n/translations'
import {
  useAppStore,
  type GraphicAbstract as GraphicAbstractType,
  type PublicationFigure,
  type PublicationTable,
} from '../../stores/appStore'
import type { GraphicAbstractContent } from '../../types'
import { downloadBlob } from '../../utils/download'
import { normalizeImageBase64 } from '../../utils/image'

// Typed content interfaces for message payloads (avoid `as any`)
interface FigureReviewContent {
  figure?: Record<string, unknown>
}

interface IntegratedReportContent {
  report?: string
  figures?: Record<string, unknown>[]
  tables?: PublicationTable[]
}

// Identity for gallery de-duplication. Names are unique per session (the agent
// upserts publication figures by name), while ids are regenerated for
// workspace-only figures every time the brief gallery is assembled — keying on
// the id alone duplicates them when a report is regenerated.
const figureKey = (f: PublicationFigure) => f.name || f.id

// Review and integrated-report messages may carry a compact figure payload.
// Keep provenance already loaded from the authoritative session inventory when
// such a message updates the same figure without repeating those fields.
export const mergeFigureMetadata = (
  incoming: PublicationFigure,
  existing?: PublicationFigure,
): PublicationFigure => {
  const incomingBase64 = normalizeImageBase64(incoming.pngBase64)
  const existingBase64 = normalizeImageBase64(existing?.pngBase64)
  const renderIdentityChanged = !!existing && (
    incoming.id !== existing.id
    || (incoming.revisionCount ?? 0) > (existing.revisionCount ?? 0)
  )
  return {
    ...incoming,
    // Compact events often omit image bytes. They may inherit the preview only
    // while they still describe the same render. A replacement id or a higher
    // revision count means the old base64 is stale even when the display name
    // is unchanged; let the id-based PNG endpoint load the new bytes instead.
    pngBase64: incomingBase64
      ?? (renderIdentityChanged ? undefined : existingBase64),
    planStepId: incoming.planStepId ?? existing?.planStepId,
    planId: incoming.planId ?? existing?.planId,
    reviewUnavailable: incoming.reviewUnavailable ?? existing?.reviewUnavailable,
    purpose: incoming.purpose ?? existing?.purpose,
    isPublication: incoming.isPublication ?? existing?.isPublication,
  }
}

export function useResultsController() {
  const { currentSession, messages, language } = useAppStore()
  const { getReport } = useSession()
  const { t } = useTranslation(language)
  const [report, setReport] = useState<any>(null)
  const [expandedRounds, setExpandedRounds] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const [publicationFigures, setPublicationFigures] = useState<PublicationFigure[]>([])
  const [publicationTables, setPublicationTables] = useState<PublicationTable[]>([])
  const [integratedReport, setIntegratedReport] = useState<string | null>(null)
  const [graphicAbstract, setGraphicAbstract] = useState<GraphicAbstractType | null>(null)
  const [curatedKeys, setCuratedKeys] = useState<Set<string>>(new Set())
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
    const transformFigure = (f: any): PublicationFigure => ({
      id: f.id,
      name: f.name,
      description: f.description,
      figureType: f.figure_type || f.figureType || '',
      pngPath: f.png_path || f.pngPath || '',
      pdfPath: f.pdf_path || f.pdfPath || '',
      pngBase64: normalizeImageBase64(f.png_base64)
        ?? normalizeImageBase64(f.pngBase64),
      reviewStatus: f.review_status || f.reviewStatus || 'pending',
      reviewUnavailable: f.review_unavailable ?? f.reviewUnavailable ?? false,
      piFeedback: f.pi_feedback || f.piFeedback,
      revisionCount: f.revision_count ?? f.revisionCount ?? 0,
      createdAt: f.created_at || f.createdAt || '',
      round: f.round ?? f.round_number ?? undefined,
      layoutBlocking: f.layout_blocking ?? f.layoutBlocking ?? false,
      diagnostic: f.diagnostic ?? false,
      planStepId: f.plan_step_id ?? f.planStepId ?? undefined,
      planId: f.plan_id ?? f.planId ?? undefined,
      purpose: f.purpose ?? undefined,
      isPublication: f.is_publication ?? f.isPublication ?? true,
    })
    for (const msg of messages) {
      if (
        msg.type === 'figure_review'
        || msg.type === 'figure_approved'
        || msg.type === 'figure_revision'
      ) {
        const content = msg.content as FigureReviewContent
        if (content?.figure) {
          const fig = transformFigure(content.figure)
          setPublicationFigures(prev => {
            const incomingKey = figureKey(fig)
            const existing = prev.find(
              (f: PublicationFigure) => figureKey(f) === incomingKey
            )
            if (existing) {
              return prev.map((f: PublicationFigure) =>
                figureKey(f) === incomingKey ? mergeFigureMetadata(fig, f) : f
              )
            }
            return [...prev, fig]
          })
        }
      }
      if (msg.type === 'integrated_report') {
        const content = msg.content as IntegratedReportContent
        if (content?.report) {
          setIntegratedReport(content.report)
        }
        if (Array.isArray(content?.figures) && content.figures.length > 0) {
          const inventoryIds = new Set(
            (currentSession?.publicationFigures || []).map(fig => fig.id)
          )
          const inventoryNames = new Set(
            (currentSession?.publicationFigures || []).map(fig => fig.name)
          )
          const curated = content.figures
            .map(transformFigure)
            .filter(fig => (
              fig.isPublication === false
              || (fig.id ? inventoryIds.has(fig.id) : inventoryNames.has(fig.name))
            ))
          setPublicationFigures(prev => {
            const merged = new Map<string, PublicationFigure>()
            for (const fig of curated) {
              const existing = prev.find(f => figureKey(f) === figureKey(fig))
              merged.set(figureKey(fig), mergeFigureMetadata(fig, existing))
            }
            for (const fig of prev) {
              if (!merged.has(figureKey(fig))) merged.set(figureKey(fig), fig)
            }
            return [...merged.values()]
          })
          setCuratedKeys(new Set(curated.map(figureKey)))
        }
        if (content?.tables) {
          setPublicationTables(content.tables)
        }
      }
      if (msg.type === 'graphic_abstract') {
        const content = msg.content as GraphicAbstractContent
        const textDiagram = content?.text_diagram || content?.textDiagram || ''
        const imageBase64 = normalizeImageBase64(content?.image_base64)
          ?? normalizeImageBase64(content?.imageBase64)
        if (imageBase64 || textDiagram) {
          setGraphicAbstract({
            id: content.id || 'ga-1',
            prompt: content.prompt || '',
            imageBase64,
            textDiagram: textDiagram || undefined,
            isTextMode: content.is_text_mode ?? content.isTextMode ?? !!textDiagram,
            createdAt: content.created_at || content.createdAt || new Date().toISOString()
          })
        }
      }
    }
  }, [currentSession, messages])
  const orderedFigures = curatedKeys.size === 0
    ? publicationFigures
    : [
        ...publicationFigures.filter(f => curatedKeys.has(figureKey(f))),
        ...publicationFigures.filter(f => !curatedKeys.has(figureKey(f))),
      ]
  const curatedCount = publicationFigures.filter(f => curatedKeys.has(figureKey(f))).length
  const toggleRound = (key: string) => {
    const newExpanded = new Set(expandedRounds)
    if (newExpanded.has(key)) {
      newExpanded.delete(key)
    } else {
      newExpanded.add(key)
    }
    setExpandedRounds(newExpanded)
  }
  const downloadReport = () => {
    if (!report?.final_summary) return
    const content = `# ${currentSession?.name}\n\n${report.final_summary}`
    const blob = new Blob([content], { type: 'text/markdown' })
    downloadBlob(blob, `${currentSession?.name || 'report'}.md`)
  }
  return {
    currentSession, curatedCount, downloadReport, expandedRounds, graphicAbstract,
    integratedReport, language, loading, orderedFigures, publicationFigures,
    publicationTables, report, t, toggleRound,
  }
}

export type ResultsController = ReturnType<typeof useResultsController>
