import { useCallback, useEffect, useRef, useState } from 'react'

import { useSession } from '../../hooks/useApi'
import { useTranslation } from '../../i18n/translations'
import {
  useAppStore,
  type GraphicAbstract as GraphicAbstractType,
  type PublicationFigure,
  type PublicationTable,
  type Session,
} from '../../stores/appStore'
import type { GraphicAbstractContent } from '../../types'
import { STRIPPED_FIGURE_PLACEHOLDER } from '../../constants/storage'
import { chatOwnerKey } from '../../utils/chatOwnership'
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

interface OwnedResultsState {
  ownerKey: string | null
  report: any
  publicationFigures: PublicationFigure[]
  publicationTables: PublicationTable[]
  integratedReport: string | null
  graphicAbstract: GraphicAbstractType | null
  curatedKeys: Set<string>
  expandedRounds: Set<string>
  loading: boolean
  error: string | null
  reportRequestGeneration: number
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

/** Every accumulated artifact value, reset unconditionally per owner. */
function makeOwnedState(
  ownerKey: string | null,
  session: Session | null,
): OwnedResultsState {
  return {
    ownerKey,
    report: null,
    publicationFigures: session?.publicationFigures ?? [],
    publicationTables: session?.publicationTables ?? [],
    integratedReport: session?.integratedReport ?? null,
    graphicAbstract: session?.graphicAbstract ?? null,
    curatedKeys: new Set<string>(),
    expandedRounds: new Set<string>(),
    loading: false,
    error: null,
    reportRequestGeneration: 0,
  }
}

/** Install authoritative session values, then merge message artifacts. */
function mergeOwnerArtifacts(
  base: OwnedResultsState,
  messages: ReturnType<typeof useAppStore.getState>['messages'],
  session: Session | null,
): OwnedResultsState {
  let next: OwnedResultsState = {
    ...base,
    publicationFigures: base.publicationFigures,
    publicationTables: base.publicationTables,
    integratedReport: base.integratedReport,
    graphicAbstract: base.graphicAbstract,
  }

  for (const msg of messages) {
    if (
      msg.type === 'figure_review'
      || msg.type === 'figure_approved'
      || msg.type === 'figure_revision'
    ) {
      const content = msg.content as FigureReviewContent
      if (content?.figure) {
        const fig = transformFigure(content.figure)
        const incomingKey = figureKey(fig)
        const existing = next.publicationFigures.find(
          (f: PublicationFigure) => figureKey(f) === incomingKey
        )
        next = {
          ...next,
          publicationFigures: existing
            ? next.publicationFigures.map((f: PublicationFigure) =>
                figureKey(f) === incomingKey ? mergeFigureMetadata(fig, f) : f
              )
            : [...next.publicationFigures, fig],
        }
      }
    }
    if (msg.type === 'integrated_report') {
      const content = msg.content as IntegratedReportContent
      if (
        content?.report
        && content.report !== STRIPPED_FIGURE_PLACEHOLDER
        && !content.report.startsWith('@scai-blob:')
      ) {
        next = { ...next, integratedReport: content.report }
      }
      if (Array.isArray(content?.figures) && content.figures.length > 0) {
        const inventoryFigures = session?.publicationFigures || []
        const inventoryIds = new Set(inventoryFigures.map(fig => fig.id))
        const inventoryNames = new Set(inventoryFigures.map(fig => fig.name))
        const curated = content.figures
          .map(transformFigure)
          .filter(fig => (
            fig.isPublication === false
            || (fig.id ? inventoryIds.has(fig.id) : inventoryNames.has(fig.name))
          ))
        const merged = new Map<string, PublicationFigure>()
        for (const fig of curated) {
          const existing = next.publicationFigures.find(
            f => figureKey(f) === figureKey(fig)
          )
          merged.set(figureKey(fig), mergeFigureMetadata(fig, existing))
        }
        for (const fig of next.publicationFigures) {
          if (!merged.has(figureKey(fig))) merged.set(figureKey(fig), fig)
        }
        next = {
          ...next,
          publicationFigures: [...merged.values()],
          curatedKeys: new Set(curated.map(figureKey)),
        }
      }
      if (content?.tables) {
        next = { ...next, publicationTables: content.tables }
      }
    }
    if (msg.type === 'graphic_abstract') {
      const content = msg.content as GraphicAbstractContent
      const textDiagram = content?.text_diagram || content?.textDiagram || ''
      const imageBase64 = normalizeImageBase64(content?.image_base64)
        ?? normalizeImageBase64(content?.imageBase64)
      if (imageBase64 || textDiagram) {
        next = {
          ...next,
          graphicAbstract: {
            id: content.id || 'ga-1',
            prompt: content.prompt || '',
            imageBase64,
            textDiagram: textDiagram || undefined,
            isTextMode: content.is_text_mode ?? content.isTextMode ?? !!textDiagram,
            createdAt: content.created_at || content.createdAt || new Date().toISOString()
          },
        }
      }
    }
  }

  return next
}

export function useResultsController() {
  const { currentSession, messages, language } = useAppStore()
  const { getReport } = useSession()
  const { t } = useTranslation(language)
  const ownerKey = chatOwnerKey(currentSession?.id, currentSession?.restoreBranchId)
  const [ownedState, setOwnedState] = useState<OwnedResultsState>(
    () => makeOwnedState(ownerKey, currentSession),
  )
  const requestGenerationRef = useRef(0)
  const ownerKeyRef = useRef(ownerKey)
  ownerKeyRef.current = ownerKey
  const previousOwnerRef = useRef(ownerKey)
  const previousSessionRef = useRef(currentSession)
  const getReportRef = useRef(getReport)
  getReportRef.current = getReport

  // A branch-only `updateSession` changes the owner key but leaves the prior
  // branch's artifact arrays on the session object by reference. Those values
  // are not the new owner's inventory; exposing them is exactly the
  // source-figure-under-child-id defect. Session replacement and a reload
  // bring fresh arrays, which are authoritative immediately.
  const branchOnlyOwnerChange = Boolean(
    previousOwnerRef.current !== ownerKey
    && previousSessionRef.current
    && currentSession
    && previousSessionRef.current.id === currentSession.id
  )
  const carriedOldArtifacts = Boolean(
    branchOnlyOwnerChange
    && currentSession
    && previousSessionRef.current
    && currentSession.publicationFigures === previousSessionRef.current.publicationFigures
    && currentSession.publicationTables === previousSessionRef.current.publicationTables
    && currentSession.integratedReport === previousSessionRef.current.integratedReport
    && currentSession.graphicAbstract === previousSessionRef.current.graphicAbstract
  )
  const authoritativeSession = carriedOldArtifacts ? null : currentSession

  // The render-time owner boundary: a state written for an old owner is never
  // visible, even before the synchronization effect below runs.
  const visibleState = ownedState.ownerKey === ownerKey
    ? ownedState
    : makeOwnedState(ownerKey, authoritativeSession)

  useEffect(() => {
    previousOwnerRef.current = ownerKey
    previousSessionRef.current = currentSession
  }, [ownerKey, currentSession])

  useEffect(() => {
    setOwnedState(previous => {
      const base = previous.ownerKey === ownerKey
        ? {
            ...previous,
            publicationFigures: authoritativeSession?.publicationFigures ?? [],
            publicationTables: authoritativeSession?.publicationTables ?? [],
            integratedReport: authoritativeSession?.integratedReport ?? null,
            graphicAbstract: authoritativeSession?.graphicAbstract ?? null,
          }
        : makeOwnedState(ownerKey, authoritativeSession)
      // Message-derived figures are merged only after the authoritative
      // inventory for this owner is installed in the same reducer.
      return mergeOwnerArtifacts(base, messages, authoritativeSession)
    })
  }, [ownerKey, currentSession, messages, carriedOldArtifacts])

  const loadReport = useCallback(async () => {
    const requestOwnerKey = ownerKeyRef.current
    if (!requestOwnerKey) return
    const generation = requestGenerationRef.current + 1
    requestGenerationRef.current = generation
    setOwnedState(previous => {
      const base = previous.ownerKey === requestOwnerKey
        ? previous
        : makeOwnedState(requestOwnerKey, null)
      return {
        ...base,
        loading: true,
        error: null,
        reportRequestGeneration: generation,
      }
    })
    try {
      const data = await getReportRef.current()
      setOwnedState(previous => (
        previous.ownerKey === requestOwnerKey
        && previous.reportRequestGeneration === generation
          ? { ...previous, report: data, loading: false, error: null }
          : previous
      ))
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to load report'
      setOwnedState(previous => (
        previous.ownerKey === requestOwnerKey
        && previous.reportRequestGeneration === generation
          ? { ...previous, loading: false, error: message }
          : previous
      ))
      console.error('Failed to load report:', e)
    } finally {
      setOwnedState(previous => (
        previous.ownerKey === requestOwnerKey
        && previous.reportRequestGeneration === generation
          ? { ...previous, loading: false }
          : previous
      ))
    }
  }, [])

  useEffect(() => {
    if (currentSession?.status === 'completed') {
      void loadReport()
    }
  }, [currentSession?.status, ownerKey, loadReport])

  const toggleRound = useCallback((key: string) => {
    setOwnedState(previous => {
      // Capture the rendered owner as part of the callback. A toggle retained
      // by an old branch must not mutate the replacement branch after its
      // state has been installed.
      if (previous.ownerKey !== ownerKey || ownerKeyRef.current !== ownerKey) return previous
      const expandedRounds = new Set(previous.expandedRounds)
      if (expandedRounds.has(key)) {
        expandedRounds.delete(key)
      } else {
        expandedRounds.add(key)
      }
      return { ...previous, expandedRounds }
    })
  }, [ownerKey])

  const {
    curatedKeys, error, expandedRounds, graphicAbstract, integratedReport, loading,
    publicationFigures, publicationTables, report,
  } = visibleState

  const orderedFigures = curatedKeys.size === 0
    ? publicationFigures
    : [
        ...publicationFigures.filter(f => curatedKeys.has(figureKey(f))),
        ...publicationFigures.filter(f => !curatedKeys.has(figureKey(f))),
      ]
  const curatedCount = publicationFigures.filter(f => curatedKeys.has(figureKey(f))).length
  const downloadReport = () => {
    if (!report?.final_summary) return
    const content = `# ${currentSession?.name}\n\n${report.final_summary}`
    const blob = new Blob([content], { type: 'text/markdown' })
    downloadBlob(blob, `${currentSession?.name || 'report'}.md`)
  }
  return {
    currentSession, curatedCount, downloadReport, error, expandedRounds,
    graphicAbstract, integratedReport, language, loadReport, loading,
    orderedFigures, publicationFigures, publicationTables, report, t,
    toggleRound,
  }
}

export type ResultsController = ReturnType<typeof useResultsController>
