import { useCallback, useEffect, useRef, useState } from 'react'

import { useSession } from '../../hooks/useApi'
import { useTranslation } from '../../i18n/translations'
import {
  useAppStore,
  type GraphicAbstract as GraphicAbstractType,
  type IntegratedReportPublication,
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
  generation_id?: string
  publication_revision?: number
}

interface OwnedResultsState {
  ownerKey: string | null
  report: any
  publicationFigures: PublicationFigure[]
  publicationTables: PublicationTable[]
  integratedReport: string | null
  integratedReportPublication: IntegratedReportPublication | null
  integratedReportRevision: number
  integratedReportGenerationId: string | null
  graphicAbstract: GraphicAbstractType | null
  curatedKeys: Set<string>
  expandedRounds: Set<string>
  loading: boolean
  error: string | null
  reportRequestGeneration: number
}

// Registered revisions may deliberately share a display name. Report binding
// therefore follows the immutable artifact id; the name is only a fallback for
// old workspace messages that predate stable ids.
const figureKey = (f: PublicationFigure) => f.id || `workspace:${f.name}`

const withoutReportBinding = (figure: PublicationFigure): PublicationFigure => {
  const {
    reportNumber: _reportNumber,
    reportGenerationId: _reportGenerationId,
    piReportNote: _piReportNote,
    ...unbound
  } = figure
  return unbound
}

export function reportBoundFigures(
  session: Session | null,
): { figures: PublicationFigure[]; curatedKeys: Set<string> } {
  const inventory = (session?.publicationFigures ?? []).map(withoutReportBinding)
  const publication = session?.integratedReportPublication
  if (!publication) return { figures: inventory, curatedKeys: new Set() }
  const inventoryById = new Map(inventory.map(figure => [figure.id, figure]))
  const curated = publication.figureRefs.map(ref => {
    const existing = inventoryById.get(ref.figureId)
    return {
      ...(existing || {
        id: ref.figureId,
        name: ref.name,
        description: ref.description,
        figureType: ref.figureType,
        pngPath: '',
        pdfPath: '',
        reviewStatus: ref.reviewStatus,
        revisionCount: ref.revisionCount,
        createdAt: publication.publishedAt,
      }),
      id: ref.figureId,
      name: ref.name,
      description: ref.description,
      figureType: ref.figureType,
      reviewStatus: ref.reviewStatus,
      reviewUnavailable: ref.reviewUnavailable,
      revisionCount: ref.revisionCount,
      round: ref.roundNumber,
      diagnostic: ref.diagnostic,
      isPublication: ref.isPublication,
      reportNumber: ref.number,
      reportGenerationId: publication.generationId,
      piReportNote: ref.piReportNote,
      // Numbered report cards must load through the generation-bound endpoint;
      // an inventory preview is not proof that the frozen bytes still exist.
      pngBase64: undefined,
    } satisfies PublicationFigure
  })
  const curatedIds = new Set(curated.map(figure => figure.id))
  return {
    figures: [...curated, ...inventory.filter(figure => !curatedIds.has(figure.id))],
    curatedKeys: new Set(curated.map(figure => figure.id)),
  }
}

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
      ?? (renderIdentityChanged || incoming.reportGenerationId
        ? undefined
        : existingBase64),
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
  reportNumber: f.report_number ?? f.reportNumber ?? undefined,
  reportGenerationId: f.report_generation_id ?? f.reportGenerationId ?? undefined,
  piReportNote: f.pi_report_note ?? f.piReportNote ?? undefined,
})

export function mergeIntegratedReportFigures(
  existingFigures: PublicationFigure[],
  rawFigures: Record<string, unknown>[],
  legacyNameFallback = false,
): { figures: PublicationFigure[]; curatedKeys: Set<string> } {
  const unboundFigures = existingFigures.map(withoutReportBinding)
  const curated = rawFigures.map(transformFigure).map(figure => {
    if (!legacyNameFallback || figure.id) return figure
    const existing = unboundFigures.find(candidate => (
      candidate.name === figure.name
      && (figure.isPublication !== false || candidate.isPublication === false)
    ))
    return existing ? { ...figure, id: existing.id } : figure
  })
  const merged = new Map<string, PublicationFigure>()
  // Legacy workspace scans generated new ids on each report. Keep that name
  // fallback separate from registered artifact identity.
  const mergeKey = (figure: PublicationFigure) => (
    legacyNameFallback && figure.isPublication === false
      ? `workspace:${figure.name}`
      : figureKey(figure)
  )
  for (const fig of curated) {
    const existing = unboundFigures.find(
      candidate => mergeKey(candidate) === mergeKey(fig),
    )
    merged.set(mergeKey(fig), mergeFigureMetadata(fig, existing))
  }
  for (const fig of unboundFigures) {
    if (!merged.has(mergeKey(fig))) merged.set(mergeKey(fig), fig)
  }
  return {
    figures: [...merged.values()],
    curatedKeys: new Set(curated.map(figureKey)),
  }
}

/** Every accumulated artifact value, reset unconditionally per owner. */
function makeOwnedState(
  ownerKey: string | null,
  session: Session | null,
): OwnedResultsState {
  const bound = reportBoundFigures(session)
  return {
    ownerKey,
    report: null,
    publicationFigures: bound.figures,
    publicationTables: session?.publicationTables ?? [],
    integratedReport: session?.integratedReport ?? null,
    integratedReportPublication: session?.integratedReportPublication ?? null,
    integratedReportRevision: session?.integratedReportPublication?.publicationRevision ?? 0,
    integratedReportGenerationId: session?.integratedReportPublication?.generationId ?? null,
    graphicAbstract: session?.graphicAbstract ?? null,
    curatedKeys: bound.curatedKeys,
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
    integratedReportPublication: base.integratedReportPublication,
    integratedReportRevision: base.integratedReportRevision,
    integratedReportGenerationId: base.integratedReportGenerationId,
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
        // A publication freezes the card's render and review metadata. Ordinary
        // review history must not replace that binding, even if its report event
        // was never delivered. New artifact ids remain independent gallery cards.
        if (existing?.reportNumber != null && existing.reportGenerationId) continue
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
      const currentRevision = next.integratedReportRevision
      const incomingRevision = content.publication_revision ?? 0
      const incomingGeneration = content.generation_id || null
      // A capped/replayed legacy event cannot replace an authoritative bundle.
      if (currentRevision > 0 && incomingRevision < currentRevision) continue
      if (
        currentRevision > 0
        && incomingRevision === currentRevision
        && next.integratedReportGenerationId
        && incomingGeneration
        && incomingGeneration !== next.integratedReportGenerationId
      ) continue
      if (incomingRevision > 0) {
        next = {
          ...next,
          integratedReportRevision: incomingRevision,
          integratedReportGenerationId: incomingGeneration,
        }
      }
      if (
        content?.report
        && content.report !== STRIPPED_FIGURE_PLACEHOLDER
        && !content.report.startsWith('@scai-blob:')
      ) {
        next = { ...next, integratedReport: content.report }
      }
      if (Array.isArray(content?.figures)) {
        // The versioned report event is the authority for its frozen catalogue.
        // It may legitimately reference an older render that is absent from the
        // mutable current inventory. Clear every prior report number before
        // applying the new catalogue so removed figures do not retain old labels.
        const isVersionedPublication = incomingRevision > 0 && !!incomingGeneration
        const inventoryFigures = session?.publicationFigures || []
        const inventoryIds = new Set(inventoryFigures.map(figure => figure.id))
        const inventoryNames = new Set(inventoryFigures.map(figure => figure.name))
        const reportFigures = isVersionedPublication
          ? content.figures
          : content.figures.filter((raw) => {
              const figure = transformFigure(raw)
              return figure.isPublication === false
                || (figure.id
                  ? inventoryIds.has(figure.id)
                  : inventoryNames.has(figure.name))
            })
        const merged = mergeIntegratedReportFigures(
          next.publicationFigures,
          reportFigures,
          !isVersionedPublication,
        )
        next = {
          ...next,
          publicationFigures: merged.figures,
          curatedKeys: merged.curatedKeys,
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
    && currentSession.integratedReportPublication
      === previousSessionRef.current.integratedReportPublication
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
            publicationFigures: reportBoundFigures(authoritativeSession).figures,
            publicationTables: authoritativeSession?.publicationTables ?? [],
            integratedReport: authoritativeSession?.integratedReport ?? null,
            integratedReportPublication: authoritativeSession?.integratedReportPublication ?? null,
            integratedReportRevision: authoritativeSession?.integratedReportPublication?.publicationRevision ?? 0,
            integratedReportGenerationId: authoritativeSession?.integratedReportPublication?.generationId ?? null,
            graphicAbstract: authoritativeSession?.graphicAbstract ?? null,
            curatedKeys: reportBoundFigures(authoritativeSession).curatedKeys,
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
    curatedKeys, error, expandedRounds, graphicAbstract, integratedReport,
    integratedReportPublication, loading,
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
    integratedReportPublication,
    orderedFigures, publicationFigures, publicationTables, report, t,
    toggleRound,
  }
}

export type ResultsController = ReturnType<typeof useResultsController>
