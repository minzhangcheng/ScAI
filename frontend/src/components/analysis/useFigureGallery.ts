import { useCallback, useEffect, useRef, useState } from 'react'

import { useTranslation } from '../../i18n/translations'
import { useAppStore, type PublicationFigure } from '../../stores/appStore'
import { chatOwnerKey } from '../../utils/chatOwnership'
import { fetchAndDownload } from '../../utils/download'
import { reportBranchQuery } from '../../utils/reportBranch'

const API_BASE = '/api'

// How many figures the gallery shows before "show all". A completed run can
// produce dozens; leading with a readable handful keeps the page browsable.
const PREVIEW_COUNT = 8

export interface FigureGalleryProps {
  figures: PublicationFigure[]
  /** How many leading entries the PI curated for the brief report, if any. */
  curatedCount?: number
  targetReportNumber?: number | null
  targetRequestId?: number
}

interface OwnedGalleryView {
  ownerKey: string | null
  selectedFigure: PublicationFigure | null
  showAll: boolean
  collapsed: boolean
}

export function useFigureGallery({
  figures, curatedCount, targetReportNumber, targetRequestId,
}: FigureGalleryProps) {
  const { currentSession, language } = useAppStore()
  const { t } = useTranslation(language)
  const ownerKey = chatOwnerKey(
    currentSession?.id, currentSession?.restoreBranchId,
  )
  const ownerKeyRef = useRef(ownerKey)
  ownerKeyRef.current = ownerKey
  const [view, setView] = useState<OwnedGalleryView>(() => ({
    ownerKey,
    selectedFigure: null,
    showAll: false,
    collapsed: false,
  }))

  // A view written for an old owner is invisible even before this effect runs.
  const visibleView = view.ownerKey === ownerKey
    ? view
    : { ownerKey, selectedFigure: null, showAll: false, collapsed: false }

  useEffect(() => {
    setView(previous => (
      previous.ownerKey === ownerKey
        ? previous
        : { ownerKey, selectedFigure: null, showAll: false, collapsed: false }
    ))
  }, [ownerKey])

  const setSelectedFigure = useCallback((figure: PublicationFigure | null) => {
    setView(previous => (
      previous.ownerKey === ownerKey
        && ownerKeyRef.current === ownerKey
        ? { ...previous, selectedFigure: figure }
        : previous
    ))
  }, [ownerKey])

  const setShowAll = useCallback((
    updater: boolean | ((value: boolean) => boolean),
  ) => {
    setView(previous => {
      if (previous.ownerKey !== ownerKey || ownerKeyRef.current !== ownerKey) return previous
      const value = typeof updater === 'function'
        ? updater(previous.showAll)
        : updater
      return { ...previous, showAll: value }
    })
  }, [ownerKey])

  const setCollapsed = useCallback((
    updater: boolean | ((value: boolean) => boolean),
  ) => {
    setView(previous => {
      if (previous.ownerKey !== ownerKey || ownerKeyRef.current !== ownerKey) return previous
      const value = typeof updater === 'function'
        ? updater(previous.collapsed)
        : updater
      return { ...previous, collapsed: value }
    })
  }, [ownerKey])

  const previewCount = Math.max(PREVIEW_COUNT, curatedCount ?? 0)
  const showAll = visibleView.showAll
  const collapsed = visibleView.collapsed
  const selectedFigure = visibleView.selectedFigure
  const visibleFigures = showAll ? figures : figures.slice(0, previewCount)
  const hiddenCount = figures.length - visibleFigures.length

  useEffect(() => {
    if (targetReportNumber == null) return
    const targetIndex = figures.findIndex(
      figure => figure.reportNumber === targetReportNumber,
    )
    if (targetIndex < 0) return
    setCollapsed(false)
    if (targetIndex >= previewCount) setShowAll(true)
  }, [figures, previewCount, setCollapsed, setShowAll, targetReportNumber, targetRequestId])
  const branchQuery = reportBranchQuery(currentSession?.restoreBranchId)
  const ownsRenderedOwner = () => {
    const active = useAppStore.getState().currentSession
    return Boolean(
      currentSession
      && ownerKey
      && chatOwnerKey(active?.id, active?.restoreBranchId) === ownerKey,
    )
  }

  const downloadPDF = async (figure: PublicationFigure) => {
    if (!currentSession || !ownsRenderedOwner()) return
    try {
      await fetchAndDownload(
        figure.reportNumber && figure.reportGenerationId
          ? `${API_BASE}/session/${currentSession.id}/integrated-report/${figure.reportGenerationId}/figure/${figure.reportNumber}${branchQuery ? `${branchQuery}&` : '?'}kind=pdf`
          : `${API_BASE}/session/${currentSession.id}/figure/${figure.id}/pdf${branchQuery}`,
        `${figure.name}.pdf`
      )
    } catch (e) {
      console.error('Failed to download PDF:', e)
    }
  }

  const downloadJPG = async (figure: PublicationFigure) => {
    if (!currentSession || !ownsRenderedOwner()) return
    try {
      await fetchAndDownload(
        figure.reportNumber && figure.reportGenerationId
          ? `${API_BASE}/session/${currentSession.id}/integrated-report/${figure.reportGenerationId}/figure/${figure.reportNumber}${branchQuery}`
          : `${API_BASE}/session/${currentSession.id}/figure/${figure.id}/png${branchQuery}`,
        `${figure.name}.jpg`
      )
    } catch (e) {
      console.error('Failed to download JPG:', e)
    }
  }

  return {
    branchQuery, collapsed, currentSession, downloadJPG, downloadPDF, figures, hiddenCount, language,
    selectedFigure, setCollapsed, setSelectedFigure, setShowAll, showAll, t, visibleFigures,
  }
}

export type FigureGalleryController = ReturnType<typeof useFigureGallery>
