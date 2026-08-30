import { useState } from 'react'

import { useTranslation } from '../../i18n/translations'
import { useAppStore, type PublicationFigure } from '../../stores/appStore'
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
}

export function useFigureGallery({ figures, curatedCount }: FigureGalleryProps) {
  const { currentSession, language } = useAppStore()
  const { t } = useTranslation(language)
  const [selectedFigure, setSelectedFigure] = useState<PublicationFigure | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  const previewCount = Math.max(PREVIEW_COUNT, curatedCount ?? 0)
  const visibleFigures = showAll ? figures : figures.slice(0, previewCount)
  const hiddenCount = figures.length - visibleFigures.length
  const branchQuery = reportBranchQuery(currentSession?.restoreBranchId)

  const downloadPDF = async (figure: PublicationFigure) => {
    if (!currentSession) return
    try {
      await fetchAndDownload(
        `${API_BASE}/session/${currentSession.id}/figure/${figure.id}/pdf${branchQuery}`,
        `${figure.name}.pdf`
      )
    } catch (e) {
      console.error('Failed to download PDF:', e)
    }
  }

  const downloadJPG = async (figure: PublicationFigure) => {
    if (!currentSession) return
    try {
      await fetchAndDownload(
        `${API_BASE}/session/${currentSession.id}/figure/${figure.id}/png${branchQuery}`,
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
