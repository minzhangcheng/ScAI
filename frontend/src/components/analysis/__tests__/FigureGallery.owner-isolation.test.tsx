/** A selected lightbox figure belongs to the exact session/branch owner. */
import { renderHook, act } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { useFigureGallery } from '../useFigureGallery'
import { useAppStore, type PublicationFigure, type Session } from '../../../stores/appStore'

function session(id: string, branch: string): Session {
  return {
    id,
    name: id,
    status: 'waiting_input',
    currentRound: 1,
    createdAt: '2026-09-01T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
    restoreBranchId: branch,
  }
}

function figure(name: string): PublicationFigure {
  return {
    id: `id-${name}`,
    name,
    description: name,
    figureType: 'umap',
    pngPath: `/w/${name}.jpg`,
    pdfPath: `/w/${name}.pdf`,
    reviewStatus: 'approved',
    revisionCount: 0,
    createdAt: '2026-09-01T00:00:00Z',
  }
}

describe('Figure gallery ownership', () => {
  beforeEach(() => {
    useAppStore.setState({ language: 'en' } as any)
  })

  it('drops a source selection when only the branch owner changes', () => {
    useAppStore.setState({
      currentSession: session('same-session', 'main'),
    } as any)
    const sourceFigure = figure('source-figure')
    const { result, rerender } = renderHook(
      () => useFigureGallery({ figures: [sourceFigure] }),
    )

    act(() => result.current.setSelectedFigure(sourceFigure))
    expect(result.current.selectedFigure?.name).toBe('source-figure')
    act(() => result.current.setShowAll(true))
    expect(result.current.showAll).toBe(true)

    act(() => {
      useAppStore.getState().updateSession({ restoreBranchId: 'restore-r001' })
    })
    rerender()

    expect(result.current.selectedFigure).toBeNull()
    expect(result.current.showAll).toBe(false)
    expect(result.current.collapsed).toBe(false)
  })
})
