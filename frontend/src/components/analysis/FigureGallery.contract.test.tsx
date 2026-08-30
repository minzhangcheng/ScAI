import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore, type PublicationFigure, type Session } from '../../stores/appStore'
import { fetchAndDownload } from '../../utils/download'
import { FigureGallery } from './FigureGallery'

vi.mock('../../utils/download', () => ({ fetchAndDownload: vi.fn() }))

function figure(index: number): PublicationFigure {
  return {
    id: `figure-${index}`,
    name: `Figure ${index}`,
    description: `Description ${index}`,
    figureType: 'umap',
    pngPath: `/figure-${index}.jpg`,
    pdfPath: `/figure-${index}.pdf`,
    pngBase64: 'AAA',
    reviewStatus: index === 1 ? 'accepted_with_reservations' : 'approved',
    revisionCount: 0,
    createdAt: '2026-08-27T00:00:00Z',
  }
}

const session: Session = {
  id: 'session-gallery', name: 'Gallery', status: 'completed', currentRound: 1,
  createdAt: '2026-08-27T00:00:00Z', instruction: 'test', treatmentGroups: [],
  dataFiles: [], biochemistryIndicators: [], roundResults: [],
}

describe('FigureGallery contract', () => {
  beforeEach(() => {
    vi.mocked(fetchAndDownload).mockReset().mockResolvedValue(undefined)
    useAppStore.setState({ currentSession: session, language: 'en' })
  })

  it('keeps status text, preview threshold, order, and show-all behavior', () => {
    render(<FigureGallery figures={Array.from({ length: 10 }, (_, i) => figure(i + 1))} />)

    expect(screen.getByText('Accepted with reservations')).toBeInTheDocument()
    expect(screen.getByAltText('Figure 1')).toBeInTheDocument()
    expect(screen.queryByAltText('Figure 9')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Show all (10)' }))
    expect(screen.getByAltText('Figure 9')).toBeInTheDocument()
  })

  it('labels an unavailable review instead of showing generic pending', () => {
    render(<FigureGallery figures={[{
      ...figure(1),
      reviewStatus: 'pending',
      reviewUnavailable: true,
    }]} />)

    expect(screen.getByText('Review unavailable')).toBeInTheDocument()
    expect(screen.queryByText('Pending Review')).not.toBeInTheDocument()
  })

  it('keeps the image download endpoint and filename', async () => {
    render(<FigureGallery figures={[figure(1)]} />)
    fireEvent.click(screen.getByRole('button', { name: 'JPG' }))
    expect(fetchAndDownload).toHaveBeenCalledWith(
      '/api/session/session-gallery/figure/figure-1/png',
      'Figure 1.jpg',
    )
  })

  it('uses the image endpoint when a persisted sentinel reaches the gallery', () => {
    const persisted = { ...figure(1), pngBase64: '[image]' }
    render(<FigureGallery figures={[persisted]} />)

    expect(screen.getByAltText('Figure 1')).toHaveAttribute(
      'src',
      '/api/session/session-gallery/figure/figure-1/png',
    )
    fireEvent.click(screen.getByAltText('Figure 1'))
    const previews = screen.getAllByAltText('Figure 1')
    expect(previews[previews.length - 1]).toHaveAttribute(
      'src',
      '/api/session/session-gallery/figure/figure-1/png',
    )
  })

  it('scopes restored previews and downloads to the active branch', async () => {
    useAppStore.setState({
      currentSession: { ...session, restoreBranchId: 'restore/a b' },
    })
    const persisted = { ...figure(1), pngBase64: '[image]' }
    render(<FigureGallery figures={[persisted]} />)

    expect(screen.getByAltText('Figure 1')).toHaveAttribute(
      'src',
      '/api/session/session-gallery/figure/figure-1/png?branch_id=restore%2Fa%20b',
    )
    fireEvent.click(screen.getByRole('button', { name: 'JPG' }))
    expect(fetchAndDownload).toHaveBeenCalledWith(
      '/api/session/session-gallery/figure/figure-1/png?branch_id=restore%2Fa%20b',
      'Figure 1.jpg',
    )
  })
})
