import { useState } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore, type PublicationFigure, type Session } from '../../stores/appStore'
import { fetchAndDownload } from '../../utils/download'
import { FigureGallery } from './FigureGallery'
import { IntegratedReport } from './IntegratedReport'

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

  it('uses the frozen report endpoint for a numbered figure preview and download', async () => {
    const bound = {
      ...figure(1),
      pngBase64: undefined,
      reportNumber: 1,
      reportGenerationId: 'generation-1',
    }
    render(<FigureGallery figures={[bound]} curatedCount={1} />)

    expect(screen.getByAltText('Figure 1')).toHaveAttribute(
      'src',
      '/api/session/session-gallery/integrated-report/generation-1/figure/1',
    )
    fireEvent.click(screen.getByRole('button', { name: 'JPG' }))
    expect(fetchAndDownload).toHaveBeenCalledWith(
      '/api/session/session-gallery/integrated-report/generation-1/figure/1',
      'Figure 1.jpg',
    )
    fireEvent.click(screen.getByRole('button', { name: 'PDF' }))
    expect(fetchAndDownload).toHaveBeenCalledWith(
      '/api/session/session-gallery/integrated-report/generation-1/figure/1?kind=pdf',
      'Figure 1.pdf',
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

  it('expands a collapsed gallery and reveals a hidden numbered target', async () => {
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView
    const figures = Array.from({ length: 10 }, (_, index) => ({
      ...figure(index + 1),
      reportNumber: index + 1,
      reportGenerationId: 'report-generation',
    }))
    function Harness() {
      const [target, setTarget] = useState<number | null>(null)
      const [requestId, setRequestId] = useState(0)
      return <>
        <IntegratedReport
          report="See [Figure 10](#figure-10)."
          onFigureReference={(number) => {
            setTarget(number)
            setRequestId(value => value + 1)
          }}
        />
        <FigureGallery
          figures={figures}
          targetReportNumber={target}
          targetRequestId={requestId}
        />
      </>
    }

    render(<Harness />)
    fireEvent.click(screen.getByRole('button', { name: /Figure Gallery/ }))
    expect(screen.queryByAltText('Figure 1')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('link', { name: 'Figure 10' }))

    expect(await screen.findByAltText('Figure 10')).toBeInTheDocument()
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalled())
    const firstScrollCount = scrollIntoView.mock.calls.length
    fireEvent.click(screen.getByRole('link', { name: 'Figure 10' }))
    await waitFor(() => {
      expect(scrollIntoView.mock.calls.length).toBeGreaterThan(firstScrollCount)
    })
  })
})
