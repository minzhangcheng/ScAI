import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore, type PublicationFigure, type Session } from '../../stores/appStore'
import { fetchAndDownload } from '../../utils/download'
import { DownloadButtons } from './DownloadButtons'
import { IntegratedReport } from './IntegratedReport'

vi.mock('../../utils/download', async () => {
  const actual = await vi.importActual<typeof import('../../utils/download')>(
    '../../utils/download',
  )
  return { ...actual, fetchAndDownload: vi.fn(), downloadBlob: vi.fn() }
})

const session: Session = {
  id: 'session-report', name: 'Report', status: 'completed', currentRound: 1,
  createdAt: '2026-08-29T00:00:00Z', instruction: 'test', treatmentGroups: [],
  dataFiles: [], biochemistryIndicators: [], roundResults: [],
  restoreBranchId: 'restore/a b',
}

const figure: PublicationFigure = {
  id: 'figure-1', name: 'Figure 1', description: 'Result', figureType: 'umap',
  pngPath: '/figure.png', pdfPath: '/figure.pdf', reviewStatus: 'approved',
  revisionCount: 0, createdAt: '2026-08-29T00:00:00Z',
}

describe('restored report downloads', () => {
  beforeEach(() => {
    vi.mocked(fetchAndDownload).mockReset().mockResolvedValue(undefined)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      text: async () => '<html></html>',
    }))
    useAppStore.setState({ currentSession: session, language: 'en' })
  })

  it('scopes bulk publication downloads to the restored branch', async () => {
    render(<DownloadButtons figures={[figure]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download All Figures (1)' }))

    await waitFor(() => expect(fetchAndDownload).toHaveBeenCalledWith(
      '/api/session/session-report/report/download-all-figures?branch_id=restore%2Fa%20b',
      'figures_session-report.zip',
      { method: 'POST' },
    ))
  })

  it('scopes integrated HTML downloads to the restored branch', async () => {
    render(<IntegratedReport report="Report text" />)
    fireEvent.click(screen.getByRole('button', { name: 'HTML' }))

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/session/session-report/report/html?branch_id=restore%2Fa%20b',
      expect.objectContaining({ headers: expect.any(Object) }),
    ))
  })

  it('does not publish an old branch download failure under the new owner', async () => {
    let rejectDownload!: (error: Error) => void
    vi.mocked(fetchAndDownload).mockImplementationOnce(() => new Promise<void>(
      (_resolve, reject) => { rejectDownload = reject },
    ))
    render(<DownloadButtons figures={[figure]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download All Figures (1)' }))
    expect(screen.getByRole('button', { name: 'Downloading...' })).toBeDisabled()

    act(() => {
      useAppStore.getState().updateSession({ restoreBranchId: 'restore/new-owner' })
    })
    expect(screen.getByRole('button', { name: 'Download All Figures (1)' })).toBeEnabled()

    await act(async () => {
      rejectDownload(new Error('source branch download failed'))
      await Promise.resolve()
    })

    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download All Figures (1)' })).toBeEnabled()
  })

  it('renders GitHub-flavored Markdown tables', () => {
    render(<IntegratedReport report={[
      '| Candidate | Evidence |',
      '|---|---|',
      '| **MT stress** | Observed |',
    ].join('\n')} />)

    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Candidate' })).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: 'MT stress' })).toBeInTheDocument()
  })
})
