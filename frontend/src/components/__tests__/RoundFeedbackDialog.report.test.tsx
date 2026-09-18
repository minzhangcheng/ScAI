import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { RoundFeedbackDialog } from '../RoundFeedbackDialog'
import { fetchAndDownload } from '../../utils/download'

vi.mock('../../utils/download', () => ({ fetchAndDownload: vi.fn() }))

const props = {
  roundNumber: 2, maxRounds: 5, language: 'en' as const, countdownSeconds: -1,
  treatmentGroups: [], onSend: vi.fn(), onSkip: vi.fn(), onRequestSupplementary: vi.fn(),
  roundSummary: {
    round: 2, summary: 'Saved findings', findings: [],
    dataRequests: { pathology: [], biochemistry: [], other: [] },
    status: 'waiting_input' as const, phase: 'innovation' as const, integrationCandidates: [],
  },
}
const reportUrl = '/api/session/s/round-report/segment/html?branch_id=child'

describe('round report download', () => {
  beforeEach(() => vi.clearAllMocks())

  it('downloads the selected round without submitting or continuing', async () => {
    vi.mocked(fetchAndDownload).mockResolvedValue(undefined)
    render(<RoundFeedbackDialog {...props} reportUrl={reportUrl} />)
    fireEvent.click(screen.getByRole('button', { name: /Download this round/ }))
    await waitFor(() => expect(fetchAndDownload).toHaveBeenCalledWith(reportUrl, 'round_2_report.html'))
    expect(props.onSkip).not.toHaveBeenCalled()
    expect(props.onSend).not.toHaveBeenCalled()
  })

  it('shows a concrete error and allows retry', async () => {
    vi.mocked(fetchAndDownload).mockRejectedValue(new Error('Download failed (404): Round report unavailable'))
    render(<RoundFeedbackDialog {...props} reportUrl={reportUrl} />)
    fireEvent.click(screen.getByRole('button', { name: /Download this round/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Download failed (404)')
    expect(screen.getByRole('button', { name: /Download this round/ })).toBeEnabled()
  })

  it('does not advertise a download for an older round without a saved report', () => {
    render(<RoundFeedbackDialog {...props} />)
    expect(screen.queryByRole('button', { name: /Download this round/ })).not.toBeInTheDocument()
  })
})
