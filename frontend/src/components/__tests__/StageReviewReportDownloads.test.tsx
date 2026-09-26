import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { AnnotationConfirmationDialog } from '../AnnotationConfirmationDialog'
import { ExplorationReportDialog } from '../ExplorationReportDialog'
import { fetchAndDownload } from '../../utils/download'

vi.mock('../../utils/download', () => ({ fetchAndDownload: vi.fn() }))

function dialog(stage: number, url?: string) {
  const common = {
    language: 'en' as const, countdownSeconds: -1,
    detailedReportUrl: '/api/session/s/detailed-report/html',
    onClose: vi.fn(), onRequestSupplementary: vi.fn().mockResolvedValue(undefined),
  }
  return stage === 1 ? (
    <AnnotationConfirmationDialog
      {...common} annotations={[]} tissue="" organism="" tissueSummary=""
      reportHtmlUrl={url} onConfirm={vi.fn().mockResolvedValue(undefined)}
    />
  ) : (
    <ExplorationReportDialog
      {...common} roundSummary="" explorationReportUrl={url}
      report={{ cluster_overview: '', data_quality_warnings: [], strongest_heterogeneity: [],
        key_pathway_changes: [], abnormal_markers: [], low_confidence_annotations: [],
        pi_direction_assessment: '', research_direction_candidates: [] }}
      onChooseDirections={vi.fn().mockResolvedValue(undefined)}
      onLetPiDecide={vi.fn().mockResolvedValue(undefined)}
    />
  )
}

describe.each([1, 2])('Stage %i review report download', stage => {
  beforeEach(() => vi.mocked(fetchAndDownload).mockReset())

  it('uses the authenticated download helper and preserves the report version and branch', async () => {
    const url = `/api/session/s/stage-review-report/stage-${stage}-v2/html?branch_id=restore-a`
    vi.mocked(fetchAndDownload).mockResolvedValue(undefined)
    render(dialog(stage, url))
    fireEvent.click(screen.getByRole('button', { name: `Download Stage ${stage} Review Report` }))
    await waitFor(() => expect(fetchAndDownload).toHaveBeenCalledWith(
      url, `round_1_stage_${stage}_review_report.html`, { preferResponseFilename: true },
    ))
    await waitFor(() => expect(screen.getByRole('button', {
      name: `Download Stage ${stage} Review Report`,
    })).toBeEnabled())
    expect(screen.queryByText('View Detailed Report')).not.toBeInTheDocument()
  })

  it('shows unavailable without falling back to the cumulative report', () => {
    render(dialog(stage))
    expect(screen.getByText(`Stage ${stage} review report is currently unavailable for download`))
      .toBeInTheDocument()
    expect(screen.queryByText('View Detailed Report')).not.toBeInTheDocument()
    expect(fetchAndDownload).not.toHaveBeenCalled()
  })

  it('shows a download error and permits retry', async () => {
    vi.mocked(fetchAndDownload).mockRejectedValueOnce(new Error('unauthorized'))
    render(dialog(stage, `/api/session/s/stage-review-report/stage-${stage}-v1/html`))
    fireEvent.click(screen.getByRole('button', { name: `Download Stage ${stage} Review Report` }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Report download failed. Please retry.')
    expect(screen.getByRole('button', { name: `Download Stage ${stage} Review Report` })).toBeEnabled()
  })
})
