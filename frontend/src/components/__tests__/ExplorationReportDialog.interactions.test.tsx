import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ExplorationReportDialog, type ExplorationReport } from '../ExplorationReportDialog'

const report: ExplorationReport = {
  cluster_overview: '',
  data_quality_warnings: [],
  strongest_heterogeneity: [],
  key_pathway_changes: [],
  abnormal_markers: [],
  low_confidence_annotations: [],
  pi_direction_assessment: '',
  research_direction_candidates: [{
    direction: 'Candidate direction',
    rationale: 'Candidate rationale',
  }],
}

describe('ExplorationReportDialog interactions', () => {
  it('submits selected and custom directions with their notes', async () => {
    const onChooseDirections = vi.fn().mockResolvedValue(undefined)
    render(
      <ExplorationReportDialog
        report={report}
        roundSummary="Round complete"
        language="en"
        countdownSeconds={-1}
        onChooseDirections={onChooseDirections}
        onLetPiDecide={vi.fn().mockResolvedValue(undefined)}
        onRequestSupplementary={vi.fn().mockResolvedValue(undefined)}
        onClose={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText('Candidate direction'))
    fireEvent.change(screen.getByPlaceholderText(/Notes \(optional\): Tips for PI/), {
      target: { value: 'Candidate note' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Add Direction' }))
    fireEvent.change(screen.getByPlaceholderText('Research direction...'), {
      target: { value: 'Custom direction' },
    })
    fireEvent.change(screen.getByPlaceholderText('Notes (optional)...'), {
      target: { value: 'Custom note' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Start Innovation Analysis' }))

    await waitFor(() => expect(onChooseDirections).toHaveBeenCalledWith([
      { direction: 'Candidate direction', notes: 'Candidate note', isCustom: false },
      { direction: 'Custom direction', notes: 'Custom note', isCustom: true },
    ]))
  })
})
