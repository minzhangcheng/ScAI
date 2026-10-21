import { render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { IndicatorForm } from '../data/IndicatorForm'
import { transformLoadResponse } from '../../utils/transformLoadResponse'
import { useTranslation } from '../../i18n/translations'

describe('Indicator provenance after session reload', () => {
  it('shows image estimates and limitations without labeling the records as manual input', () => {
    const notes = [
      'value read from image chart.png, panel B',
      'chart estimate read against an axis, not a printed number',
      'image extraction limitations: Raw numerical data are unavailable',
    ]
    const { sessionUpdates } = transformLoadResponse({
      session_id: 'session', name: 'Study', status: 'completed',
      current_round: 1, created_at: '', initial_instruction: '',
      biochemistry_indicators: [
        { id: 'image', indicator: 'Relative signal', value: 2.2, unit: '',
          source_group_label: 'Unmatched group', provenance_notes: notes },
        { id: 'legacy', indicator: 'Legacy measurement', value: 10, unit: 'g' },
      ],
    })
    render(<IndicatorForm groups={[]} indicators={sessionUpdates.biochemistryIndicators!}
      onAdd={vi.fn()} onDelete={vi.fn()} language="en" t={useTranslation('en').t} />)

    expect(screen.getByRole('heading', { name: 'Biochemistry Indicators' })).toBeVisible()
    expect(screen.queryByText('Manual Indicators')).not.toBeInTheDocument()
    const imageRow = screen.getByText('Relative signal').closest('tr')!
    for (const note of notes) expect(within(imageRow).getByText(note)).toBeVisible()
    expect(within(imageRow).getByText('Unmatched group')).toBeVisible()
    expect(within(imageRow).getByText('2.2')).toBeVisible()
    const legacyRow = screen.getByText('Legacy measurement').closest('tr')!
    expect(within(legacyRow).getByText('Source not recorded')).toBeVisible()
    expect(within(legacyRow).getByText('10')).toBeVisible()
  })
})
