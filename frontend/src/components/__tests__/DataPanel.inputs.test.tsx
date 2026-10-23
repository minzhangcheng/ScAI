import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { DataPanel } from '../data/DataPanel'
import { useAppStore, type Session } from '../../stores/appStore'
import { transformLoadResponse } from '../../utils/transformLoadResponse'

const { deleteIndicator } = vi.hoisted(() => ({ deleteIndicator: vi.fn() }))
vi.mock('../../hooks/useApi', () => ({
  useSession: () => ({ deleteIndicator }),
}))

describe('Data panel user inputs', () => {
  it('keeps uploads and manual entries while hiding extracted and legacy records from rows and counts', () => {
    const { sessionUpdates } = transformLoadResponse({
      session_id: 'session', name: 'Study', status: 'completed',
      current_round: 1, created_at: '', initial_instruction: '',
      treatment_groups: [{ id: 'group', name: 'Control', description: '', sampleCount: 1, order: 0 }],
      data_files: [{ id: 'file', filename: 'chart.png', filePath: '/uploads/chart.png',
        section: 'biochemistry', dataType: 'image', groupId: 'group', timestamp: '' }],
      biochemistry_indicators: [
        { id: 'manual', indicator: 'Entered measurement', value: 5, unit: 'g',
          origin: 'manual', groupId: 'group' },
        { id: 'image', indicator: 'Image estimate', value: 2.2, unit: '',
          origin: 'extracted', groupId: 'group', provenance_notes: ['chart estimate'] },
        { id: 'legacy', indicator: 'Legacy measurement', value: 10, unit: 'g', groupId: 'group' },
      ],
    })
    useAppStore.setState({ language: 'en', currentSession: {
      id: 'session', createdAt: '', roundResults: [], ...sessionUpdates,
    } as Session })
    render(<DataPanel />)

    expect(screen.getByText('chart.png')).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Manual Indicators' })).toBeVisible()
    const manualRow = screen.getByText('Entered measurement').closest('tr')!
    expect(within(manualRow).getByText('5')).toBeVisible()
    expect(screen.queryByText('Image estimate')).not.toBeInTheDocument()
    expect(screen.queryByText('Legacy measurement')).not.toBeInTheDocument()
    expect(screen.queryByText('Source and limitations')).not.toBeInTheDocument()
    expect(screen.queryByText('Source not recorded')).not.toBeInTheDocument()
    const overview = screen.getByRole('heading', { name: 'Data Overview' }).closest('section')!
    expect(within(overview).getByText('1 files + 1 indicators')).toBeVisible()
    expect(useAppStore.getState().currentSession?.biochemistryIndicators).toHaveLength(3)
    fireEvent.click(within(manualRow).getByRole('button'))
    expect(deleteIndicator).toHaveBeenCalledWith('manual')
  })
})
