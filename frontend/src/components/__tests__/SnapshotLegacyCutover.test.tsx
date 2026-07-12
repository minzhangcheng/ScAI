import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import {
  LegacySnapshotInventory,
  isLegacyRestorable,
  legacyActionsFor,
  type LegacySnapshotItem,
} from '../snapshots/LegacySnapshotInventory'

const legacySnap: LegacySnapshotItem = {
  id: 'v1-snap-1',
  name: 'Auto: pre_round_wait (Round 1)',
  phase: 'pre_round_wait',
  format_version: 1,
  restorable: false,
  unsupported: true,
}

describe('Snapshot V1 legacy cutover UI', () => {
  it('labels legacy snapshots unsupported and non-restorable', () => {
    render(
      <LegacySnapshotInventory
        snapshots={[legacySnap]}
        warningVersion="0.33.1"
      />,
    )

    expect(screen.getByTestId('legacy-inventory')).toBeInTheDocument()
    expect(screen.getByTestId('legacy-warning')).toHaveTextContent(/unsupported for restore/i)
    expect(screen.getByTestId('legacy-warning')).toHaveTextContent('0.33.1')
    expect(screen.getAllByText(/unsupported/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/not restorable/i)).toBeInTheDocument()
  })

  it('offers only metadata, export, and delete — never restore', () => {
    const onViewMetadata = vi.fn()
    const onExport = vi.fn()
    const onDelete = vi.fn()

    render(
      <LegacySnapshotInventory
        snapshots={[legacySnap]}
        warningVersion="0.33.1"
        onViewMetadata={onViewMetadata}
        onExport={onExport}
        onDelete={onDelete}
      />,
    )

    expect(screen.getByTestId('legacy-metadata-v1-snap-1')).toBeInTheDocument()
    expect(screen.getByTestId('legacy-export-v1-snap-1')).toBeInTheDocument()
    expect(screen.getByTestId('legacy-delete-v1-snap-1')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /restore/i })).not.toBeInTheDocument()
    expect(screen.queryByTestId('legacy-restore-v1-snap-1')).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId('legacy-metadata-v1-snap-1'))
    fireEvent.click(screen.getByTestId('legacy-export-v1-snap-1'))
    fireEvent.click(screen.getByTestId('legacy-delete-v1-snap-1'))
    expect(onViewMetadata).toHaveBeenCalledWith('v1-snap-1')
    expect(onExport).toHaveBeenCalledWith('v1-snap-1')
    expect(onDelete).toHaveBeenCalledWith('v1-snap-1')
  })

  it('legacy action helper excludes restore', () => {
    expect(isLegacyRestorable(legacySnap)).toBe(false)
    expect(legacyActionsFor(legacySnap)).toEqual(['metadata', 'export', 'delete'])
    expect(legacyActionsFor(legacySnap)).not.toContain('restore')
  })

  it('empty inventory renders empty state without restore affordances', () => {
    render(<LegacySnapshotInventory snapshots={[]} warningVersion="0.33.1" />)
    expect(screen.getByTestId('legacy-inventory-empty')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /restore/i })).not.toBeInTheDocument()
  })
})
