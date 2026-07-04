import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { IntegrationGatePanel } from '../IntegrationGatePanel'

const candidates = [
  { direction: 'New direction A', direction_id: 'int_001' },
  { direction: 'New direction B', direction_id: 'int_002' },
]

describe('IntegrationGatePanel', () => {
  it('renders candidates and both actions', () => {
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={() => {}} onFinalize={() => {}} />)
    expect(screen.getByText(/New direction A/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /report|撰写/i })).toBeInTheDocument()
  })

  it('calls onFinalize on write-report click', () => {
    const onFinalize = vi.fn()
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={() => {}} onFinalize={onFinalize} />)
    fireEvent.click(screen.getByRole('button', { name: /report|撰写/i }))
    expect(onFinalize).toHaveBeenCalled()
  })

  it('selection resets when candidates change across cycles', () => {
    const onSubmit = vi.fn()
    const c1 = [{ direction: 'A' }, { direction: 'B' }] // index-fallback keys "0","1"
    const { rerender } = render(
      <IntegrationGatePanel cycle={1} candidates={c1} onSubmit={onSubmit} onFinalize={() => {}} />
    )
    fireEvent.click(screen.getAllByRole('checkbox')[0]) // select index 0 ("A")

    const c2 = [{ direction: 'C' }, { direction: 'D' }] // new cycle, same index keys
    rerender(
      <IntegrationGatePanel cycle={2} candidates={c2} onSubmit={onSubmit} onFinalize={() => {}} />
    )
    // No stale checkbox should be pre-checked after the candidate swap.
    screen.getAllByRole('checkbox').forEach((cb) => expect(cb).not.toBeChecked())
  })
})
