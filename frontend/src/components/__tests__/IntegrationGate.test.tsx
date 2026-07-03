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
})
