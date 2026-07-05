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
      onSubmit={() => {}} onFinalize={() => {}} onLetPiDecide={() => {}} countdownSeconds={-1} language="zh" />)
    expect(screen.getByText(/New direction A/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /撰写|report/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /PI 决定|PI decide/i })).toBeInTheDocument()
  })

  it('calls onFinalize on write-report click', () => {
    const onFinalize = vi.fn()
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={() => {}} onFinalize={onFinalize} onLetPiDecide={() => {}} countdownSeconds={-1} language="zh" />)
    fireEvent.click(screen.getByRole('button', { name: /撰写|report/i }))
    expect(onFinalize).toHaveBeenCalled()
  })

  it('calls onLetPiDecide when PI decide button clicked', () => {
    const onLetPiDecide = vi.fn()
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={() => {}} onFinalize={() => {}} onLetPiDecide={onLetPiDecide} countdownSeconds={-1} language="zh" />)
    fireEvent.click(screen.getByRole('button', { name: /PI 决定|PI decide/i }))
    expect(onLetPiDecide).toHaveBeenCalled()
  })

  it('selection resets when candidates change across cycles', () => {
    const onSubmit = vi.fn()
    const c1 = [{ direction: 'A' }, { direction: 'B' }]
    const { rerender } = render(
      <IntegrationGatePanel cycle={1} candidates={c1} onSubmit={onSubmit} onFinalize={() => {}} onLetPiDecide={() => {}} countdownSeconds={-1} language="zh" />
    )
    fireEvent.click(screen.getAllByRole('checkbox')[0])

    const c2 = [{ direction: 'C' }, { direction: 'D' }]
    rerender(
      <IntegrationGatePanel cycle={2} candidates={c2} onSubmit={onSubmit} onFinalize={() => {}} onLetPiDecide={() => {}} countdownSeconds={-1} language="zh" />
    )
    screen.getAllByRole('checkbox').forEach((cb) => expect(cb).not.toBeChecked())
  })

  it('renders English text when language is en', () => {
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={() => {}} onFinalize={() => {}} onLetPiDecide={() => {}} countdownSeconds={-1} language="en" />)
    expect(screen.getByRole('button', { name: /write report/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /let PI decide/i })).toBeInTheDocument()
    expect(screen.getByText(/Integration Analysis/i)).toBeInTheDocument()
  })
})
