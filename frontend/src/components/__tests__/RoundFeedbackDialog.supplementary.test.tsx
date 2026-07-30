import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { RoundFeedbackDialog } from '../RoundFeedbackDialog'

const roundSummary = {
  round: 2,
  summary: 'Round summary text',
  findings: ['finding one'],
  dataRequests: { pathology: [], biochemistry: [], other: [] },
  status: 'waiting_input' as const,
  phase: 'innovation' as const,
  integrationCandidates: [],
}

function baseProps(overrides: Record<string, unknown> = {}) {
  return {
    roundNumber: 2,
    maxRounds: 5,
    roundSummary,
    language: 'en' as const,
    countdownSeconds: -1,  // manual mode: no auto-skip during the test
    treatmentGroups: [],
    onSend: vi.fn(async () => {}),
    onSkip: vi.fn(async () => {}),
    onRequestSupplementary: vi.fn(async () => {}),
    ...overrides,
  }
}

describe('RoundFeedbackDialog supplementary analysis', () => {
  it('submits a round-target supplementary request without ending the round', async () => {
    const onRequestSupplementary = vi.fn(async () => {})
    const onSend = vi.fn(async () => {})
    const onSkip = vi.fn(async () => {})
    render(<RoundFeedbackDialog {...baseProps({ onRequestSupplementary, onSend, onSkip })} />)

    fireEvent.click(screen.getByRole('button', { name: 'Request More Analysis' }))
    fireEvent.change(screen.getByPlaceholderText(/one more comparison/i), {
      target: { value: 'compare the two groups again' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Submit' }))

    await waitFor(() => expect(onRequestSupplementary).toHaveBeenCalledWith('compare the two groups again'))
    // The request must not travel through the guidance/continue path that ends the round
    expect(onSend).not.toHaveBeenCalled()
    expect(onSkip).not.toHaveBeenCalled()
    // Dialog stays mounted and the modal closes, so the gate is usable again
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Submit' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Request More Analysis' })).toBeEnabled()
    expect(screen.getByText('Round summary text')).toBeInTheDocument()
  })

  it('blocks gate-leaving actions while a supplementary request is in flight', () => {
    render(<RoundFeedbackDialog {...baseProps({ isSupplementaryRunning: true })} />)

    expect(screen.getByRole('button', { name: 'Analyzing...' })).toBeDisabled()
    expect(screen.getByRole('button', { name: /Continue/ })).toBeDisabled()
    expect(screen.getByTitle('Send')).toBeDisabled()
  })

  it('keeps continue and guidance working when no request is in flight', async () => {
    const onSend = vi.fn(async () => {})
    const onSkip = vi.fn(async () => {})
    render(<RoundFeedbackDialog {...baseProps({ onSend, onSkip })} />)

    fireEvent.change(screen.getByPlaceholderText(/additional guidance/i), {
      target: { value: 'focus on the top hits' },
    })
    fireEvent.click(screen.getByTitle('Send'))
    await waitFor(() => expect(onSend).toHaveBeenCalledWith('focus on the top hits', []))

    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))
    await waitFor(() => expect(onSkip).toHaveBeenCalled())
  })
})
