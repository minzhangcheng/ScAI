import { act, render, screen, fireEvent, waitFor } from '@testing-library/react'

import { describe, it, expect, vi } from 'vitest'
import { IntegrationGatePanel } from '../IntegrationGatePanel'

const candidates = [
  { direction: 'New direction A', direction_id: 'int_001' },
  { direction: 'New direction B', direction_id: 'int_002' },
]

describe('IntegrationGatePanel', () => {

  it('does not retry a rejected no-wait PI decision, but lets the user retry manually', async () => {
    const onLetPiDecide = vi.fn().mockRejectedValue(new Error('upstream unavailable'))
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={async () => {}} onLetPiDecide={onLetPiDecide}
      onClose={() => {}} countdownSeconds={0} language="en" />)

    expect(await screen.findByText(/Action failed: upstream unavailable/i)).toBeInTheDocument()

    // Flush the state transition from rejection and verify it did not schedule another auto-call.
    await act(async () => {})
    expect(onLetPiDecide).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: /let PI decide/i })).not.toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: /let PI decide/i }))
    await waitFor(() => {
      expect(onLetPiDecide).toHaveBeenCalledTimes(2)
    })
  })

  it('moves focus into the dialog and traps Tab in both directions', async () => {
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={async () => {}} onLetPiDecide={async () => {}}
      onClose={() => {}} countdownSeconds={-1} language="en" open />)

    const dialog = screen.getByRole('dialog', { name: /Integration Analysis/i })
    const close = screen.getByRole('button', { name: /close/i })
    const continueButton = screen.getByRole('button', { name: /continue with selected/i })
    await waitFor(() => expect(close).toHaveFocus())

    continueButton.focus()
    fireEvent.keyDown(dialog, { key: 'Tab' })
    expect(close).toHaveFocus()

    fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true })
    expect(continueButton).toHaveFocus()
  })

  it('traps focus among enabled controls while an action is submitting', async () => {
    const pending = new Promise<void>(() => {})
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={() => pending} onFinalize={async () => {}} onLetPiDecide={async () => {}}
      onClose={() => {}} countdownSeconds={-1} language="en" open />)

    const dialog = screen.getByRole('dialog', { name: /Integration Analysis/i })
    const close = screen.getByRole('button', { name: /close/i })
    const checkboxes = screen.getAllByRole('checkbox')
    fireEvent.click(checkboxes[0])
    fireEvent.click(screen.getByRole('button', { name: /continue with selected/i }))
    await waitFor(() => expect(screen.getByRole('button', { name: /continue with selected/i })).toBeDisabled())

    checkboxes[1].focus()
    fireEvent.keyDown(dialog, { key: 'Tab' })
    expect(close).toHaveFocus()

    fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true })
    expect(checkboxes[1]).toHaveFocus()
  })
})
