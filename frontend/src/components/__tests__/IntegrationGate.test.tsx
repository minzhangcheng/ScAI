import { act, render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { IntegrationGatePanel } from '../IntegrationGatePanel'

const candidates = [
  { direction: 'New direction A', direction_id: 'int_001' },
  { direction: 'New direction B', direction_id: 'int_002' },
]

describe('IntegrationGatePanel', () => {
  it('renders candidates and both actions', () => {
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={async () => {}} onLetPiDecide={async () => {}} countdownSeconds={-1} language="zh" />)
    expect(screen.getByText(/New direction A/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /撰写|report/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /PI 决定|PI decide/i })).toBeInTheDocument()
  })

  it('calls onFinalize on write-report click', () => {
    const onFinalize = vi.fn()
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={onFinalize} onLetPiDecide={async () => {}} countdownSeconds={-1} language="zh" />)
    fireEvent.click(screen.getByRole('button', { name: /撰写|report/i }))
    expect(onFinalize).toHaveBeenCalled()
  })

  it('calls onLetPiDecide when PI decide button clicked', () => {
    const onLetPiDecide = vi.fn()
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={async () => {}} onLetPiDecide={onLetPiDecide} countdownSeconds={-1} language="zh" />)
    fireEvent.click(screen.getByRole('button', { name: /PI 决定|PI decide/i }))
    expect(onLetPiDecide).toHaveBeenCalled()
  })

  it('selection resets when candidates change across cycles', () => {
    const onSubmit = vi.fn()
    const c1 = [{ direction: 'A' }, { direction: 'B' }]
    const { rerender } = render(
      <IntegrationGatePanel cycle={1} candidates={c1} onSubmit={onSubmit} onFinalize={async () => {}} onLetPiDecide={async () => {}} countdownSeconds={-1} language="zh" />
    )
    fireEvent.click(screen.getAllByRole('checkbox')[0])

    const c2 = [{ direction: 'C' }, { direction: 'D' }]
    rerender(
      <IntegrationGatePanel cycle={2} candidates={c2} onSubmit={onSubmit} onFinalize={async () => {}} onLetPiDecide={async () => {}} countdownSeconds={-1} language="zh" />
    )
    screen.getAllByRole('checkbox').forEach((cb) => expect(cb).not.toBeChecked())
  })

  it('renders report link when reportUrl is provided', () => {
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={async () => {}} onLetPiDecide={async () => {}} countdownSeconds={-1} language="en"
      reportUrl="/api/session/s/integration-report/html" />)
    expect(screen.getByText(/View Integration Report/i)).toBeInTheDocument()
  })

  it('does not render report link when reportUrl is undefined', () => {
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={async () => {}} onLetPiDecide={async () => {}} countdownSeconds={-1} language="en" />)
    expect(screen.queryByText(/View Integration Report/i)).not.toBeInTheDocument()
  })

  it('renders English text when language is en', () => {
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={async () => {}} onLetPiDecide={async () => {}} countdownSeconds={-1} language="en" />)
    expect(screen.getByRole('button', { name: /write report/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /let PI decide/i })).toBeInTheDocument()
    expect(screen.getByText(/Integration Analysis/i)).toBeInTheDocument()
  })

  it('rejects an empty selection locally without calling onSubmit', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={onSubmit} onFinalize={async () => {}} onLetPiDecide={async () => {}}
      countdownSeconds={-1} language="en" />)

    fireEvent.click(screen.getByRole('button', { name: /continue with selected/i }))

    expect(await screen.findByText(/Select at least one research direction/i)).toBeInTheDocument()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('keeps the gate open and shows the rejected submit error', async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error('Session is not waiting for input'))
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={onSubmit} onFinalize={async () => {}} onLetPiDecide={async () => {}}
      countdownSeconds={-1} language="en" />)

    fireEvent.click(screen.getAllByRole('checkbox')[0])
    fireEvent.click(screen.getByRole('button', { name: /continue with selected/i }))

    expect(await screen.findByText(/Action failed: Session is not waiting for input/i)).toBeInTheDocument()
    expect(screen.getByText(/New direction A/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /continue with selected/i })).not.toBeDisabled()
    })
  })

  it.each([
    ['Let PI Decide', 'onLetPiDecide'],
    ['Write Report', 'onFinalize'],
  ] as const)('shows rejected %s errors without dismissing the gate', async (buttonName, failingProp) => {
    const reject = vi.fn().mockRejectedValue(new Error(`${buttonName} rejected`))
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}}
      onFinalize={failingProp === 'onFinalize' ? reject : async () => {}}
      onLetPiDecide={failingProp === 'onLetPiDecide' ? reject : async () => {}}
      countdownSeconds={-1} language="en" />)

    fireEvent.click(screen.getByRole('button', { name: new RegExp(buttonName, 'i') }))

    expect(await screen.findByText(new RegExp(`Action failed: ${buttonName} rejected`, 'i'))).toBeInTheDocument()
    expect(screen.getByText(/Integration Analysis/i)).toBeInTheDocument()
  })

  it('does not retry a rejected no-wait PI decision, but lets the user retry manually', async () => {
    const onLetPiDecide = vi.fn().mockRejectedValue(new Error('upstream unavailable'))
    render(<IntegrationGatePanel cycle={1} candidates={candidates}
      onSubmit={async () => {}} onFinalize={async () => {}} onLetPiDecide={onLetPiDecide}
      countdownSeconds={0} language="en" />)

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
})
