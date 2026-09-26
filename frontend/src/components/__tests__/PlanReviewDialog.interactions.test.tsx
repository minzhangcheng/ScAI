import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PlanReviewDialog, type AnalysisPlan } from '../PlanReviewDialog'
import { useAppStore } from '../../stores/appStore'

function plan(status = 'pending', stepStatus = 'pending'): AnalysisPlan {
  return {
    id: 'plan-1',
    phase: 'exploration',
    stage: 'stage1',
    overview: 'Plan overview',
    steps: [{
      id: 1,
      title: 'Quality control',
      method: 'scanpy',
      input: 'adata',
      expected_output: 'QC metrics',
      rationale: 'Establish data quality',
      depends_on: [],
      status: stepStatus,
    }],
    key_decision_points: [],
    expected_runtime: '5 minutes',
    status,
  }
}

const baseProps = {
  open: true,
  dialogueHtml: '<p>dialogue</p>',
  phase: 'exploration',
  round: 1,
  stage: 'stage1',
  onApprove: vi.fn().mockResolvedValue(undefined),
  onSkip: vi.fn().mockResolvedValue(undefined),
  onModify: vi.fn().mockResolvedValue(undefined),
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('PlanReviewDialog interactions', () => {

  it('resets the countdown when the plan fingerprint changes', async () => {
    vi.useFakeTimers()
    useAppStore.setState({ language: 'en' })
    const onSkip = vi.fn().mockResolvedValue(undefined)
    const { rerender } = render(
      <PlanReviewDialog
        {...baseProps}
        plan={plan('pending', 'pending')}
        countdownSeconds={2}
        onSkip={onSkip}
      />,
    )

    await act(async () => { vi.advanceTimersByTime(1000) })
    expect(onSkip).not.toHaveBeenCalled()

    rerender(
      <PlanReviewDialog
        {...baseProps}
        plan={plan('revised', 'completed')}
        countdownSeconds={2}
        onSkip={onSkip}
      />,
    )
    expect(screen.getByText('0:02')).toBeInTheDocument()
    await act(async () => { vi.advanceTimersByTime(1000) })
    expect(onSkip).not.toHaveBeenCalled()
  })

  it('auto-skips when a fresh countdown expires', async () => {
    vi.useFakeTimers()
    useAppStore.setState({ language: 'en' })
    const onSkip = vi.fn().mockResolvedValue(undefined)
    render(
      <PlanReviewDialog
        {...baseProps}
        plan={plan()}
        countdownSeconds={1}
        onSkip={onSkip}
      />,
    )

    await act(async () => {
      vi.advanceTimersByTime(1500)
      await Promise.resolve()
    })
    expect(onSkip).toHaveBeenCalledTimes(1)
  })
})
