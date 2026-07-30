import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SnapshotFailureDialog } from '../snapshots/SnapshotFailureDialog'
import { RuntimeChoiceDialog } from '../snapshots/RuntimeChoiceDialog'
import { RuntimeWarningBanner } from '../snapshots/RuntimeWarningBanner'
import type { SnapshotFailureInfo } from '../../types/snapshotV2'

const baseFailure: SnapshotFailureInfo = {
  failure: {
    snapshot_id: 'snap-1',
    session_id: 'sess-1',
    failed_at: '2025-01-01T00:00:00Z',
    code: 'required_runtime_unpersistable',
    message: "Required variable 'adata' could not be serialized",
    details: {
      name: 'adata',
      type: 'anndata._core.anndata.AnnData',
      serializer: 'h5ad',
      required_variables: ['adata', 'obs_meta'],
    },
  },
  allowed_actions: ['retry', 'manage', 'terminate'],
}

describe('SnapshotFailureDialog', () => {
  it('renders backend error code; Skip hidden unless allowed_actions includes it', () => {
    render(
      <SnapshotFailureDialog
        failure={baseFailure}
        onRetry={vi.fn()}
        onProceed={vi.fn()}
        onSkip={vi.fn()}
        onManage={vi.fn()}
        onTerminate={vi.fn()}
      />,
    )

    expect(screen.getByRole('dialog', { name: /snapshot failure/i })).toBeInTheDocument()
    expect(screen.getByText(/required_runtime_unpersistable/i)).toBeInTheDocument()
    // baseFailure has base actions (no skip) → no Skip button.
    expect(screen.queryByRole('button', { name: /skip snapshot/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /manage/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /terminate/i })).toBeInTheDocument()
  })

  it('offers a Skip button when allowed_actions includes skip', () => {
    const onSkip = vi.fn()
    render(
      <SnapshotFailureDialog
        failure={{ ...baseFailure, allowed_actions: ['retry', 'skip', 'manage', 'terminate'] }}
        onRetry={vi.fn()}
        onProceed={vi.fn()}
        onSkip={onSkip}
        onManage={vi.fn()}
        onTerminate={vi.fn()}
      />,
    )
    const skipBtn = screen.getByRole('button', { name: /skip snapshot/i })
    fireEvent.click(skipBtn)
    expect(onSkip).toHaveBeenCalledTimes(1)
  })

  it('shows named required-variable diagnostics from failure details', () => {
    render(
      <SnapshotFailureDialog
        failure={baseFailure}
        onRetry={vi.fn()}
        onProceed={vi.fn()}
        onSkip={vi.fn()}
        onManage={vi.fn()}
        onTerminate={vi.fn()}
      />,
    )

    expect(screen.getByText('adata', { selector: 'code' })).toBeInTheDocument()
    expect(screen.getByText('obs_meta', { selector: 'code' })).toBeInTheDocument()
  })

  it('retry does not regenerate reports — only re-enters capture', () => {
    const onRetry = vi.fn()
    render(
      <SnapshotFailureDialog
        failure={{
          ...baseFailure,
          failure: {
            ...baseFailure.failure,
            code: 'post_report_capture_failed',
            message: 'Terminal snapshot failed after report',
            details: { phase: 'post_report', regenerate_reports: false },
          },
        }}
        onRetry={onRetry}
        onProceed={vi.fn()}
        onSkip={vi.fn()}
        onManage={vi.fn()}
        onTerminate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /retry/i }))
    expect(onRetry).toHaveBeenCalledTimes(1)
    expect(onRetry).toHaveBeenCalledWith(
      expect.objectContaining({ regenerateReports: false }),
    )
    expect(screen.getByText(/does not regenerate|capture only|re-enter capture/i)).toBeInTheDocument()
  })

  it('collapses a huge traceback message behind a details disclosure while keeping actions reachable', () => {
    const summary =
      "required_runtime_unpersistable: AnnData variable 'dds' write failed"
    const hugeMessage = summary + '\n' + Array.from({ length: 400 }, (_, i) =>
      `  File "/dep/venv/lib/python3.11/site-packages/anndata/_io/specs/registry.py", line ${i}, in write_elem`,
    ).join('\n')
    render(
      <SnapshotFailureDialog
        failure={{
          ...baseFailure,
          failure: { ...baseFailure.failure, message: hugeMessage },
        }}
        onRetry={vi.fn()}
        onProceed={vi.fn()}
        onSkip={vi.fn()}
        onManage={vi.fn()}
        onTerminate={vi.fn()}
      />,
    )

    // First line is shown as a summary; full traceback is tucked into a
    // collapsible <details>, so the dialog does not expand past the viewport.
    expect(screen.getByText(summary)).toBeInTheDocument()
    const disclosure = screen.getByText(/technical details/i)
    expect(disclosure.tagName.toLowerCase()).toBe('summary')
    // Action buttons remain rendered (not pushed off-screen by the message).
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /terminate/i })).toBeInTheDocument()
  })

  it('discloses pre_restore context when failure phase is pre_restore', () => {
    render(
      <SnapshotFailureDialog
        failure={{
          ...baseFailure,
          failure: {
            ...baseFailure.failure,
            code: 'pre_restore_capture_failed',
            message: 'Could not capture pre-restore snapshot',
            details: { phase: 'pre_restore' },
          },
        }}
        onRetry={vi.fn()}
        onProceed={vi.fn()}
        onSkip={vi.fn()}
        onManage={vi.fn()}
        onTerminate={vi.fn()}
      />,
    )

    const dialog = screen.getByRole('dialog', { name: /snapshot failure/i })
    expect(dialog.textContent).toMatch(/pre[_ -]?restore/i)
    expect(dialog.textContent).toMatch(
      /current workflow remains|no live mutation|before.*mutation/i,
    )
  })

  it('memory gate offers Proceed and Skip but never Retry', () => {
    // Retry is absent by construction: the preflight is deterministic, so a
    // retry would measure the same objects and refuse again forever.
    const onProceed = vi.fn()
    render(
      <SnapshotFailureDialog
        failure={{
          failure: {
            ...baseFailure.failure,
            code: 'snapshot_memory_insufficient',
            message:
              "Snapshot capture needs about 22.0 GiB to copy 'adata' (20.0 GiB) "
              + 'before writing it, but only 10.0 GiB of memory headroom is available',
            details: { name: 'adata', phase: 'pre_execution_stage2' },
          },
          allowed_actions: ['proceed', 'skip', 'terminate'],
        }}
        onRetry={vi.fn()}
        onProceed={onProceed}
        onSkip={vi.fn()}
        onManage={vi.fn()}
        onTerminate={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: /^retry$/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /manage/i })).toBeNull()
    const proceed = screen.getByRole('button', { name: /capture anyway/i })
    fireEvent.click(proceed)
    expect(onProceed).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: /skip snapshot/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /terminate/i })).toBeTruthy()
  })

  it('memory gate explains that the analysis itself was not interrupted', () => {
    render(
      <SnapshotFailureDialog
        failure={{
          failure: { ...baseFailure.failure, code: 'snapshot_memory_insufficient' },
          allowed_actions: ['proceed', 'skip', 'terminate'],
        }}
        onRetry={vi.fn()}
        onProceed={vi.fn()}
        onSkip={vi.fn()}
        onManage={vi.fn()}
        onTerminate={vi.fn()}
      />,
    )

    const dialog = screen.getByRole('dialog', { name: /snapshot failure/i })
    expect(dialog.textContent).toMatch(/only the snapshot was refused/i)
    expect(dialog.textContent).not.toMatch(/Retry re-enters capture only/i)
  })
})

describe('RuntimeChoiceDialog', () => {
  it('requires explicit reset or keep — no skip and no silent default', () => {
    const onChoose = vi.fn()
    render(
      <RuntimeChoiceDialog
        open
        runtimeStatus="missing"
        onChoose={onChoose}
        onAbort={vi.fn()}
      />,
    )

    expect(screen.getByRole('dialog', { name: /runtime/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^skip$/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /reset|empty kernel/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /keep/i })).toBeInTheDocument()
  })

  it('keep choice uses risk language', () => {
    render(
      <RuntimeChoiceDialog
        open
        runtimeStatus="corrupt"
        onChoose={vi.fn()}
        onAbort={vi.fn()}
      />,
    )

    expect(screen.getByRole('dialog', { name: /runtime/i }).textContent).toMatch(
      /risk|inconsistent|not a complete restore/i,
    )
  })
})

describe('RuntimeWarningBanner', () => {
  it('shows persistent keep warning when status is intentionally_kept', () => {
    render(
      <RuntimeWarningBanner
        runtimeRestoreStatus="intentionally_kept"
        warning="Live kernel retained after restore; runtime may be inconsistent with restored state."
      />,
    )

    const status = screen.getByRole('status')
    expect(status).toBeInTheDocument()
    expect(status.textContent).toMatch(/intentionally kept|inconsistent|risk/i)
  })

  it('shows read-only / re-run guidance when status is unavailable', () => {
    render(
      <RuntimeWarningBanner
        runtimeRestoreStatus="unavailable"
        warning={null}
      />,
    )

    const status = screen.getByRole('status')
    expect(status.textContent).toMatch(/read-only|re-?run|blocked/i)
    expect(screen.queryByText(/intentionally kept/i)).not.toBeInTheDocument()
  })

  it('renders nothing when status is complete', () => {
    const { container } = render(
      <RuntimeWarningBanner runtimeRestoreStatus="complete" warning={null} />,
    )
    expect(container).toBeEmptyDOMElement()
  })
})
