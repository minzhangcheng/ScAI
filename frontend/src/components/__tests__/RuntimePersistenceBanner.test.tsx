import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { RuntimePersistenceBanner } from '../runtime/RuntimePersistenceBanner'

/**
 * Spec 5.1's disclosure. A2-3 removed the gate that used to stop the analysis
 * on a failed save; between then and now the record reached the store with
 * nothing rendering it, so a failed save was invisible. This is what closes
 * that gap.
 */
describe('RuntimePersistenceBanner', () => {
  it('says nothing when the committed generation is current', () => {
    const { container } = render(
      <RuntimePersistenceBanner record={{ coherence: 'current' }} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('says nothing when no record has arrived yet', () => {
    const { container } = render(<RuntimePersistenceBanner record={null} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('discloses a stale generation as not promising continuation', () => {
    render(<RuntimePersistenceBanner record={{ coherence: 'stale' }} />)

    const text = screen.getByRole('status').textContent ?? ''
    expect(text).toContain('viewable and exportable')
    expect(text).toContain('not guaranteed')
  })

  it('measures the window from the last successful save, not the last attempt', () => {
    render(
      <RuntimePersistenceBanner
        record={{
          coherence: 'stale',
          exposure_since: '2026-09-06T09:30:00',
          error_summary: 'no space left on device',
        }}
      />,
    )

    const text = screen.getByRole('status').textContent ?? ''
    expect(text).toContain('Last successful save')
    expect(text).toContain('no space left on device')
  })

  it('offers a re-run when nothing has ever committed', () => {
    render(<RuntimePersistenceBanner record={{ coherence: 'unavailable' }} />)

    const text = screen.getByRole('status').textContent ?? ''
    expect(text).toContain('No committed runtime')
    expect(text).toContain('re-running from the data')
  })

  it('treats an inconsistent generation as stale rather than silent', () => {
    render(<RuntimePersistenceBanner record={{ coherence: 'inconsistent' }} />)
    expect(screen.getByRole('status')).toBeTruthy()
  })

  it('renders both locales', () => {
    const { rerender } = render(
      <RuntimePersistenceBanner record={{ coherence: 'stale' }} isZh />,
    )
    expect(screen.getByRole('status').textContent).toContain('结果仍可查看和导出')

    rerender(<RuntimePersistenceBanner record={{ coherence: 'unavailable' }} isZh />)
    expect(screen.getByRole('status').textContent).toContain('重新运行')
  })

  it('ignores an unparseable timestamp rather than printing Invalid Date', () => {
    render(
      <RuntimePersistenceBanner
        record={{ coherence: 'stale', exposure_since: 'not a time' }}
      />,
    )
    expect(screen.getByRole('status').textContent).not.toContain('Invalid Date')
  })

  /**
   * A running kernel is stale by construction. `current_revision` advances on
   * every cell while `committed_revision` only advances at a save point, so
   * between two round-boundary saves the verdict is `stale` continuously --
   * observed on both live sessions of a deployment, `stale` on every sample
   * across four minutes with `last_attempt_status` "succeeded" throughout.
   * Rendering spec 5.1's warning for that made it permanent, and a warning
   * that is always on says nothing. The disclosure stays; its severity does
   * not.
   */
  describe('while the analysis is still live', () => {
    it('reports unsaved work as ordinary progress, not as a warning', () => {
      render(
        <RuntimePersistenceBanner
          record={{
            coherence: 'stale',
            status: 'live_dirty',
            last_attempt_status: 'succeeded',
            exposure_since: '2026-09-08T08:29:33',
          }}
          sessionStatus="running"
        />,
      )

      const text = screen.getByRole('status').textContent ?? ''
      expect(text).not.toContain('not guaranteed')
      expect(text).toContain('Analysis in progress')
      expect(text).toContain('Last successful save')
    })

    it('keeps the warning when the last save attempt failed', () => {
      render(
        <RuntimePersistenceBanner
          record={{ coherence: 'stale', last_attempt_status: 'failed' }}
          sessionStatus="running"
        />,
      )

      expect(screen.getByRole('status').textContent).toContain('not guaranteed')
    })

    it('keeps the warning for a generation that is inconsistent, not merely behind', () => {
      render(
        <RuntimePersistenceBanner
          record={{ coherence: 'inconsistent', last_attempt_status: 'succeeded' }}
          sessionStatus="running"
        />,
      )

      expect(screen.getByRole('status').textContent).toContain('not guaranteed')
    })

    it('covers the states that wait on a user with the kernel still held', () => {
      const { rerender } = render(
        <RuntimePersistenceBanner
          record={{ coherence: 'stale', last_attempt_status: 'succeeded' }}
          sessionStatus="waiting_plan_review"
        />,
      )
      expect(screen.getByRole('status').textContent).toContain('Analysis in progress')

      rerender(
        <RuntimePersistenceBanner
          record={{ coherence: 'stale', last_attempt_status: 'succeeded' }}
          sessionStatus="generating_report"
        />,
      )
      expect(screen.getByRole('status').textContent).toContain('Analysis in progress')
    })

    it('warns once the run is no longer live, pause included', () => {
      for (const sessionStatus of ['paused', 'completed', 'failed', 'stopped', 'pending'] as const) {
        const { unmount } = render(
          <RuntimePersistenceBanner
            record={{ coherence: 'stale', last_attempt_status: 'succeeded' }}
            sessionStatus={sessionStatus}
          />,
        )
        expect(screen.getByRole('status').textContent, sessionStatus).toContain(
          'not guaranteed',
        )
        unmount()
      }
    })

    it('renders the live wording in Chinese', () => {
      render(
        <RuntimePersistenceBanner
          record={{ coherence: 'stale', last_attempt_status: 'succeeded' }}
          sessionStatus="running"
          isZh
        />,
      )

      const text = screen.getByRole('status').textContent ?? ''
      expect(text).toContain('分析进行中')
      expect(text).not.toContain('不保证')
    })
  })
})
