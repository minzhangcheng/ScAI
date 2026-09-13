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
   * Spec B2-3. Resuming onto an earlier generation is the outcome B2-2 chose
   * over re-running from the data, and it is not free: the work between that
   * generation and the interruption is gone. The restored record reads
   * `current` -- the pair holds a generation whose revisions match -- so this
   * is the one disclosure that has to survive a coherent record.
   */
  describe('after a pair restore', () => {
    it('says what continuing from the earlier generation cost', () => {
      render(
        <RuntimePersistenceBanner
          record={{
            coherence: 'current',
            pair_restore: {
              at: '2026-09-08T12:00:00Z',
              restored_revision: 44,
              discarded_revisions: 17,
              discarded_since: '2026-09-08T08:10:19Z',
            },
          }}
        />,
      )

      const text = screen.getByRole('status').textContent ?? ''
      expect(text).toContain('Resumed from the last successful save')
      expect(text).toContain('cannot be recovered')
      expect(text).toContain('17 cell executions discarded')
    })

    it('outranks the running-kernel wording, which would hide it', () => {
      render(
        <RuntimePersistenceBanner
          record={{
            coherence: 'stale',
            last_attempt_status: 'succeeded',
            pair_restore: { restored_revision: 44, discarded_revisions: 3 },
          }}
          sessionStatus="running"
        />,
      )

      const text = screen.getByRole('status').textContent ?? ''
      expect(text).toContain('Resumed from the last successful save')
      expect(text).not.toContain('Analysis in progress')
    })

    it('counts one execution in the singular', () => {
      render(
        <RuntimePersistenceBanner
          record={{ coherence: 'current', pair_restore: { discarded_revisions: 1 } }}
        />,
      )
      expect(screen.getByRole('status').textContent).toContain(
        '1 cell execution discarded',
      )
    })

    it('says nothing about a count it does not have', () => {
      render(
        <RuntimePersistenceBanner
          record={{ coherence: 'current', pair_restore: { restored_revision: 44 } }}
        />,
      )
      expect(screen.getByRole('status').textContent).not.toContain('discarded')
    })

    it('renders in Chinese', () => {
      render(
        <RuntimePersistenceBanner
          record={{
            coherence: 'current',
            pair_restore: { discarded_since: '2026-09-08T08:10:19Z' },
          }}
          isZh
        />,
      )
      const text = screen.getByRole('status').textContent ?? ''
      expect(text).toContain('已从上次成功保存处继续')
      expect(text).toContain('无法找回')
    })
  })

  /** An ordinary live save gap is expected and intentionally silent. */
  describe('while the analysis is still live', () => {
    it('does not claim re-running is required before the first save attempt', () => {
      const { container } = render(
        <RuntimePersistenceBanner
          record={{
            coherence: 'unavailable',
            status: 'live_dirty',
            last_attempt_status: null,
          }}
          sessionStatus="running"
        />,
      )

      expect(container).toBeEmptyDOMElement()
    })

    it('does not report ordinary unsaved work', () => {
      const { container } = render(
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

      expect(container).toBeEmptyDOMElement()
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

    it('stays silent while waiting with the kernel still held', () => {
      const { container, rerender } = render(
        <RuntimePersistenceBanner
          record={{ coherence: 'stale', last_attempt_status: 'succeeded' }}
          sessionStatus="waiting_plan_review"
        />,
      )
      expect(container).toBeEmptyDOMElement()

      rerender(
        <RuntimePersistenceBanner
          record={{ coherence: 'stale', last_attempt_status: 'succeeded' }}
          sessionStatus="generating_report"
        />,
      )
      expect(container).toBeEmptyDOMElement()
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

    it('does not render the ordinary live state in Chinese', () => {
      const { container } = render(
        <RuntimePersistenceBanner
          record={{ coherence: 'stale', last_attempt_status: 'succeeded' }}
          sessionStatus="running"
          isZh
        />,
      )

      expect(container).toBeEmptyDOMElement()
    })
  })
})
