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
})
