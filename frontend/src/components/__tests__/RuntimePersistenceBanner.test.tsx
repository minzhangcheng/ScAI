import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import {
  RuntimePersistenceBanner,
  type RuntimePersistenceRecord,
} from '../runtime/RuntimePersistenceBanner'

const cases: Array<{
  label: string
  record: RuntimePersistenceRecord
  status?: string
  warning: string | null
}> = [
  { label: 'current commit', record: { coherence: 'current' }, warning: null },
  { label: 'no commit', record: { coherence: 'unavailable' }, warning: 'No committed runtime' },
  { label: 'live work before the first save', status: 'running', warning: null,
    record: { coherence: 'unavailable', status: 'live_dirty', last_attempt_status: null } },
  { label: 'failed live save', status: 'running', warning: 'not guaranteed',
    record: { coherence: 'stale', last_attempt_status: 'failed' } },
  { label: 'inconsistent generation', status: 'running', warning: 'not guaranteed',
    record: { coherence: 'inconsistent', last_attempt_status: 'succeeded' } },
  ...['paused', 'completed', 'failed', 'stopped', 'pending'].map(status => ({
    label: `stale ${status} session`, status, warning: 'not guaranteed',
    record: { coherence: 'stale', last_attempt_status: 'succeeded' },
  })),
]

describe('Runtime persistence disclosure', () => {
  it.each(cases)('discloses $label', ({ record, status, warning }) => {
    const { container } = render(
      <RuntimePersistenceBanner record={record} sessionStatus={status} />,
    )
    if (warning === null) expect(container).toBeEmptyDOMElement()
    else expect(screen.getByRole('status')).toHaveTextContent(warning)
  })

  it('measures exposure from the successful commit and preserves the save error', () => {
    const { rerender } = render(<RuntimePersistenceBanner record={{
      coherence: 'stale', exposure_since: '2026-09-06T09:30:00',
      error_summary: 'no space left on device',
    }} />)
    expect(screen.getByRole('status')).toHaveTextContent('Last successful save')
    expect(screen.getByRole('status')).toHaveTextContent('no space left on device')
    rerender(<RuntimePersistenceBanner record={{
      coherence: 'stale', exposure_since: 'not a time',
    }} />)
    expect(screen.getByRole('status')).not.toHaveTextContent('Invalid Date')
  })

  it.each(['current', 'stale'])('discloses discarded work despite %s coherence', coherence => {
    render(<RuntimePersistenceBanner sessionStatus="running" record={{
      coherence, last_attempt_status: 'succeeded',
      pair_restore: { restored_revision: 44, discarded_revisions: 17 },
    }} />)
    const status = screen.getByRole('status')
    expect(status).toHaveTextContent('Resumed from the last successful save')
    expect(status).toHaveTextContent('cannot be recovered')
    expect(status).toHaveTextContent('17 cell executions discarded')
    expect(status).not.toHaveTextContent('Analysis in progress')
  })
})
