import { describe, expect, it } from 'vitest'

import { reportBranchQuery, withReportBranch } from './reportBranch'

describe('report branch URLs', () => {
  it('encodes branch ids and omits missing ids', () => {
    expect(reportBranchQuery('restore/a b')).toBe('?branch_id=restore%2Fa%20b')
    expect(reportBranchQuery()).toBe('')
    expect(reportBranchQuery(null)).toBe('')
  })

  it('preserves existing queries and does not duplicate branch_id', () => {
    expect(withReportBranch('/report?download=1', 'restore/a')).toBe(
      '/report?download=1&branch_id=restore%2Fa',
    )
    expect(withReportBranch('/report?branch_id=existing', 'restore/a')).toBe(
      '/report?branch_id=existing',
    )
    expect(withReportBranch('/report', undefined)).toBe('/report')
  })
})
