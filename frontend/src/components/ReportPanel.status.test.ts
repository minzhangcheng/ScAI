import { describe, expect, it } from 'vitest'

import {
  partialReportNotice,
  reportBranchQuery,
  reportCompletionContent,
  type DetailedReport,
} from './ReportPanel'

function report(overrides: Partial<DetailedReport> = {}): DetailedReport {
  return {
    id: 'report-1',
    session_id: 'session-1',
    title: 'Report',
    abstract: '',
    sections: [],
    conclusions: '',
    figures: [],
    tables: [],
    generated_at: '2026-08-02T00:00:00Z',
    status: 'partial',
    progress: 100,
    current_step: 'Partial report approved',
    ...overrides,
  }
}

describe('ReportPanel partial report status', () => {
  it('encodes the restored lineage in every report endpoint query', () => {
    expect(reportBranchQuery('branch/A B')).toBe('?branch_id=branch%2FA%20B')
    expect(reportBranchQuery()).toBe('')
  })

  it('preserves a partial terminal status in the websocket message', () => {
    expect(reportCompletionContent(report(), 'en')).toEqual({
      status: 'partial',
      message: 'Report generated (partial)',
    })
  })

  it('does not mislabel an approved partial report as a figure gap', () => {
    const notice = partialReportNotice(report({
      partial_reason: 'approved_partial',
    }), 'en')

    expect(notice).toContain('approved partial result scope')
    expect(notice).not.toContain('figure coverage')
  })

  it('identifies both kinds of figure coverage gap', () => {
    const notice = partialReportNotice(report({
      partial_reason: 'figure_coverage',
      figureless_sections: ['direction_2'],
      unreported_figures: ['missing.png'],
    }), 'en')

    expect(notice).toContain('sections without result figures: direction_2')
    expect(notice).toContain('figures not included in the report: missing.png')
  })

  it('names the planned sections a report was published without', () => {
    const notice = partialReportNotice(report({
      partial_reason: 'section_coverage',
      missing_sections: ['direction_2', 'integrated_findings'],
      figureless_sections: ['direction_1'],
    }), 'en')

    expect(notice).toContain('absent sections: direction_2, integrated_findings')
    expect(notice).toContain('sections without result figures: direction_1')
  })

  it('labels a degraded fallback independently from figure coverage', () => {
    const notice = partialReportNotice(report({
      partial_reason: 'degraded_fallback',
      figureless_sections: [],
      unreported_figures: [],
    }), 'en')

    expect(notice).toContain('degraded fallback report')
    expect(notice).not.toContain('figure coverage')
  })
})
