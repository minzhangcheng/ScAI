import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import DirectionEvidence from './DirectionEvidence'
import type { DirectionLiteratureEvidence } from '../ExplorationReportDialog'

const baseEvidence: DirectionLiteratureEvidence = {
  direction_id: 'd1', direction: 'Direction', status: 'completed',
  search_tier: 'standard', confidence: 'low',
}

function showEvidence(overrides: Partial<DirectionLiteratureEvidence>) {
  render(<DirectionEvidence
    candidate={{ direction_id: 'd1', direction: 'Direction' }}
    evidence={{ ...baseEvidence, ...overrides }}
    expanded onToggle={vi.fn()} language="en"
  />)
}

describe('DirectionEvidence completion and retained evidence', () => {
  it.each([
    ['direction_search_v1', 'Search workflow completed', 'Evidence verified'],
    [undefined, 'Evidence verified', 'Search workflow completed'],
  ])('labels completion according to policy %s', (policy, label, otherLabel) => {
    showEvidence({ policy_version: policy })
    expect(screen.getByText(label!)).toBeInTheDocument()
    expect(screen.queryByText(otherLabel!)).not.toBeInTheDocument()
  })

  it.each([false, true])('discloses timeout with retained references: %s', retained => {
    showEvidence({
      status: 'partial', timed_out: true, direct_evidence_count: retained ? 1 : 0,
      direct_evidence: retained ? [{
        pmid: '12345678', title: 'Retained source', relevance: 'high',
        supports: 'direct', why_it_matters: 'Retained before timeout.',
      }] : [],
    })
    expect(screen.getByText(retained ? 'Partial evidence' : 'Search timed out')).toBeInTheDocument()
    expect(screen.queryByText(retained ? 'Search timed out' : 'Partial evidence')).not.toBeInTheDocument()
    expect(screen.getByText(retained
      ? /references below are partial/
      : /timed out before references could be retained/)).toBeInTheDocument()
    expect(screen.getByText(`Direct: ${retained ? 1 : 0}`)).toBeInTheDocument()
    if (retained) expect(screen.getByText(/PMID: 12345678/)).toBeInTheDocument()
    else expect(screen.queryByText(/PMID:/)).not.toBeInTheDocument()
  })
})
