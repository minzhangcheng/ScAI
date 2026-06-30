import { describe, it, expect } from 'vitest'
import { transformRoundResults } from '../transformLoadResponse'

describe('cross-modal candidate preservation', () => {
  it('preserves cross_modal_candidates through transformRoundResults', () => {
    const raw = [{
      round: 2,
      pi_summary: 's',
      cross_modal_candidates: [{
        theme: 'iron injury',
        mechanism_statement: 'm',
        confidence: 'medium' as const,
        relation_to_current_direction: 'uncovered' as const,
        report_recommendation: 'secondary' as const,
      }],
    }]
    const out = transformRoundResults(raw)
    expect(out[0].crossModalCandidates).toBeDefined()
    expect(out[0].crossModalCandidates![0].theme).toBe('iron injury')
  })

  it('defaults to empty array when absent', () => {
    const out = transformRoundResults([{ round: 1, pi_summary: 's' }])
    expect(out[0].crossModalCandidates).toEqual([])
  })
})
