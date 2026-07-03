import { describe, it, expect } from 'vitest'
import { buildDirectionsPayload } from '../AnalysisPanel'

describe('buildDirectionsPayload', () => {
  it('includes is_custom for each direction', () => {
    const out = buildDirectionsPayload([
      { direction: 'A', notes: 'n', isCustom: false },
      { direction: 'B', notes: '', isCustom: true },
    ])
    expect(out).toEqual([
      { direction: 'A', notes: 'n', is_custom: false },
      { direction: 'B', notes: '', is_custom: true },
    ])
  })
})
