import { describe, it, expect } from 'vitest'
import { resultIdentityKey, roundKey } from '../roundKey'

describe('roundKey', () => {
  it('distinguishes an integration round from a same-numbered normal round', () => {
    expect(roundKey('innovation', 3)).not.toEqual(roundKey('integration', 3))
  })

  it('prefixes integration rounds', () => {
    expect(roundKey('integration', 3)).toBe('int-3')
  })

  it('treats exploration/innovation/empty phase as a plain round key', () => {
    expect(roundKey('innovation', 2)).toBe('r-2')
    expect(roundKey('', 2)).toBe('r-2')
    expect(roundKey(undefined, 2)).toBe('r-2')
  })

  it('supports independent expand state for same-numbered rounds (Set semantics)', () => {
    const expanded = new Set<string>([roundKey('innovation', 3)])
    expect(expanded.has(roundKey('innovation', 3))).toBe(true)
    expect(expanded.has(roundKey('integration', 3))).toBe(false)
  })
})

describe('resultIdentityKey', () => {
  it('uses the backend direction normalization contract for fallback identity', () => {
    expect(resultIdentityKey({
      lineageId: 'main',
      phase: 'innovation',
      round: 6,
      direction: '  Candidate   Axis  ',
    })).toBe('main:innovation:6:candidate axis')
  })

  it('keeps main and restored branches distinct on the same direction and round', () => {
    const main = resultIdentityKey({
      phase: 'innovation',
      round: 6,
      direction: 'Candidate Axis',
    })
    const branch = resultIdentityKey({
      lineageId: 'restore-snapshot-6',
      phase: 'innovation',
      round: 6,
      directionKey: 'candidate axis',
    })
    expect(main).toBe('main:innovation:6:candidate axis')
    expect(branch).toBe('restore-snapshot-6:innovation:6:candidate axis')
    expect(main).not.toBe(branch)
  })
})
