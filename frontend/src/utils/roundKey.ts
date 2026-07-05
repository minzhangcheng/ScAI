/**
 * Stable per-round React key / identity. Integration rounds intentionally reuse
 * the previous round number, so keying by number alone collides with the last
 * innovation round. Qualify integration rounds with an "int-" prefix.
 */
export function roundKey(phase: string | undefined, round: number): string {
  return phase === 'integration' ? `int-${round}` : `r-${round}`
}
