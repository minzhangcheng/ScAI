/**
 * Stable per-round React key / identity. Integration rounds intentionally reuse
 * the previous round number, so keying by number alone collides with the last
 * innovation round. Qualify integration rounds with an "int-" prefix.
 */
export function roundKey(phase: string | undefined, round: number): string {
  return phase === 'integration' ? `int-${round}` : `r-${round}`
}

export interface ResultIdentityFields {
  resultIdentityKey?: string
  result_identity_key?: string
  lineageId?: string
  lineage_id?: string
  phase?: string
  round?: number
  roundNumber?: number
  round_number?: number
  directionKey?: string
  direction_key?: string
  direction?: string
}

function normalizedDirection(value: unknown): string {
  return String(value || '').trim().toLowerCase().replace(/\s+/g, ' ')
}

/** Canonical round identity shared by reload, WebSocket upsert and React keys. */
export function resultIdentityKey(result: ResultIdentityFields): string {
  const explicit = result.resultIdentityKey || result.result_identity_key
  if (explicit) return explicit
  const lineage = result.lineageId || result.lineage_id || 'main'
  const phase = result.phase || 'exploration'
  const round = result.round ?? result.roundNumber ?? result.round_number ?? 0
  const direction = normalizedDirection(
    result.directionKey || result.direction_key || result.direction,
  )
  return `${lineage}:${phase}:${round}:${direction}`
}
