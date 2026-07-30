import { describe, expect, it } from 'vitest'
import { resolveSnapshotFailureActions } from '../snapshotV2'

/**
 * The parser whitelists action names, so an action the backend starts sending
 * but this list does not know is dropped silently — the button never renders and
 * TypeScript cannot catch it, because the filter's type predicate is hand-written.
 * These tests pin every action the backend can emit.
 */
describe('resolveSnapshotFailureActions', () => {
  it('preserves proceed from a memory-preflight gate', () => {
    expect(
      resolveSnapshotFailureActions({
        code: 'snapshot_memory_insufficient',
        allowed_actions: ['proceed', 'skip', 'terminate'],
      }),
    ).toEqual(['proceed', 'skip', 'terminate'])
  })

  it('preserves proceed when the gate forbids skipping', () => {
    expect(
      resolveSnapshotFailureActions({
        code: 'snapshot_memory_insufficient',
        allowed_actions: ['proceed', 'terminate'],
      }),
    ).toEqual(['proceed', 'terminate'])
  })

  it('preserves the ordinary action set unchanged', () => {
    expect(
      resolveSnapshotFailureActions({
        allowed_actions: ['retry', 'skip', 'manage', 'terminate'],
      }),
    ).toEqual(['retry', 'skip', 'manage', 'terminate'])
  })

  it('still drops names that are not real actions', () => {
    expect(
      resolveSnapshotFailureActions({
        allowed_actions: ['retry', 'nonsense', 42, null],
      }),
    ).toEqual(['retry'])
  })

  it('falls back to producer-derived actions when allowed_actions is absent', () => {
    expect(
      resolveSnapshotFailureActions({
        retry_context: { producer: 'workflow_boundary' },
      }),
    ).toEqual(['retry', 'skip', 'manage', 'terminate'])
    expect(
      resolveSnapshotFailureActions({
        retry_context: { producer: 'pre_restore' },
      }),
    ).toEqual(['retry', 'manage', 'terminate'])
  })
})
