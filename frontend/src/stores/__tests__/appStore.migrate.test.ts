import { describe, it, expect } from 'vitest'
import { migratePersistedState } from '../appStore'

describe('migratePersistedState', () => {
  it('drops volatile runtime state when migrating from v0 (unversioned)', () => {
    const legacy = {
      language: 'zh',
      models: { pi: 'gpt' },
      messages: [{ id: '1', type: 'findings', content: { suggestions: 'legacy-string' } }],
      chatMessages: [{ id: 'c1' }],
      currentSession: { id: 's1', name: 'old' },
    }
    const out = migratePersistedState(legacy, 0) as Record<string, unknown>
    // Volatile server-derived state is cleared (would carry crash-prone shapes)
    expect(out.messages).toBeUndefined()
    expect(out.chatMessages).toBeUndefined()
    expect(out.currentSession).toBeUndefined()
    // User settings are preserved
    expect(out.language).toBe('zh')
    expect(out.models).toEqual({ pi: 'gpt' })
  })

  it('leaves state untouched when already at current version', () => {
    const current = {
      language: 'en',
      messages: [{ id: '1' }],
      currentSession: { id: 's1' },
    }
    const out = migratePersistedState(current, 1) as Record<string, unknown>
    expect(out.messages).toEqual([{ id: '1' }])
    expect(out.currentSession).toEqual({ id: 's1' })
  })

  it('tolerates non-object persisted state', () => {
    expect(migratePersistedState(null, 0)).toEqual({})
    expect(migratePersistedState(undefined, 0)).toEqual({})
  })
})
