import { afterEach, it, expect, vi } from 'vitest'
import { partialize } from '../appStore'

afterEach(() => {
  vi.restoreAllMocks()
})

it('does not persist api keys (nested or top-level)', () => {
  const state = {
    apiConfig: { apiKey: 'sk-1', llmBaseUrl: 'http://x', proxyUrl: '' },
    deepResearchApiKey: 'sk-2',
    googleApiKey: 'sk-3',
    language: 'en',
    prompts: {},
    models: {},
    temperatures: {},
    maxRounds: 5,
    roundWaitSeconds: 300,
    agentServerConfigs: {},
    agentVisionSettings: {},
    literatureMode: 'disabled',
    deepResearchModel: '',
    deepResearchLlmBaseUrl: '',
    pubmedEmail: '',
    literatureMaxSearchRounds: 3,
    messagePanelWidth: 400,
    chatMessages: [],
    messages: [{ id: '1', type: 'findings', content: { suggestions: 'x' } }],
  } as any
  const persisted = partialize(state)
  expect(persisted.apiConfig?.apiKey ?? '').toBe('')
  expect((persisted as any).deepResearchApiKey).toBeUndefined()
  expect((persisted as any).googleApiKey).toBeUndefined()
})

it('does not persist realtime messages (server-authoritative, re-hydrated from /load)', () => {
  const state = {
    apiConfig: { apiKey: '', llmBaseUrl: '', proxyUrl: '' },
    chatMessages: [{ id: 'c1', figures: [] }],
    messages: [{ id: '1', type: 'findings', content: {} }],
    currentSession: { id: 's1' },
  } as any
  const persisted = partialize(state) as any
  // Realtime messages must NOT be persisted; they come back from the server.
  expect(persisted.messages).toBeUndefined()
  // Chat messages and session are still persisted (different restore sources).
  expect(persisted.chatMessages).toHaveLength(1)
  expect(persisted.currentSession).toEqual({ id: 's1' })
})

it('does not persist a theme override because color scheme follows the browser', () => {
  const persisted = partialize({
    apiConfig: { apiKey: '', llmBaseUrl: '', proxyUrl: '' },
    chatMessages: [],
    theme: 'dark',
    colorScheme: 'light',
  } as any) as any

  expect(persisted.theme).toBeUndefined()
  expect(persisted.colorScheme).toBeUndefined()
})

it('omits authoritative and image-heavy transient state from the local cache', () => {
  const state = {
    apiConfig: { apiKey: '', llmBaseUrl: '', proxyUrl: '' },
    chatMessages: [{ id: 'c1', figures: [{ name: 'plot', base64: 'encoded-image' }] }],
    messages: [{ id: '1', type: 'figure', content: { image: 'encoded-image' } }],
    currentSession: {
      id: 's1',
      curationReviewContext: { image_base64: 'encoded-review-image' },
    },
  } as any

  const persisted = partialize(state) as any
  expect(persisted.messages).toBeUndefined()
  expect(persisted.currentSession.curationReviewContext).toBeUndefined()
  expect(persisted.chatMessages[0].figures[0].base64).toBeUndefined()
})

it('swallows quota failures from the persistence adapter', async () => {
  const { quotaSafeLocalStorage } = await import('../persistence')
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
  const storage = quotaSafeLocalStorage as Storage
  const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new DOMException('quota', 'QuotaExceededError')
  })

  expect(() => storage.setItem('scai-test', 'value')).not.toThrow()
  setItem.mockRestore()
  warn.mockRestore()
})
