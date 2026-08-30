import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearToken, setToken } from '../../utils/authStorage'
import { PERSIST_STORAGE_KEY, PERSIST_VERSION } from '../../constants/storage'
import { migratePersistedState, partialize, useAppStore } from '../appStore'

const AGENTS = [
  'pi',
  'pathology',
  'bioinformatics',
  'biochemistry',
  'literature',
  'painter',
  'supervisor',
  'technique',
] as const

function canaryConfigs() {
  return Object.fromEntries(
    AGENTS.map((agent) => [
      agent,
      {
        llmBaseUrl: `https://${agent}.example.test/v1`,
        apiKey: `agent-secret-${agent}`,
        contextWindow: 128000,
      },
    ]),
  )
}

function seedSecrets() {
  const configs = canaryConfigs()
  useAppStore.setState({
    agentServerConfigs: configs as never,
    apiConfig: {
      ...useAppStore.getState().apiConfig,
      apiKey: 'global-installation-secret',
    },
    deepResearchApiKey: 'deep-research-secret',
    googleApiKey: 'google-secret',
    authReady: false,
    authUser: null,
  })
  return configs
}

function readPersistedState(): Record<string, any> {
  const raw = localStorage.getItem(PERSIST_STORAGE_KEY)
  expect(raw).not.toBeNull()
  return JSON.parse(raw!).state ?? {}
}

function expectNoCanaries(value: unknown) {
  const serialized = JSON.stringify(value)
  for (const agent of AGENTS) {
    expect(serialized).not.toContain(`agent-secret-${agent}`)
  }
  expect(serialized).not.toContain('global-installation-secret')
  expect(serialized).not.toContain('deep-research-secret')
  expect(serialized).not.toContain('google-secret')
}

function expectEndpointHints(value: Record<string, any>) {
  for (const agent of AGENTS) {
    expect(value[agent]).toEqual({
      llmBaseUrl: `https://${agent}.example.test/v1`,
      apiKey: '',
      contextWindow: 128000,
    })
  }
}

function writeEnvelope(version: number, state: Record<string, unknown>) {
  localStorage.setItem(
    PERSIST_STORAGE_KEY,
    JSON.stringify({ state, version }),
  )
}

async function expectSignedOutSecretsCleared(action: () => void | Promise<void>) {
  seedSecrets()
  await action()

  const state = useAppStore.getState()
  expect(state.agentServerConfigs).toEqual({})
  expect(state.apiConfig.apiKey).toBe('')
  expect(state.deepResearchApiKey).toBe('')
  expect(state.googleApiKey).toBe('')
  expectNoCanaries(state)
  expectNoCanaries(readPersistedState())
}

beforeEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
  clearToken()
  useAppStore.setState({
    agentServerConfigs: {},
    apiConfig: { ...useAppStore.getState().apiConfig, apiKey: '' },
    deepResearchApiKey: '',
    googleApiKey: '',
    authReady: false,
    authUser: null,
  })
})

afterEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
  clearToken()
})

describe('agent server credential persistence boundaries', () => {
  it('scrubs every per-agent key at partialize and localStorage write', () => {
    const configs = seedSecrets()
    const persisted = partialize({
      ...useAppStore.getState(),
      agentServerConfigs: configs,
    } as any) as any

    expectNoCanaries(persisted)
    expectEndpointHints(persisted.agentServerConfigs)

    expectNoCanaries(readPersistedState())
    expectEndpointHints(readPersistedState().agentServerConfigs)
  })

  it.each([0, 1, 2])('scrubs migration version %s while preserving endpoint hints', (version) => {
    const migrated = migratePersistedState(
      { agentServerConfigs: canaryConfigs() },
      version,
    ) as Record<string, any>

    expectNoCanaries(migrated)
    expectEndpointHints(migrated.agentServerConfigs)
  })

  it('scrubs malformed migration input and uses persisted version 2', () => {
    const migrated = migratePersistedState(
      {
        agentServerConfigs: {
          pi: null,
          pathology: { llmBaseUrl: 42, apiKey: 'agent-secret-pathology' },
          unknown: { llmBaseUrl: 'https://unknown.example.test', apiKey: 'agent-secret-unknown' },
        },
      },
      1,
    ) as Record<string, any>

    expect(PERSIST_VERSION).toBe(2)
    expect(migrated.agentServerConfigs).toEqual({})
    expectNoCanaries(migrated)
  })

  it('scrubs hydration and merge while preserving every endpoint hint', async () => {
    writeEnvelope(1, {
      _persistedAt: Date.now(),
      agentServerConfigs: canaryConfigs(),
      apiConfig: { llmBaseUrl: 'https://global.example.test/v1', apiKey: 'global-installation-secret' },
      deepResearchApiKey: 'deep-research-secret',
      googleApiKey: 'google-secret',
    })

    await useAppStore.persist.rehydrate()

    expectEndpointHints(useAppStore.getState().agentServerConfigs as any)
    expectNoCanaries(useAppStore.getState())
    expectEndpointHints(readPersistedState().agentServerConfigs)
    expectNoCanaries(readPersistedState())
  })

  it('clears all secrets on explicit logout', async () => {
    await expectSignedOutSecretsCleared(() => useAppStore.getState().logout())
  })

  it('clears all secrets on no-token startup', async () => {
    await expectSignedOutSecretsCleared(() => {
      clearToken()
      return useAppStore.getState().loadAuthFromStorage()
    })
  })

  it('clears all secrets when /api/auth/me rejects', async () => {
    setToken('expired-token')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }))
    await expectSignedOutSecretsCleared(() => useAppStore.getState().loadAuthFromStorage())
  })

  it('clears all secrets when /api/auth/me throws', async () => {
    setToken('network-token')
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('network down')))
    await expectSignedOutSecretsCleared(() => useAppStore.getState().loadAuthFromStorage())
  })
})
