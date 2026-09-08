import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore } from '../../stores/appStore'
import { defaultAgentSettings } from '../../utils/lastAgentSettings'
import { usePromptsController } from './usePromptsController'

const { apiFetchMock } = vi.hoisted(() => ({
  apiFetchMock: vi.fn(),
}))

vi.mock('../../hooks/useApi', () => ({
  apiFetch: apiFetchMock,
  applyStoreUpdates: vi.fn(),
}))

vi.mock('../../utils/transformLoadResponse', () => ({
  transformAgentSettings: vi.fn(() => ({})),
}))

function requestBody(callIndex: number): Record<string, unknown> {
  const settingsCalls = apiFetchMock.mock.calls.filter(([endpoint]) => (
    String(endpoint).includes('/agent-settings')
  ))
  const options = settingsCalls[callIndex][1] as RequestInit
  return JSON.parse(String(options.body)) as Record<string, unknown>
}

describe('per-agent context-window save semantics', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    apiFetchMock.mockResolvedValue(defaultAgentSettings('en'))
    useAppStore.setState({
      currentSession: { id: 'context-session' } as never,
      agentServerConfigs: {
        pi: {
          llmBaseUrl: '',
          apiKey: '',
          contextWindow: 128000,
          effectiveContextWindow: 128000,
        },
      },
    })
  })

  it('distinguishes an untouched override, an integer update, and an explicit reset', async () => {
    const { result } = renderHook(() => usePromptsController())

    await act(async () => {
      await result.current?.handleSave()
    })
    expect(requestBody(0)).not.toHaveProperty('context_window')

    act(() => {
      result.current?.setEditedContextWindow(96000)
      result.current?.setContextWindowTouched(true)
    })
    await act(async () => {
      await result.current?.handleSave()
    })
    expect(requestBody(1).context_window).toBe(96000)

    act(() => {
      result.current?.setEditedContextWindow(undefined)
      result.current?.setContextWindowTouched(true)
    })
    await act(async () => {
      await result.current?.handleSave()
    })
    expect(requestBody(2).context_window).toBeNull()
  })

  it('reports a default-save failure without caching the new defaults locally', async () => {
    const response = {
      ...defaultAgentSettings('en'),
      default_settings_saved: false,
    }
    apiFetchMock.mockResolvedValue(response)
    useAppStore.setState({
      authUser: { id: 'user-1', username: 'tester', role: 'user', status: 'active' },
      language: 'en',
    })
    const { result } = renderHook(() => usePromptsController())

    await act(async () => {
      await result.current?.handleSave()
    })

    expect(result.current?.saveError).toBe(
      'Session settings were saved, but defaults for new analyses could not be saved. Please try again.',
    )
    expect(localStorage.getItem('scai-last-agent-settings:user-1')).toBeNull()
  })

  it('clears the old model budget in defaults for a new analysis', async () => {
    useAppStore.setState({ currentSession: null, authUser: null })
    const { result } = renderHook(() => usePromptsController())
    act(() => { result.current?.setEditedModel('replacement-model') })
    await act(async () => { await result.current?.handleSave() })
    expect(useAppStore.getState().agentServerConfigs.pi?.contextWindow).toBeUndefined()
  })
})
