import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore } from '../../stores/appStore'
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
  const options = apiFetchMock.mock.calls[callIndex][1] as RequestInit
  return JSON.parse(String(options.body)) as Record<string, unknown>
}

describe('per-agent context-window save semantics', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiFetchMock.mockResolvedValue({})
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
})
