import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore } from '../../../stores/appStore'
import { defaultAgentSettings, saveLastAgentSettings } from '../../../utils/lastAgentSettings'
import { useSessionData } from './useSessionData'

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }))

vi.mock('../client', () => ({ apiFetch: apiFetchMock }))

describe('new-session Agent defaults', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    useAppStore.setState({
      authUser: {
        id: 'user-1', username: 'tester', role: 'user', status: 'active',
      },
      language: 'en',
    })
  })

  it('marks a newly added indicator as manual immediately', async () => {
    useAppStore.setState({ currentSession: {
      id: 'session', name: 'Study', status: 'pending', currentRound: 0,
      createdAt: '', instruction: '', treatmentGroups: [], dataFiles: [],
      biochemistryIndicators: [], roundResults: [],
    } })
    apiFetchMock.mockResolvedValue({ success: true, data: { indicator_id: 'manual' } })
    const { result } = renderHook(() => useSessionData(useAppStore.getState()))
    await act(async () => { await result.current.addIndicator('Measured value', 5, 'g') })
    expect(useAppStore.getState().currentSession?.biochemistryIndicators).toEqual([
      expect.objectContaining({ id: 'manual', indicator: 'Measured value', origin: 'manual' }),
    ])
  })

  it('copies the per-user nonsecret snapshot into the start request', async () => {
    const settings = defaultAgentSettings('en')
    settings.models.pi = 'saved-model'
    settings.agent_server_configs.pi = {
      llm_base_url: 'http://localhost:8000/v1',
      api_key: 'must-not-reach-session-start',
    }
    saveLastAgentSettings('user-1', settings)
    apiFetchMock.mockResolvedValue({
      success: true,
      session_id: 'new-session',
      data: { name: 'New Session' },
    })
    const store = {
      currentSession: null,
      setCurrentSession: vi.fn(),
      clearMessages: vi.fn(),
      setLoading: vi.fn(),
      setError: vi.fn(),
    }
    const { result } = renderHook(() => useSessionData(store))

    await act(async () => {
      await result.current.createSession('Analyze this')
    })

    const request = apiFetchMock.mock.calls[0][1] as RequestInit
    const body = JSON.parse(String(request.body))
    expect(body.agent_settings.models.pi).toBe('saved-model')
    expect(body.agent_settings.agent_server_configs.pi).toEqual({
      llm_base_url: 'http://localhost:8000/v1',
    })
    expect(JSON.stringify(body)).not.toContain('must-not-reach-session-start')
  })

  it('lets the backend supply effective defaults when no browser snapshot exists', async () => {
    useAppStore.setState({ pubmedEmail: 'configured@example.test' })
    apiFetchMock.mockResolvedValue({
      success: true,
      session_id: 'new-session',
      data: { name: 'New Session' },
    })
    const store = {
      currentSession: null,
      setCurrentSession: vi.fn(),
      clearMessages: vi.fn(),
      setLoading: vi.fn(),
      setError: vi.fn(),
    }
    const { result } = renderHook(() => useSessionData(store))

    await act(async () => {
      await result.current.createSession('Analyze this')
    })

    const request = apiFetchMock.mock.calls[0][1] as RequestInit
    const body = JSON.parse(String(request.body))
    expect(body.agent_settings).toEqual({
      literature_config: { pubmed_email: 'configured@example.test' },
    })
    expect(body.agent_settings).not.toHaveProperty('models')
    expect(body.agent_settings).not.toHaveProperty('vision_settings')
  })
})
