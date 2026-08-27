import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useSession } from '../api/useSession'
import { useAppStore, type Session } from '../../stores/appStore'
import type { SessionLoadResponse } from '../../types'

vi.mock('../../utils/authStorage', () => ({
  getToken: vi.fn(() => 'test-token'),
  clearToken: vi.fn(),
}))

function session(id: string): Session {
  return {
    id,
    name: id,
    status: 'completed',
    currentRound: 0,
    createdAt: '2026-08-03T00:00:00Z',
    instruction: '',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }
}

function loadResponse(id: string, marker: number): SessionLoadResponse {
  return {
    session_id: id,
    name: id,
    status: 'completed',
    current_round: marker,
    created_at: '2026-08-03T00:00:00Z',
    treatment_groups: [],
    data_files: [],
    biochemistry_indicators: [],
    round_results: [],
    state_version: marker,
    snapshot_count: marker,
    messages: [{ id: `message-${id}`, type: 'message', content: id }],
    agent_errors: [{
      id: `error-${id}`,
      agent: 'pi',
      error_type: 'test',
      message: id,
      round_number: marker,
      timestamp: '2026-08-03T00:00:00Z',
      resolved: false,
    }],
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('loadSession response ownership', () => {
  const pending: Array<{
    url: string
    signal?: AbortSignal
    resolve: (response: Response) => void
  }> = []

  beforeEach(() => {
    pending.length = 0
    vi.stubGlobal('fetch', vi.fn((url: string, options?: RequestInit) => (
      new Promise<Response>((resolve) => {
        pending.push({ url, signal: options?.signal ?? undefined, resolve })
      })
    )))
    useAppStore.setState({
      currentSession: session('selected-before-load'),
      messages: [{
        id: 'initial-message',
        type: 'message',
        content: 'initial',
        timestamp: '2026-08-03T00:00:00Z',
      }],
      agentErrors: [{
        id: 'initial-error',
        agent: 'pi',
        errorType: 'test',
        message: 'initial',
        round: 0,
        timestamp: '2026-08-03T00:00:00Z',
        resolved: false,
      }],
      stateVersion: 7,
      snapshotCount: 7,
      sessionLoading: false,
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('preserves the hook surface order and resume alias identity', () => {
    const { result } = renderHook(() => useSession())

    expect(Object.keys(result.current)).toEqual([
      'currentSession', 'loading', 'error', 'createSession', 'loadSession',
      'retryRealtimeHistory', 'listSessions', 'deleteSession', 'saveSession',
      'updateSessionInfo', 'updateRoundWait', 'addGroup', 'updateGroup', 'deleteGroup',
      'uploadData', 'deleteDataFile', 'addIndicator', 'deleteIndicator', 'startAnalysis',
      'pauseAnalysis', 'resumeAnalysis', 'continueRound', 'stopAnalysis', 'exportState',
      'getReport', 'updateSession',
    ])
    expect(result.current.resumeAnalysis).toBe(result.current.continueRound)
  })

  it('drops a superseded response and keeps both loading indicators owned by the newer request', async () => {
    const { result } = renderHook(() => useSession())
    let staleLoad!: Promise<unknown>
    let currentLoad!: Promise<Session>

    act(() => {
      staleLoad = result.current.loadSession('A').catch(error => error)
    })
    act(() => {
      currentLoad = result.current.loadSession('B')
    })

    expect(pending).toHaveLength(2)
    expect(pending[0].signal?.aborted).toBe(true)

    await act(async () => {
      pending[0].resolve(jsonResponse(loadResponse('A', 100)))
      await staleLoad
    })

    expect(result.current.loading).toBe(true)
    expect(useAppStore.getState().sessionLoading).toBe(true)
    expect(useAppStore.getState().currentSession?.id).toBe('selected-before-load')
    expect(useAppStore.getState().messages.map(message => message.id)).toEqual(['initial-message'])
    expect(useAppStore.getState().agentErrors.map(error => error.id)).toEqual(['initial-error'])
    expect(useAppStore.getState().stateVersion).toBe(7)
    expect(useAppStore.getState().snapshotCount).toBe(7)

    await act(async () => {
      pending[1].resolve(jsonResponse(loadResponse('B', 2)))
      await currentLoad
    })

    expect(result.current.loading).toBe(false)
    expect(useAppStore.getState().sessionLoading).toBe(false)
    expect(useAppStore.getState().currentSession?.id).toBe('B')
    expect(useAppStore.getState().messages.map(message => message.id)).toEqual(['message-B'])
    expect(useAppStore.getState().agentErrors.map(error => error.id)).toEqual(['error-B'])
    expect(useAppStore.getState().stateVersion).toBe(2)
    expect(useAppStore.getState().snapshotCount).toBe(2)
  })

  it('does not let a stale same-session lifecycle error clear the newer load', async () => {
    useAppStore.setState({ currentSession: session('same') })
    const { result } = renderHook(() => useSession())
    let staleLoad!: Promise<unknown>
    let currentLoad!: Promise<Session>

    act(() => {
      staleLoad = result.current.loadSession('same').catch(error => error)
    })
    act(() => {
      currentLoad = result.current.loadSession('same')
    })

    await act(async () => {
      pending[1].resolve(jsonResponse(loadResponse('same', 3)))
      await currentLoad
    })
    await act(async () => {
      pending[0].resolve(jsonResponse({
        detail: { code: 'SESSION_GONE', message: 'retired generation' },
      }, 410))
      await staleLoad
    })

    expect(useAppStore.getState().currentSession?.id).toBe('same')
    expect(useAppStore.getState().currentSession?.currentRound).toBe(3)
    expect(result.current.error).toBeNull()
  })

  it('commits session state before paging history and merges a live arrival', async () => {
    const { result } = renderHook(() => useSession())
    let load!: Promise<Session>

    act(() => {
      load = result.current.loadSession('history')
    })
    expect(pending[0].url).toBe('/api/session/history/load')

    await act(async () => {
      pending[0].resolve(jsonResponse({ ...loadResponse('history', 4), messages: undefined }))
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(useAppStore.getState().currentSession?.id).toBe('history')
    expect(pending[1].url).toContain('/api/session/history/messages')

    act(() => useAppStore.getState().addMessage({
      id: 'live-arrival',
      type: 'message',
      content: 'live',
      timestamp: '2026-08-03T00:00:02Z',
    }))
    await act(async () => {
      pending[1].resolve(jsonResponse({
        messages: [{ id: 'paged', type: 'message', content: 'paged', timestamp: '2026-08-03T00:00:01Z' }],
        page_size: 1,
        has_more: false,
        next_before: null,
      }))
      await load
    })

    expect(useAppStore.getState().messages.map(message => message.id)).toEqual(['paged', 'live-arrival'])
    expect(result.current.error).toBeNull()
  })
})
