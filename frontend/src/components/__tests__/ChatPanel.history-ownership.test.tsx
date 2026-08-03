import { act, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ChatPanel } from '../ChatPanel'
import { useAppStore, type Session } from '../../stores/appStore'

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

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('ChatPanel history response ownership', () => {
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
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: vi.fn(),
    })
    useAppStore.setState({
      language: 'en',
      currentSession: session('A'),
      chatMessages: [],
      chatLoading: false,
      chatDelegationActive: false,
      chatDelegationPlan: null,
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('aborts A and drops its late success after B history has loaded', async () => {
    const view = render(<ChatPanel />)
    expect(pending.map(request => request.url)).toEqual(['/api/session/A/chat'])

    act(() => useAppStore.getState().setCurrentSession(session('B')))
    expect(pending.map(request => request.url)).toEqual([
      '/api/session/A/chat',
      '/api/session/B/chat',
    ])
    expect(pending[0].signal?.aborted).toBe(true)

    await act(async () => {
      pending[1].resolve(jsonResponse({
        messages: [{
          id: 'message-B',
          role: 'assistant',
          content: 'history B',
          timestamp: '2026-08-03T00:00:00Z',
        }],
      }))
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(useAppStore.getState().chatMessages.map(message => message.id)).toEqual(['message-B'])

    await act(async () => {
      pending[0].resolve(jsonResponse({
        messages: [{
          id: 'message-A',
          role: 'assistant',
          content: 'stale history A',
          timestamp: '2026-08-03T00:00:00Z',
        }],
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.id).toBe('B')
    expect(useAppStore.getState().chatMessages.map(message => message.id)).toEqual(['message-B'])
    view.unmount()
  })

  it('does not clear B history when the retired A request fails late', async () => {
    const view = render(<ChatPanel />)
    act(() => useAppStore.getState().setCurrentSession(session('B')))

    await act(async () => {
      pending[1].resolve(jsonResponse({
        messages: [{
          id: 'message-B',
          role: 'assistant',
          content: 'history B',
          timestamp: '2026-08-03T00:00:00Z',
        }],
      }))
      await Promise.resolve()
      await Promise.resolve()
    })
    await act(async () => {
      pending[0].resolve(jsonResponse({ detail: 'late failure' }, 500))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.id).toBe('B')
    expect(useAppStore.getState().chatMessages.map(message => message.id)).toEqual(['message-B'])
    view.unmount()
  })
})
