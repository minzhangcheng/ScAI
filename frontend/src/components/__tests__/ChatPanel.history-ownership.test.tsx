import { act, fireEvent, render } from '@testing-library/react'
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
    reject: (reason?: unknown) => void
  }> = []

  beforeEach(() => {
    pending.length = 0
    vi.stubGlobal('fetch', vi.fn((url: string, options?: RequestInit) => (
      new Promise<Response>((resolve, reject) => {
        pending.push({
          url,
          signal: options?.signal ?? undefined,
          resolve,
          reject,
        })
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
    vi.useRealTimers()
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

  it('aborts and drops an in-flight poll after switching sessions', async () => {
    vi.useFakeTimers()
    const view = render(<ChatPanel />)
    await act(async () => {
      pending[0].resolve(jsonResponse({ messages: [] }))
      await Promise.resolve()
      useAppStore.getState().setChatLoading(true)
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    const stalePoll = pending[1]
    expect(stalePoll.url).toBe('/api/session/A/chat')

    act(() => useAppStore.getState().setCurrentSession(session('B')))
    expect(stalePoll.signal?.aborted).toBe(true)
    const historyB = pending[2]
    await act(async () => {
      historyB.resolve(jsonResponse({
        messages: [{
          id: 'message-B', role: 'assistant', content: 'history B',
          timestamp: '2026-08-03T00:00:00Z',
        }],
      }))
      await Promise.resolve()
      stalePoll.resolve(jsonResponse({
        messages: [{
          id: 'message-A', role: 'assistant', content: 'stale poll A',
          timestamp: '2026-08-03T00:00:00Z',
        }],
      }))
      await Promise.resolve()
    })

    expect(useAppStore.getState().chatMessages.map(message => message.id)).toEqual(['message-B'])
    expect(useAppStore.getState().chatLoading).toBe(false)
    view.unmount()
  })

  it('does not let an old send rejection clear the new session request', async () => {
    const view = render(<ChatPanel />)
    await act(async () => {
      pending[0].resolve(jsonResponse({ messages: [] }))
      await Promise.resolve()
    })

    const textarea = view.container.querySelector('textarea') as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: 'request A' } })
    fireEvent.click(Array.from(view.container.querySelectorAll('button')).at(-1)!)
    const sendA = pending[1]
    expect(sendA.url).toBe('/api/session/A/chat')

    act(() => useAppStore.getState().setCurrentSession(session('B')))
    expect(sendA.signal?.aborted).toBe(true)
    await act(async () => {
      pending[2].resolve(jsonResponse({ messages: [] }))
      await Promise.resolve()
    })

    fireEvent.change(textarea, { target: { value: 'request B' } })
    fireEvent.click(Array.from(view.container.querySelectorAll('button')).at(-1)!)
    const sendB = pending[3]
    const controllerB = useAppStore.getState().chatAbortController

    await act(async () => {
      sendA.reject(new DOMException('aborted', 'AbortError'))
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.id).toBe('B')
    expect(useAppStore.getState().chatLoading).toBe(true)
    expect(useAppStore.getState().chatAbortController).toBe(controllerB)
    expect(useAppStore.getState().chatMessages.some(
      message => message.content === 'Chat cancelled.'
    )).toBe(false)

    await act(async () => {
      sendB.resolve(jsonResponse({ content: 'response B' }))
      await Promise.resolve()
    })
    expect(useAppStore.getState().chatLoading).toBe(false)
    view.unmount()
  })

  it('does not let a stale delegation stop clear the new session request', async () => {
    const view = render(<ChatPanel />)
    await act(async () => {
      pending[0].resolve(jsonResponse({ messages: [] }))
      await Promise.resolve()
      useAppStore.getState().setChatLoading(true)
      useAppStore.getState().setChatDelegationActive(true)
    })

    fireEvent.click(view.getByTitle('Stop'))
    const cancelA = pending[1]
    expect(cancelA.url).toBe('/api/session/A/chat/cancel')

    act(() => useAppStore.getState().setCurrentSession(session('B')))
    await act(async () => {
      pending[2].resolve(jsonResponse({ messages: [] }))
      await Promise.resolve()
    })

    const textarea = view.container.querySelector('textarea') as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: 'request B' } })
    fireEvent.click(Array.from(view.container.querySelectorAll('button')).at(-1)!)
    const sendB = pending[3]
    const controllerB = useAppStore.getState().chatAbortController

    await act(async () => {
      cancelA.resolve(jsonResponse({ success: true }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.id).toBe('B')
    expect(useAppStore.getState().chatLoading).toBe(true)
    expect(useAppStore.getState().chatAbortController).toBe(controllerB)

    await act(async () => {
      sendB.resolve(jsonResponse({ content: 'response B' }))
      await Promise.resolve()
    })
    expect(useAppStore.getState().chatLoading).toBe(false)
    view.unmount()
  })
})
