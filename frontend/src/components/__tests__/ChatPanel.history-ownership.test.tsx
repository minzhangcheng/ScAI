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
    body?: BodyInit | null
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
          body: options?.body,
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

  it('drops a legacy persisted chat cache before loading the selected session', async () => {
    useAppStore.setState({
      chatMessages: [{
        role: 'assistant',
        content: 'legacy stale cache',
        timestamp: '2026-08-03T00:00:00Z',
      } as any],
    })
    const view = render(<ChatPanel />)

    await act(async () => {
      pending[0].resolve(jsonResponse({
        messages: [{
          id: 'message-A',
          role: 'assistant',
          content: 'authoritative A',
          timestamp: '2026-08-03T00:00:00Z',
        }],
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().chatMessages.map(message => message.content)).toEqual([
      'authoritative A',
    ])
    view.unmount()
  })

  it('retains the current request optimistic message when initial history fails', async () => {
    const view = render(<ChatPanel />)
    const textarea = view.container.querySelector('textarea') as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: 'current optimistic request' } })
    fireEvent.click(Array.from(view.container.querySelectorAll('button')).at(-1)!)

    const optimistic = useAppStore.getState().chatMessages[0]
    expect(optimistic.content).toBe('current optimistic request')

    await act(async () => {
      pending[0].resolve(jsonResponse({ detail: 'history unavailable' }, 503))
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(useAppStore.getState().chatMessages).toEqual([optimistic])

    await act(async () => {
      pending[1].resolve(jsonResponse({
        id: 'assistant-current',
        content: 'current response',
        reply_to: optimistic.id,
      }))
      await Promise.resolve()
      await Promise.resolve()
    })
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

  it('round-trips one optimistic UUID through progressive assistant figures', async () => {
    vi.useFakeTimers()
    const view = render(<ChatPanel />)
    await act(async () => {
      pending[0].resolve(jsonResponse({ messages: [] }))
      await Promise.resolve()
    })

    const textarea = view.container.querySelector('textarea') as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: 'show the result' } })
    fireEvent.click(Array.from(view.container.querySelectorAll('button')).at(-1)!)

    const send = pending[1]
    const optimisticUser = useAppStore.getState().chatMessages[0]
    expect(optimisticUser.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i)
    expect(send.body).toBeInstanceOf(FormData)
    expect((send.body as FormData).get('message_id')).toBe(optimisticUser.id)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    const poll = pending[2]
    await act(async () => {
      poll.resolve(jsonResponse({
        messages: [
          { ...optimisticUser, timestamp: '2026-08-04T00:00:00Z' },
          {
            id: 'assistant-response',
            role: 'assistant',
            content: 'Here is the result.',
            timestamp: '2026-08-04T00:00:01Z',
            reply_to: optimisticUser.id,
            pending_figures: true,
          },
        ],
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    let store = useAppStore.getState()
    expect(store.chatMessages).toHaveLength(2)
    expect(store.chatMessages[1].reply_to).toBe(optimisticUser.id)
    expect(store.chatMessages[1].pending_figures).toBe(true)

    await act(async () => {
      send.resolve(jsonResponse({
        id: 'assistant-response',
        role: 'assistant',
        content: 'Here is the result.',
        timestamp: '2026-08-04T00:00:01Z',
        reply_to: optimisticUser.id,
        pending_figures: false,
        figures: [{ name: 'result_plot', path: '/data/result_plot.png' }],
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    store = useAppStore.getState()
    expect(store.chatMessages).toHaveLength(2)
    expect(store.chatMessages[1].id).toBe('assistant-response')
    expect(store.chatMessages[1].reply_to).toBe(optimisticUser.id)
    expect(store.chatMessages[1].pending_figures).toBe(false)
    expect(store.chatMessages[1].figures).toEqual([
      { name: 'result_plot', path: '/data/result_plot.png' },
    ])
    view.unmount()
  })
})
