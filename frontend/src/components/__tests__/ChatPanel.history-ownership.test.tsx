import { act, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ChatPanel } from '../ChatPanel'
import { useAppStore, type Session } from '../../stores/appStore'

vi.mock('../../utils/authStorage', () => ({
  getToken: vi.fn(() => 'test-token'),
  clearToken: vi.fn(),
}))

function session(id: string, restoreBranchId?: string): Session {
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
    ...(restoreBranchId ? { restoreBranchId } : {}),
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

  async function respond(index: number, body: unknown, status = 200) {
    await act(async () => {
      pending[index].resolve(jsonResponse(body, status))
      await Promise.resolve()
      await Promise.resolve()
    })
  }

  function sendMessage(view: ReturnType<typeof render>, text: string) {
    const textarea = view.container.querySelector('textarea') as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: text } })
    fireEvent.click(view.getAllByRole('button').slice(-1)[0])
  }

  function history(id: string) {
    return { messages: [{ id: `message-${id}`, role: 'assistant', content: `history ${id}` }] }
  }

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
      pendingChatFollowup: null,
      chatLoading: false,
      chatDelegationActive: false,
      chatDelegationPlan: null,
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it.each(['initial history', 'poll'])('drops a retired %s response after switching sessions', async kind => {
    vi.useFakeTimers()
    const view = render(<ChatPanel />)
    expect(pending.map(request => request.url)).toEqual(['/api/session/A/chat'])
    let staleIndex = 0
    if (kind === 'poll') {
      await respond(0, { messages: [] })
      act(() => useAppStore.getState().setChatLoading(true))
      await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
      staleIndex = 1
    }
    expect(pending[staleIndex].url).toBe('/api/session/A/chat')
    act(() => useAppStore.getState().setCurrentSession(session('B')))
    expect(pending[staleIndex + 1].url).toBe('/api/session/B/chat')
    expect(pending[staleIndex].signal?.aborted).toBe(true)
    await respond(staleIndex + 1, history('B'))
    expect(useAppStore.getState().chatMessages.map(message => message.id)).toEqual(['message-B'])
    await respond(staleIndex, history('A'))
    expect(useAppStore.getState().currentSession?.id).toBe('B')
    expect(useAppStore.getState().chatMessages.map(message => message.id)).toEqual(['message-B'])
    expect(useAppStore.getState().chatLoading).toBe(false)
    view.unmount()
  })

  it('retains the current request optimistic message when initial history fails', async () => {
    const view = render(<ChatPanel />)
    sendMessage(view, 'current optimistic request')

    const optimistic = useAppStore.getState().chatMessages[0]
    expect(optimistic.content).toBe('current optimistic request')

    await respond(0, { detail: 'history unavailable' }, 503)
    expect(useAppStore.getState().chatMessages).toEqual([optimistic])

    await respond(1, {
      id: 'assistant-current',
      content: 'current response',
      reply_to: optimistic.id,
    })
    view.unmount()
  })

  it.each(['send rejection', 'delegation stop'])('ignores a late %s while the next session sends', async kind => {
    const view = render(<ChatPanel />)
    await respond(0, { messages: [] })
    if (kind === 'send rejection') {
      sendMessage(view, 'request A')
      expect(pending[1].url).toBe('/api/session/A/chat')
    } else {
      act(() => useAppStore.setState({ chatLoading: true, chatDelegationActive: true }))
      fireEvent.click(view.getByTitle('Stop'))
      expect(pending[1].url).toBe('/api/session/A/chat/cancel')
    }
    act(() => useAppStore.getState().setCurrentSession(session('B')))
    if (kind === 'send rejection') expect(pending[1].signal?.aborted).toBe(true)
    await respond(2, { messages: [] })
    sendMessage(view, 'request B')
    const controllerB = useAppStore.getState().chatAbortController
    if (kind === 'send rejection') {
      await act(async () => {
        pending[1].reject(new DOMException('aborted', 'AbortError'))
        await Promise.resolve()
      })
    } else {
      await respond(1, { success: true })
    }
    expect(useAppStore.getState().currentSession?.id).toBe('B')
    expect(useAppStore.getState().chatLoading).toBe(true)
    expect(useAppStore.getState().chatAbortController).toBe(controllerB)
    expect(useAppStore.getState().chatMessages.some(
      message => message.content === 'Chat cancelled.'
    )).toBe(false)
    await respond(3, { content: 'response B' })
    expect(useAppStore.getState().chatLoading).toBe(false)
    view.unmount()
  })

  it('round-trips one optimistic UUID through progressive assistant figures', async () => {
    vi.useFakeTimers()
    const view = render(<ChatPanel />)
    await respond(0, { messages: [] })

    sendMessage(view, 'show the result')

    const send = pending[1]
    const optimisticUser = useAppStore.getState().chatMessages[0]
    expect(optimisticUser.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i)
    expect(send.body).toBeInstanceOf(FormData)
    expect((send.body as FormData).get('message_id')).toBe(optimisticUser.id)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    await respond(2, {
      messages: [
        { ...optimisticUser, timestamp: '2026-08-04T00:00:00Z' },
        {
          id: 'assistant-response', role: 'assistant', content: 'Here is the result.',
          timestamp: '2026-08-04T00:00:01Z', reply_to: optimisticUser.id, pending_figures: true,
        },
      ],
    })

    let store = useAppStore.getState()
    expect(store.chatMessages).toHaveLength(2)
    expect(store.chatMessages[1].reply_to).toBe(optimisticUser.id)
    expect(store.chatMessages[1].pending_figures).toBe(true)

    await respond(1, {
      id: 'assistant-response', role: 'assistant', content: 'Here is the result.',
      timestamp: '2026-08-04T00:00:01Z', reply_to: optimisticUser.id, pending_figures: false,
      figures: [{ name: 'result_plot', path: '/data/result_plot.png' }],
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

  it('reloads chat history when updateSession changes the branch and drops the old owner', async () => {
    useAppStore.setState({ currentSession: session('A', 'branch-a') })
    const view = render(<ChatPanel />)
    act(() => useAppStore.getState().updateSession({ restoreBranchId: 'branch-b' }))
    expect(pending).toHaveLength(2)
    expect(pending[0].signal?.aborted).toBe(true)

    await act(async () => {
      pending[1].resolve(jsonResponse({
        messages: [{ id: 'branch-b-message', role: 'assistant', content: 'branch B' }],
      }))
      await Promise.resolve()
      pending[0].resolve(jsonResponse({
        messages: [{ id: 'branch-a-message', role: 'assistant', content: 'stale branch A' }],
      }))
      await Promise.resolve()
    })
    expect(useAppStore.getState().chatMessages.map(message => message.id)).toEqual([
      'branch-b-message',
    ])
    view.unmount()
  })

  it.each(['accepted', 'failed', 'branch changed'])('handles an edited follow-up when %s', async outcome => {
    useAppStore.setState({
      currentSession: session('A', 'branch-a'),
      pendingChatFollowup: {
        sessionId: 'A', branchId: 'branch-a', followupId: 'fu-retry', instruction: 'Initial follow-up.',
      },
    })
    const view = render(<ChatPanel />)
    await respond(0, { messages: [] })
    const textarea = view.container.querySelector('textarea') as HTMLTextAreaElement
    expect(textarea.value).toBe('Initial follow-up.')
    fireEvent.change(textarea, { target: { value: 'Edited follow-up for retry.' } })
    if (outcome === 'branch changed') {
      act(() => useAppStore.getState().updateSession({ restoreBranchId: 'branch-b' }))
      expect(textarea.value).toBe('')
      expect(useAppStore.getState().pendingChatFollowup).toBeNull()
      expect(pending[0].signal?.aborted).toBe(true)
      await respond(1, { messages: [] })
    } else {
      fireEvent.click(view.getAllByRole('button').slice(-1)[0])
      const optimistic = useAppStore.getState().chatMessages[0]
      expect(optimistic.followup_id).toBe('fu-retry')
      expect(optimistic.branch_id).toBe('branch-a')
      expect((pending[1].body as FormData).get('followup_id')).toBe('fu-retry')
      expect((pending[1].body as FormData).get('branch_id')).toBe('branch-a')
      if (outcome === 'accepted') {
        await respond(1, { id: 'accepted', content: 'Accepted.', reply_to: optimistic.id })
        expect(useAppStore.getState().pendingChatFollowup).toBeNull()
      } else {
        await act(async () => {
          pending[1].reject(new Error('temporary failure'))
          await Promise.resolve()
          await Promise.resolve()
        })
        expect(textarea.value).toBe('Edited follow-up for retry.')
        expect(useAppStore.getState().pendingChatFollowup).toMatchObject({
          sessionId: 'A', branchId: 'branch-a', followupId: 'fu-retry',
          instruction: 'Edited follow-up for retry.',
        })
      }
    }
    view.unmount()
  })
})
