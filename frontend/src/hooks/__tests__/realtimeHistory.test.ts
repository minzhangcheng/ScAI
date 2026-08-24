import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore, type Message, type Session } from '../../stores/appStore'
import { ApiError } from '../api/client'

function session(id: string): Session {
  return {
    id,
    name: id,
    status: 'completed',
    currentRound: 0,
    createdAt: '2026-08-10T00:00:00Z',
    instruction: '',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }
}

function rawMessage(index: number, timestamp?: string) {
  return {
    id: `m-${index.toString().padStart(2, '0')}`,
    type: 'message',
    content: { index },
    timestamp: timestamp ?? `2026-08-10T00:00:${index.toString().padStart(2, '0')}Z`,
  }
}

function message(index: number): Message {
  return rawMessage(index) as Message
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function page(start: number, end: number, nextBefore: string | null) {
  return {
    messages: Array.from({ length: end - start }, (_, offset) => rawMessage(start + offset)),
    count: end - start,
    total: 45,
    page_size: end - start,
    has_more: nextBefore !== null,
    next_before: nextBefore,
  }
}

describe('realtime history pagination', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    useAppStore.setState({ currentSession: session('history'), messages: [] })
  })

  it('merges the complete stream in chronological batches of twenty', async () => {
    const { loadRealtimeMessageHistory, REALTIME_MESSAGE_PAGE_SIZE } = await import('../api/realtimeHistory')
    const urls: string[] = []
    vi.stubGlobal('fetch', vi.fn((input: string | URL | Request) => {
      const url = String(input)
      urls.push(url)
      if (url.includes('before=m-25')) return Promise.resolve(jsonResponse(page(5, 25, 'm-05')))
      if (url.includes('before=m-05')) return Promise.resolve(jsonResponse(page(0, 5, null)))
      return Promise.resolve(jsonResponse(page(25, 45, 'm-25')))
    }))

    await loadRealtimeMessageHistory('history', { ownsResponse: () => true })

    expect(REALTIME_MESSAGE_PAGE_SIZE).toBe(20)
    expect(urls).toHaveLength(3)
    expect(urls.every(url => url.includes('limit=20'))).toBe(true)
    expect(useAppStore.getState().messages.map(item => item.id)).toEqual(
      Array.from({ length: 45 }, (_, index) => `m-${index.toString().padStart(2, '0')}`),
    )
  })

  it('stops reconnect catch-up at the first overlapping page', async () => {
    const { loadRealtimeMessageHistory } = await import('../api/realtimeHistory')
    useAppStore.setState({
      currentSession: session('history'),
      messages: Array.from({ length: 10 }, (_, offset) => message(35 + offset)),
    })
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(page(25, 45, 'm-25')))
    vi.stubGlobal('fetch', fetchMock)

    await loadRealtimeMessageHistory('history', {
      ownsResponse: () => true,
      stopWhenOverlapping: true,
    })

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(String(fetchMock.mock.calls[0][0])).toContain('/messages?limit=20')
    expect(String(fetchMock.mock.calls[0][0])).not.toContain('/load')
    expect(useAppStore.getState().messages).toHaveLength(20)
  })

  it('normalizes offset and naive timestamps to UTC before sorting by ID', async () => {
    const { mergeRealtimeMessages } = await import('../api/realtimeHistory')
    useAppStore.setState({
      currentSession: session('history'),
      messages: [],
    })
    mergeRealtimeMessages([
      { ...rawMessage(2), id: 'z', timestamp: '2026-08-10T01:00:00+01:00' },
      { ...rawMessage(1), id: 'a', timestamp: '2026-08-10T00:00:00' },
      { ...rawMessage(3), id: 'b', timestamp: '2026-08-10T00:00:00Z' },
    ])

    expect(useAppStore.getState().messages.map(item => item.id)).toEqual(['a', 'b', 'z'])
  })

  it('exposes an initial-page failure to a retryable owner callback', async () => {
    const { loadRealtimeMessageHistory } = await import('../api/realtimeHistory')
    const failure = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      jsonResponse({ detail: 'history unavailable' }, 503),
    ))

    await loadRealtimeMessageHistory('history', {
      ownsResponse: () => true,
      onInitialFailure: failure,
    })

    expect(failure).toHaveBeenCalledTimes(1)
    expect(failure.mock.calls[0][0]).toBeInstanceOf(Error)
  })

  it('retains merged pages and reports later failures as retryable incomplete history', async () => {
    const { loadRealtimeMessageHistory, RealtimeHistoryIncompleteError } = await import('../api/realtimeHistory')
    const failure = vi.fn()
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(jsonResponse(page(25, 45, 'm-25')))
      .mockResolvedValueOnce(jsonResponse({ detail: 'history unavailable' }, 503)))

    await loadRealtimeMessageHistory('history', {
      ownsResponse: () => true,
      onFailure: failure,
    })

    expect(useAppStore.getState().messages).toHaveLength(20)
    expect(failure).toHaveBeenCalledTimes(1)
    expect(failure.mock.calls[0][0]).toBeInstanceOf(RealtimeHistoryIncompleteError)
    expect(failure.mock.calls[0][0].retryable).toBe(true)
  })

  it('reports a second cursor expiry and malformed continuation as incomplete', async () => {
    const { loadRealtimeMessageHistory, RealtimeHistoryIncompleteError } = await import('../api/realtimeHistory')
    const failure = vi.fn()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(page(25, 45, 'm-25')))
      .mockRejectedValueOnce(new ApiError(409, 'MESSAGE_CURSOR_EXPIRED', 'expired'))
      .mockRejectedValueOnce(new ApiError(409, 'MESSAGE_CURSOR_EXPIRED', 'expired again'))
    vi.stubGlobal('fetch', fetchMock)

    await loadRealtimeMessageHistory('history', {
      ownsResponse: () => true,
      onFailure: failure,
    })

    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(useAppStore.getState().messages).toHaveLength(20)
    expect(failure.mock.calls[0][0]).toBeInstanceOf(RealtimeHistoryIncompleteError)
    expect(failure.mock.calls[0][0].retryable).toBe(true)

    failure.mockClear()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      messages: [rawMessage(10)],
      page_size: 1,
      has_more: true,
      next_before: null,
    })))
    await loadRealtimeMessageHistory('history', {
      ownsResponse: () => true,
      onFailure: failure,
    })
    expect(failure.mock.calls[0][0]).toBeInstanceOf(RealtimeHistoryIncompleteError)
    expect(failure.mock.calls[0][0].retryable).toBe(true)
  })
})
