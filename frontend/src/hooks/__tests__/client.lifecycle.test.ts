import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiFetch, ApiError } from '../api/client'
import { useWebSocket } from '../api/useWebSocket'
import { useAppStore } from '../../stores/appStore'
import { clearToken, setToken } from '../../utils/authStorage'

const session = (id: string) => ({
  id,
  name: id,
  status: 'completed' as const,
  currentRound: 0,
  createdAt: new Date().toISOString(),
  instruction: '',
  treatmentGroups: [],
  dataFiles: [],
  biochemistryIndicators: [],
  roundResults: [],
})

function lifecycleFailure(status: 404 | 410, code: string): Response {
  return new Response(JSON.stringify({
    detail: { code, message: code },
  }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('apiFetch session lifecycle scoping', () => {
  beforeEach(() => {
    useAppStore.setState({ currentSession: session('active'), messages: [] })
    vi.restoreAllMocks()
  })

  it.each([
    [404, 'SESSION_NOT_FOUND'],
    [410, 'SESSION_GONE'],
  ] as const)('clears only the active lifecycle target on %s', async (status, code) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(lifecycleFailure(status, code)))

    await expect(apiFetch('/session/active/load')).rejects.toMatchObject({
      status,
      code,
    } satisfies Partial<ApiError>)
    expect(useAppStore.getState().currentSession).toBeNull()
  })

  it('does not clear a newly selected session when an old load fails', async () => {
    useAppStore.getState().setCurrentSession(session('new-active'))
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(lifecycleFailure(410, 'SESSION_GONE')),
    )

    await expect(apiFetch('/session/old-session/load')).rejects.toBeInstanceOf(ApiError)
    expect(useAppStore.getState().currentSession?.id).toBe('new-active')
  })

  it('does not clear the active session for an unrelated resource 404', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(lifecycleFailure(404, 'SESSION_NOT_FOUND')),
    )

    await expect(apiFetch('/session/active/report/missing')).rejects.toBeInstanceOf(ApiError)
    expect(useAppStore.getState().currentSession?.id).toBe('active')
  })
})

class AsyncWebSocketMock {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  static instances: AsyncWebSocketMock[] = []

  readonly url: string
  readonly protocols?: string | string[]
  readyState = AsyncWebSocketMock.CONNECTING
  closeCalls = 0
  onopen: ((event: Event) => unknown) | null = null
  onmessage: ((event: MessageEvent) => unknown) | null = null
  onerror: ((event: Event) => unknown) | null = null
  onclose: ((event: CloseEvent) => unknown) | null = null

  constructor(url: string | URL, protocols?: string | string[]) {
    this.url = String(url)
    this.protocols = protocols
    AsyncWebSocketMock.instances.push(this)
  }

  close(): void {
    this.closeCalls += 1
    this.readyState = AsyncWebSocketMock.CLOSED
  }

  async emitOpen(): Promise<void> {
    this.readyState = AsyncWebSocketMock.OPEN
    await this.onopen?.(new Event('open'))
  }

  emitError(): void {
    this.onerror?.(new Event('error'))
  }

  emitClose(code = 1006): void {
    this.readyState = AsyncWebSocketMock.CLOSED
    this.onclose?.({ code, reason: '' } as CloseEvent)
  }
}

describe('WebSocket connection attempt ownership', () => {
  beforeEach(() => {
    AsyncWebSocketMock.instances = []
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', AsyncWebSocketMock)
    setToken('test-token')
    useAppStore.setState({ currentSession: session('first'), messages: [] })
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    clearToken()
  })

  it('replaces a connecting socket immediately and ignores every retired callback', async () => {
    const { result, unmount } = renderHook(() => useWebSocket())

    act(() => result.current.connect())
    const first = AsyncWebSocketMock.instances[0]
    expect(first).toBeDefined()

    act(() => {
      useAppStore.getState().setCurrentSession(session('second'))
    })
    act(() => result.current.connect())

    expect(AsyncWebSocketMock.instances).toHaveLength(2)
    const second = AsyncWebSocketMock.instances[1]
    expect(second.url).toContain('/ws/second')
    expect(first.closeCalls).toBe(1)

    await act(async () => first.emitOpen())
    expect(result.current.connected).toBe(false)

    await act(async () => second.emitOpen())
    expect(result.current.connected).toBe(true)

    act(() => {
      first.emitError()
      first.emitClose()
      vi.advanceTimersByTime(10_000)
    })

    expect(result.current.connected).toBe(true)
    expect(second.closeCalls).toBe(0)
    expect(AsyncWebSocketMock.instances).toHaveLength(2)
    unmount()
  })

  it.each([
    'waiting_species_input',
    'generating_report',
  ] as const)('reconnects while the session is %s', (status) => {
    useAppStore.setState({
      currentSession: { ...session('active'), status },
    })
    const { result, unmount } = renderHook(() => useWebSocket())

    act(() => result.current.connect())
    expect(AsyncWebSocketMock.instances).toHaveLength(1)

    act(() => AsyncWebSocketMock.instances[0].emitClose(1006))
    expect(AsyncWebSocketMock.instances).toHaveLength(1)

    act(() => vi.advanceTimersByTime(1_000))
    expect(AsyncWebSocketMock.instances).toHaveLength(2)
    expect(AsyncWebSocketMock.instances[1].url).toContain('/ws/active')
    unmount()
  })
})
