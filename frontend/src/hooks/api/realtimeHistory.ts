import type { RawMessage, RealtimeMessagePage } from '../../types'
import type { Message } from '../../stores/appStore'
import { useAppStore } from '../../stores/appStore'
import { devLog } from '../../utils/logger'
import { transformRawMessage } from '../../utils/messageTransform'
import { ApiError, apiFetch } from './client'

export const INITIAL_REALTIME_MESSAGE_PAGE_SIZE = 100
export const RECONNECT_REALTIME_MESSAGE_PAGE_SIZE = 20
export const REALTIME_MESSAGE_PAGE_SIZE = RECONNECT_REALTIME_MESSAGE_PAGE_SIZE

interface HistoryOptions {
  signal?: AbortSignal
  ownsResponse?: () => boolean
  stopWhenOverlapping?: boolean
  onInitialFailure?: (error: unknown) => void
  onFailure?: (error: unknown) => void
}

export class RealtimeHistoryIncompleteError extends Error {
  readonly code = 'REALTIME_HISTORY_INCOMPLETE'
  readonly retryable = true

  constructor(reason: string, readonly cause?: unknown) {
    super(reason)
    this.name = 'RealtimeHistoryIncompleteError'
  }
}

function ownsSession(sessionId: string, options: HistoryOptions): boolean {
  return (
    !options.signal?.aborted
    && (!options.ownsResponse || options.ownsResponse())
    && useAppStore.getState().currentSession?.id === sessionId
  )
}

function normalizedInstant(timestamp: string | undefined): number {
  if (!timestamp) return 0
  const value = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(timestamp)
    ? timestamp
    : `${timestamp}Z`
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function compareMessages(left: Message, right: Message): number {
  const time = normalizedInstant(left.timestamp) - normalizedInstant(right.timestamp)
  return time || left.id.localeCompare(right.id)
}

function isChronological(messages: Message[]): boolean {
  return messages.every((message, index) => (
    index === 0 || compareMessages(messages[index - 1], message) <= 0
  ))
}

function yieldForPagePaint(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0))
}

/** Merge page/live arrivals without replacing the store array unnecessarily. */
export function mergeRealtimeMessages(incoming: Array<RawMessage | Message>): number {
  const transformed = incoming.map(message => transformRawMessage(message as RawMessage))
  let added = 0
  useAppStore.setState(state => {
    const existingIds = new Set(state.messages.map(message => message.id))
    const additionsById = new Map<string, Message>()
    for (const message of transformed) {
      if (existingIds.has(message.id) || additionsById.has(message.id)) continue
      additionsById.set(message.id, message)
    }
    added = additionsById.size
    if (!added) return state
    const existing = isChronological(state.messages)
      ? state.messages
      : [...state.messages].sort(compareMessages)
    const additions = [...additionsById.values()].sort(compareMessages)
    const messages: Message[] = []
    let existingIndex = 0
    let additionIndex = 0
    while (existingIndex < existing.length && additionIndex < additions.length) {
      if (compareMessages(existing[existingIndex], additions[additionIndex]) <= 0) {
        messages.push(existing[existingIndex++])
      } else {
        messages.push(additions[additionIndex++])
      }
    }
    messages.push(...existing.slice(existingIndex), ...additions.slice(additionIndex))
    return { messages }
  })
  return added
}

/** Load newest-to-oldest pages; callers can stop at an overlapping live ID. */
export async function loadRealtimeMessageHistory(
  sessionId: string,
  options: HistoryOptions = {},
): Promise<number> {
  let before: string | undefined
  let pages = 0
  let added = 0
  let restartedCursor = false
  let initial = true
  let verifiedComplete = false
  const overlapIds = new Set(useAppStore.getState().messages.map(message => message.id))
  const pageSize = options.stopWhenOverlapping
    ? RECONNECT_REALTIME_MESSAGE_PAGE_SIZE
    : INITIAL_REALTIME_MESSAGE_PAGE_SIZE
  const reportFailure = (error: unknown, wasInitial: boolean) => {
    if (wasInitial && options.onInitialFailure) {
      options.onInitialFailure(error)
      return
    }
    options.onFailure?.(error)
  }

  await yieldForPagePaint()
  while (pages < 100 && ownsSession(sessionId, options)) {
    if (!ownsSession(sessionId, options)) break
    const query = new URLSearchParams({ limit: String(pageSize) })
    if (before) query.set('before', before)
    try {
      const page = await apiFetch<RealtimeMessagePage>(
        `/session/${sessionId}/messages?${query.toString()}`,
        { signal: options.signal },
        options.ownsResponse,
      )
      if (!ownsSession(sessionId, options)) break
      const pageMessages = page.messages || []
      const pageOverlaps = pageMessages.some(message => overlapIds.has(message.id))
      added += mergeRealtimeMessages(pageMessages)
      pages += 1
      initial = false
      if (options.stopWhenOverlapping && pageOverlaps) {
        verifiedComplete = true
        break
      }
      if (!page.has_more) {
        verifiedComplete = true
        break
      }
      if (!page.next_before || page.next_before === before) {
        const error = new RealtimeHistoryIncompleteError(
          page.next_before
            ? 'Realtime history cursor did not advance'
            : 'Realtime history page is incomplete: next cursor is missing',
        )
        reportFailure(error, false)
        break
      }
      before = page.next_before
      await yieldForPagePaint()
    } catch (error) {
      if (!ownsSession(sessionId, options)) break
      if (error instanceof ApiError && error.code === 'MESSAGE_CURSOR_EXPIRED' && !restartedCursor) {
        restartedCursor = true
        before = undefined
        continue
      }
      const reported = error instanceof ApiError
        && error.code === 'MESSAGE_CURSOR_EXPIRED'
        ? new RealtimeHistoryIncompleteError(
            'Realtime history cursor expired again after one restart',
            error,
          )
        : initial
          ? error
          : new RealtimeHistoryIncompleteError(
              'Realtime history paging stopped before the stream was complete',
              error,
            )
      reportFailure(reported, initial)
      devLog('Realtime history page failed:', reported)
      break
    }
  }
  if (pages >= 100 && !verifiedComplete && ownsSession(sessionId, options)) {
    reportFailure(
      new RealtimeHistoryIncompleteError(
        'Realtime history exceeded the maximum page bound',
      ),
      false,
    )
  }
  return added
}
