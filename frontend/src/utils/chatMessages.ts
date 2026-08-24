import type { ChatMessage } from '../types'

function equalValue(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true
  if (left === null || right === null || typeof left !== typeof right) return false
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false
    return left.every((value, index) => equalValue(value, right[index]))
  }
  if (typeof left === 'object' && typeof right === 'object') {
    const leftRecord = left as Record<string, unknown>
    const rightRecord = right as Record<string, unknown>
    const keys = new Set([...Object.keys(leftRecord), ...Object.keys(rightRecord)])
    return [...keys].every(key => equalValue(leftRecord[key], rightRecord[key]))
  }
  return false
}

function sameMessage(left: ChatMessage, right: ChatMessage): boolean {
  return equalValue(left, right)
}

/** Merge server history while retaining references for unchanged records. */
export function reconcileChatHistory(
  serverMessages: ChatMessage[],
  localMessages: ChatMessage[],
): ChatMessage[] {
  const localById = new Map(localMessages.map(message => [message.id, message]))
  const serverIds = new Set(serverMessages.map(message => message.id))
  const reconciled = serverMessages.map(message => {
    const previous = localById.get(message.id)
    if (!previous) return message
    const merged = { ...previous, ...message }
    return sameMessage(previous, merged) ? previous : merged
  })
  reconciled.push(...localMessages.filter(message => !serverIds.has(message.id)))
  if (
    reconciled.length === localMessages.length
    && reconciled.every((message, index) => message === localMessages[index])
  ) {
    return localMessages
  }
  return reconciled
}

/** Update an existing response by ID or append a new response. */
export function upsertChatMessage(
  messages: ChatMessage[],
  incoming: ChatMessage,
): ChatMessage[] {
  const index = messages.findIndex(message => message.id === incoming.id)
  if (index < 0) return [...messages, incoming]
  const current = messages[index]
  const merged = { ...current, ...incoming }
  if (sameMessage(current, merged)) return messages
  const next = [...messages]
  next[index] = merged
  return next
}
