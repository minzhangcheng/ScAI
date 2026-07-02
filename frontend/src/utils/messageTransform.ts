import type { RawMessage, RawAgentError } from '../types'
import type { Message, AgentType } from '../stores/appStore'
import { useAppStore } from '../stores/appStore'

// Generate UUID using crypto API with fallback for non-HTTPS contexts
function generateId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  // Fallback for environments without crypto.randomUUID
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(bytes)
  } else {
    console.warn('crypto.getRandomValues unavailable — falling back to Math.random() for ID generation')
    for (let i = 0; i < 16; i++) bytes[i] = Math.random() * 256 | 0
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`
}

/**
 * Transform a RawMessage from the API to the frontend Message type.
 * Identical to the inline mapping code previously in useApi.ts at 3 locations.
 */
export function transformRawMessage(msg: RawMessage): Message {
  return {
    id: msg.id || generateId(),
    type: (msg.type || 'message') as Message['type'],
    agent: msg.agent as AgentType | undefined,
    content: msg.content as Message['content'],
    timestamp: msg.timestamp || new Date().toISOString(),
  }
}

/**
 * Restore agent errors into the app store from API response data.
 * Clears existing errors and re-adds unresolved ones.
 * Identical to the inline logic previously in useApi.ts at 2 locations.
 */
export function restoreAgentErrors(errors: RawAgentError[]): void {
  const { clearAgentErrors, addAgentError } = useAppStore.getState()
  clearAgentErrors()
  for (const err of errors) {
    if (!err.resolved) {
      addAgentError({
        id: err.id,
        agent: err.agent as AgentType,
        errorType: err.error_type,
        message: err.message,
        details: err.details,
        round: err.round_number,
        timestamp: err.timestamp,
        resolved: false,
      })
    }
  }
}
