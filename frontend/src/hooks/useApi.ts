// Public surface of the API layer. Implementations live in ./api/.
// Re-exported here so the useApi.ts import path — which
// every consumer and every vi.mock() target uses — keeps working unchanged.

export { ApiError, API_BASE, WS_CONNECT_TIMEOUT_MS, applyStoreUpdates, generateId, authHeaders, apiFetch } from './api/client'
import { useSession as useSessionImplementation } from './api/useSession'

// Keep the public facade (and its established test/mock target) while making
// this a local hook export. A direct re-export across lazy panel chunks creates
// a Rollup circular-chunk edge between the facade and implementation.
export function useSession() {
  return useSessionImplementation()
}
export { useWebSocket } from './api/useWebSocket'
export { useConfig, usePrompts, useUserConfig } from './api/useConfig'
