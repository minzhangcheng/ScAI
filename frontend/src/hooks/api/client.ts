// HTTP client primitives: error type, auth headers, fetch wrapper, store updates.
// Split out of frontend/src/hooks/useApi.ts verbatim.

import { useAppStore, type AppState } from '../../stores/appStore'
import { type StoreUpdates } from '../../utils/transformLoadResponse'
import { isMaskedSecret } from '../../utils/literatureConfig'
import { getToken, clearToken } from '../../utils/authStorage'

export class ApiError extends Error {
  status: number
  code: string
  runtimeStatus: string | null
  details: Record<string, unknown> | null

  constructor(
    status: number,
    code: string,
    message: string,
    runtimeStatus?: string | null,
    details?: Record<string, unknown> | null,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.runtimeStatus = runtimeStatus ?? null
    this.details = details ?? null
  }
}

export const API_BASE = '/api'
export const WS_CONNECT_TIMEOUT_MS = 10_000

function sessionLifecycleTarget(endpoint: string): string | null {
  const match = endpoint.match(/^\/session\/([^/?]+)\/(?:load|ensure)(?:\?|$)/)
  if (!match) return null
  try {
    return decodeURIComponent(match[1])
  } catch {
    return match[1]
  }
}

export function applyStoreUpdates(updates: StoreUpdates): void {
  const store = useAppStore.getState()
  const setSecretFromLoad = (
    current: string,
    incoming: string,
    setter: (value: string) => void,
  ): void => {
    if (!isMaskedSecret(incoming) || !current || isMaskedSecret(current)) {
      setter(incoming)
    }
  }
  if (updates.stateVersion != null) store.setStateVersion(updates.stateVersion)
  if (updates.snapshotCount != null) store.setSnapshotCount(updates.snapshotCount)
  if (updates.roundWaitSeconds != null) store.setRoundWaitSeconds(updates.roundWaitSeconds)
  // Presence check, not a null check: an absent gate must clear a stale one.
  if ('annotationGateData' in updates) store.setAnnotationGateData(updates.annotationGateData ?? null)
  if ('llmFailureInfo' in updates) {
    if (updates.llmFailureInfo) {
      store.setLlmFailureInfo(updates.llmFailureInfo)
      store.setShowLlmTimeoutDialog(true)
    } else {
      store.clearLlmFailureState()
    }
  }
  if (updates.literatureMode != null) store.setLiteratureMode(updates.literatureMode as AppState['literatureMode'])
  if (updates.deepResearchModel != null) store.setDeepResearchModel(updates.deepResearchModel)
  if (updates.deepResearchLlmBaseUrl != null) store.setDeepResearchLlmBaseUrl(updates.deepResearchLlmBaseUrl)
  if (updates.deepResearchApiKey != null) {
    setSecretFromLoad(store.deepResearchApiKey, updates.deepResearchApiKey, store.setDeepResearchApiKey)
  }
  if (updates.googleApiKey != null) {
    setSecretFromLoad(store.googleApiKey, updates.googleApiKey, store.setGoogleApiKey)
  }
  if (updates.pubmedEmail != null) store.setPubmedEmail(updates.pubmedEmail)
  if (updates.literatureMaxSearchRounds != null) store.setLiteratureMaxSearchRounds(updates.literatureMaxSearchRounds)
  if (updates.runtimeRestoreStatus != null || updates.runtimeRestoreWarning !== undefined || updates.snapshotFailure !== undefined) {
    store.hydrateRuntimeRestoreFromSession({
      runtime_restore_status: updates.runtimeRestoreStatus,
      runtime_restore_warning: updates.runtimeRestoreWarning,
      snapshot_failure: updates.snapshotFailure?.failure ?? (updates.snapshotFailure === null ? null : undefined),
      snapshot_failure_allowed_actions: updates.snapshotFailure?.allowed_actions,
    })
  }
  if (updates.snapshotTransaction !== undefined) {
    store.setSnapshotTransaction(updates.snapshotTransaction)
  }
}

// Generate UUID using crypto API with fallback for non-HTTPS contexts
export function generateId(): string {
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

// Get JWT token from localStorage
function getAuthToken(): string | null {
  return getToken()
}

// Build headers with auth token (exported for raw fetch calls that need auth)
export function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const token = getAuthToken()
  const headers: Record<string, string> = { ...extra }
  if (token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

// Generic fetch wrapper (exported for direct use) — auto-injects Authorization header
export async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit,
  ownsResponse?: () => boolean,
): Promise<T> {
  const token = getAuthToken()
  const isFormData = options?.body instanceof FormData
  const headers: Record<string, string> = {
    ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
    ...(options?.headers as Record<string, string> || {}),
  }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  })

  // Auto-logout only on auth-related 401 (from middleware, which uses WWW-Authenticate header)
  // Other 401s (e.g., LLM key validation) should NOT clear the user's session
  if (response.status === 401) {
    const isAuthFailure = response.headers.has('WWW-Authenticate')
    if (isAuthFailure) {
      clearToken()
      throw new ApiError(401, 'AUTH_FAILED', 'Authentication failed. Please log in again.')
    }
    // Non-auth 401 — treat as a regular error
    const error = await response.json().catch(() => ({ detail: 'Unauthorized' }))
    const detail = typeof error.detail === 'object' ? error.detail : { message: error.detail || 'Unauthorized' }
    throw new ApiError(401, detail.code || 'UNAUTHORIZED', detail.message || error.detail || 'Unauthorized', detail.runtime_status || null, detail)
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: 'Unknown error' }))
    const detail = typeof body.detail === 'object' ? body.detail : { message: body.detail || 'API request failed' }
    const lifecycleSessionId = sessionLifecycleTarget(endpoint)
    if (
      lifecycleSessionId &&
      (detail.code === 'SESSION_GONE' || detail.code === 'SESSION_NOT_FOUND') &&
      (response.status === 404 || response.status === 410)
    ) {
      // Only a load/ensure failure for the active session retires client state.
      // A 404 from an unrelated artifact or nested resource must not do so.
      const store = useAppStore.getState()
      if (
        store.currentSession?.id === lifecycleSessionId
        && (!ownsResponse || ownsResponse())
      ) {
        store.setCurrentSession(null)
        store.clearMessages()
      }
    }
    throw new ApiError(
      response.status,
      detail.code || `HTTP_${response.status}`,
      detail.message || body.detail || 'API request failed',
      detail.runtime_status || null,
      detail,
    )
  }

  if (response.status === 204) return undefined as T

  return response.json()
}
