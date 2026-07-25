// Public surface of the API layer. Implementations live in ./api/.
// Re-exported here so the useApi.ts import path — which
// every consumer and every vi.mock() target uses — keeps working unchanged.

export { ApiError, API_BASE, WS_CONNECT_TIMEOUT_MS, applyStoreUpdates, generateId, authHeaders, apiFetch } from './api/client'
export { useSession } from './api/useSession'
export { useWebSocket } from './api/useWebSocket'
export { useConfig, usePrompts, useUserConfig } from './api/useConfig'
export { listSnapshotsV2, previewSnapshotV2, restoreSnapshotV2, cancelSnapshotTransaction, getSnapshotTransaction, deleteSnapshotV2, exportSnapshotsV2, importSnapshotsV2 } from './api/snapshots'
