// Public surface of the API layer. Implementations live in ./stores/.
// Re-exported here so the appStore.ts import path — which
// every consumer and every vi.mock() target uses — keeps working unchanged.

export * from './domain'
export * from './appState'
export { partialize, migratePersistedState } from './persistence'
export { useAppStore } from './store'
