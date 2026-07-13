// Single source of truth for localStorage keys owned by the app, so that
// crash-recovery ("Reset local data") and the Zustand persist middleware
// agree on what to clear/persist.
export const PERSIST_STORAGE_KEY = 'scai-storage'

// Bump when the persisted shape changes in a backwards-incompatible way.
// `migrate` in appStore.ts handles older versions.
export const PERSIST_VERSION = 1
