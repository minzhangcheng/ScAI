// Public surface of the data panel. Implementations live in ./data/.
// Re-exported here so the DataPanel.tsx import path — which
// every consumer and every vi.mock() target uses — keeps working unchanged.
// The pieces below DataPanel were module-private before the split; they are
// exported so a test can mock or mount one directly, not as an invitation to
// import them from elsewhere.

export { SEQUENCING_TYPES, BLOCKED_EXTENSIONS, getAllowedExtensions, isBlockedExtension } from './data/uploadRules'
export { GroupManager } from './data/GroupManager'
export { DataSection } from './data/DataSection'
export { IndicatorForm } from './data/IndicatorForm'
export { DataOverviewMatrix } from './data/DataOverviewMatrix'
export { DataPanel } from './data/DataPanel'
