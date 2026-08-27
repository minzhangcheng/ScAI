import { beforeEach, describe, expect, it } from 'vitest'
import { DEFAULT_MODELS, useAppStore } from '../appStore'

describe('Agent page defaults', () => {
  beforeEach(() => {
    useAppStore.getState().resetModelsToDefault()
  })

  it('uses the configured default model for every agent', () => {
    expect(DEFAULT_MODELS).toEqual({
      pi: 'gpt-5.6-sol',
      pathology: 'gemini-3.1-pro-preview',
      bioinformatics: 'gpt-5.6-sol',
      biochemistry: 'gemini-3.1-pro-preview',
      literature: 'gpt-5.4-mini',
      painter: 'gpt-image-2',
      supervisor: 'gpt-5.4-mini',
      technique: 'gpt-5.4-mini',
    })
  })

  it('defaults Literature Agent to PubMed + LLM Only', () => {
    const state = useAppStore.getState()
    expect(state.literatureMode).toBe('pubmed_llm')
    expect(state.deepResearchModel).toBe('o4-mini-deep-research')
    expect(state.literatureMaxSearchRounds).toBe(20)
  })

  it('preserves the complete store key order', () => {
    expect(Object.keys(useAppStore.getState())).toEqual([
      'language', 'setLanguage', 'apiConfig', 'setApiConfig',
      'currentSession', 'setCurrentSession', 'updateSession',
      'messages', 'addMessage', 'setMessages', 'clearMessages',
      'prompts', 'setPrompt', 'resetPromptsToDefault', 'getDefaultPromptsForLanguage',
      'models', 'setModel', 'resetModelsToDefault',
      'temperatures', 'setTemperature', 'resetTemperaturesToDefault',
      'maxRounds', 'setMaxRounds', 'roundWaitSeconds', 'setRoundWaitSeconds',
      'agentServerConfigs', 'setAgentServerConfig',
      'agentVisionSettings', 'setAgentVisionSetting',
      'literatureMode', 'setLiteratureMode', 'deepResearchModel', 'setDeepResearchModel',
      'deepResearchLlmBaseUrl', 'setDeepResearchLlmBaseUrl',
      'deepResearchApiKey', 'setDeepResearchApiKey', 'googleApiKey', 'setGoogleApiKey',
      'pubmedEmail', 'setPubmedEmail', 'literatureMaxSearchRounds', 'setLiteratureMaxSearchRounds',
      'chatMessages', 'addChatMessage', 'setChatMessages', 'chatLoading', 'setChatLoading',
      'chatDelegationActive', 'setChatDelegationActive',
      'chatDelegationPlan', 'setChatDelegationPlan',
      'chatAbortController', 'setChatAbortController',
      'realtimeHistoryError', 'setRealtimeHistoryError',
      'activeTab', 'setActiveTab', 'sidebarOpen', 'setSidebarOpen',
      'sidebarWidth', 'setSidebarWidth', 'messagePanelWidth', 'setMessagePanelWidth',
      'reportGenerated', 'setReportGenerated', 'wsConnected', 'setWsConnected',
      'stateVersion', 'setStateVersion',
      'agentErrors', 'addAgentError', 'clearAgentErrors', 'resolveError',
      'getUnresolvedErrors', 'getErrorsByAgent',
      'showConfigPanel', 'setShowConfigPanelCallback',
      'wsConnect', 'wsDisconnect', 'setWsCallbacks',
      'readMode', 'setReadMode', 'appVersion', 'setAppVersion',
      'sessionLoading', 'setSessionLoading', 'snapshotCount', 'setSnapshotCount',
      'showSnapshotCleanupReminder', 'setShowSnapshotCleanupReminder',
      'snapshotsV2', 'setSnapshotsV2', 'snapshotStoreStatus', 'setSnapshotStoreStatus',
      'snapshotTransaction', 'setSnapshotTransaction',
      'annotationGateData', 'setAnnotationGateData',
      'snapshotFailure', 'setSnapshotFailure', 'clearSnapshotFailure',
      'runtimeRestoreStatus', 'setRuntimeRestoreStatus',
      'runtimeRestoreWarning', 'setRuntimeRestoreWarning',
      'hydrateRuntimeRestoreFromSession', 'isForwardResumeBlocked', 'isRerunEnabled',
      'openSessionBrowser', 'setOpenSessionBrowserCallback',
      'llmFailureInfo', 'setLlmFailureInfo',
      'showLlmTimeoutDialog', 'setShowLlmTimeoutDialog',
      'llmModifyPending', 'setLlmModifyPending', 'llmModifySaved', 'setLlmModifySaved',
      'clearLlmFailureState', 'speciesInputInfo', 'showSpeciesInputDialog',
      'setSpeciesInputInfo', 'setShowSpeciesInputDialog', 'clearSpeciesInputState',
      'apiConfigured', 'setApiConfigured', 'snapshotV2Enabled', 'setSnapshotV2Enabled',
      'explorationFailedInfo', 'showExplorationFailedDialog',
      'setExplorationFailedInfo', 'setShowExplorationFailedDialog', 'clearExplorationFailedState',
      'authReady', '_hydrated', 'authUser', 'login', 'register',
      'updateProfileEmail', 'updateProfilePassword', 'logout', 'loadAuthFromStorage',
    ])
  })
})
