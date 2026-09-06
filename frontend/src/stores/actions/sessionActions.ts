import type { StoreApi } from 'zustand'

import { DEFAULT_SIDEBAR_WIDTH, clampSidebarWidth } from '../../constants/layout'
import type { AppState } from '../appState'

type SetState = StoreApi<AppState>['setState']

type CurrentSessionKeys = 'currentSession' | 'setCurrentSession' | 'updateSession'

export function createCurrentSessionActions(set: SetState): Pick<AppState, CurrentSessionKeys> {
  return {
    currentSession: null,
    setCurrentSession: currentSession => set({
      currentSession,
      llmFailureInfo: null,
      showLlmTimeoutDialog: false,
      llmModifyPending: false,
      llmModifySaved: false,
      explorationFailedInfo: null,
      showExplorationFailedDialog: false,
      realtimeHistoryError: null,
    }),
    updateSession: updates => set(state => ({
      currentSession: state.currentSession ? { ...state.currentSession, ...updates } : null,
    })),
  }
}

type UiSessionKeys =
  | 'activeTab' | 'setActiveTab' | 'sidebarOpen' | 'setSidebarOpen'
  | 'sidebarWidth' | 'setSidebarWidth' | 'messagePanelWidth' | 'setMessagePanelWidth'
  | 'reportGenerated' | 'setReportGenerated' | 'wsConnected' | 'setWsConnected'
  | 'stateVersion' | 'setStateVersion'

export function createUiSessionActions(set: SetState): Pick<AppState, UiSessionKeys> {
  return {
    activeTab: 'data',
    setActiveTab: activeTab => set(state => state.activeTab === activeTab ? state : { activeTab }),
    sidebarOpen: true,
    setSidebarOpen: sidebarOpen => set({ sidebarOpen }),
    sidebarWidth: DEFAULT_SIDEBAR_WIDTH,
    setSidebarWidth: sidebarWidth => set({ sidebarWidth: clampSidebarWidth(sidebarWidth) }),
    messagePanelWidth: 50,
    setMessagePanelWidth: messagePanelWidth => set({
      messagePanelWidth: Math.max(20, Math.min(80, messagePanelWidth)),
    }),
    reportGenerated: false,
    setReportGenerated: reportGenerated => set({ reportGenerated }),
    wsConnected: false,
    setWsConnected: wsConnected => set({ wsConnected }),
    stateVersion: 0,
    setStateVersion: stateVersion => set({ stateVersion }),
  }
}

type RuntimeSessionKeys =
  | 'showConfigPanel' | 'setShowConfigPanelCallback'
  | 'wsConnect' | 'wsDisconnect' | 'setWsCallbacks'
  | 'readMode' | 'setReadMode' | 'appVersion' | 'setAppVersion'
  | 'sessionLoading' | 'setSessionLoading'

export function createRuntimeSessionActions(
  set: SetState,
): Pick<AppState, RuntimeSessionKeys> {
  return {
    showConfigPanel: null,
    setShowConfigPanelCallback: showConfigPanel => set({ showConfigPanel }),
    wsConnect: null,
    wsDisconnect: null,
    setWsCallbacks: (wsConnect, wsDisconnect) => set({ wsConnect, wsDisconnect }),
    readMode: false,
    setReadMode: readMode => set({ readMode }),
    appVersion: '',
    setAppVersion: appVersion => set({ appVersion }),
    sessionLoading: false,
    setSessionLoading: sessionLoading => set({ sessionLoading }),
  }
}

type InterventionKeys =
  | 'openSessionBrowser' | 'setOpenSessionBrowserCallback'
  | 'llmFailureInfo' | 'setLlmFailureInfo' | 'showLlmTimeoutDialog' | 'setShowLlmTimeoutDialog'
  | 'llmModifyPending' | 'setLlmModifyPending' | 'llmModifySaved' | 'setLlmModifySaved'
  | 'clearLlmFailureState'
  | 'speciesInputInfo' | 'showSpeciesInputDialog' | 'setSpeciesInputInfo'
  | 'setShowSpeciesInputDialog' | 'clearSpeciesInputState'
  | 'apiConfigured' | 'setApiConfigured'
  | 'explorationFailedInfo' | 'showExplorationFailedDialog' | 'setExplorationFailedInfo'
  | 'setShowExplorationFailedDialog' | 'clearExplorationFailedState'

export function createInterventionActions(set: SetState): Pick<AppState, InterventionKeys> {
  return {
    openSessionBrowser: null,
    setOpenSessionBrowserCallback: openSessionBrowser => set({ openSessionBrowser }),
    llmFailureInfo: null,
    setLlmFailureInfo: llmFailureInfo => set({ llmFailureInfo }),
    showLlmTimeoutDialog: false,
    setShowLlmTimeoutDialog: showLlmTimeoutDialog => set({ showLlmTimeoutDialog }),
    llmModifyPending: false,
    setLlmModifyPending: llmModifyPending => set({ llmModifyPending }),
    llmModifySaved: false,
    setLlmModifySaved: llmModifySaved => set({ llmModifySaved }),
    clearLlmFailureState: () => set({
      llmFailureInfo: null,
      showLlmTimeoutDialog: false,
      llmModifyPending: false,
      llmModifySaved: false,
    }),
    speciesInputInfo: null,
    showSpeciesInputDialog: false,
    setSpeciesInputInfo: speciesInputInfo => set({ speciesInputInfo }),
    setShowSpeciesInputDialog: showSpeciesInputDialog => set({ showSpeciesInputDialog }),
    clearSpeciesInputState: () => set({ speciesInputInfo: null, showSpeciesInputDialog: false }),
    apiConfigured: false,
    setApiConfigured: apiConfigured => set({ apiConfigured }),
    explorationFailedInfo: null,
    showExplorationFailedDialog: false,
    setExplorationFailedInfo: explorationFailedInfo => set({ explorationFailedInfo }),
    setShowExplorationFailedDialog: showExplorationFailedDialog => set({ showExplorationFailedDialog }),
    clearExplorationFailedState: () => set({
      explorationFailedInfo: null,
      showExplorationFailedDialog: false,
    }),
  }
}
