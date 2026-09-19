import type { StoreApi } from 'zustand'

import { DEFAULT_SIDEBAR_WIDTH, clampSidebarWidth } from '../../constants/layout'
import type { AppState } from '../appState'
import { chatOwnerKey } from '../../utils/chatOwnership'

type SetState = StoreApi<AppState>['setState']

type CurrentSessionKeys = 'currentSession' | 'setCurrentSession' | 'updateSession'
  | 'pendingChatFollowup' | 'setPendingChatFollowup'

export function createCurrentSessionActions(set: SetState): Pick<AppState, CurrentSessionKeys> {
  return {
    currentSession: null,
    setCurrentSession: currentSession => {
      const retired = { chatController: null as AbortController | null }
      set(state => {
        const previousOwner = chatOwnerKey(
          state.currentSession?.id, state.currentSession?.restoreBranchId,
        )
        const nextOwner = chatOwnerKey(
          currentSession?.id, currentSession?.restoreBranchId,
        )
        const ownerChanged = previousOwner !== nextOwner
        if (ownerChanged) retired.chatController = state.chatAbortController
        return {
          currentSession,
          llmFailureInfo: null,
          showLlmTimeoutDialog: false,
          llmModifyPending: false,
          llmModifySaved: false,
          explorationFailedInfo: null,
          explorationFailedOwnerKey: null,
          showExplorationFailedDialog: false,
          realtimeHistoryError: null,
          // Owner changes clear both realtime and chat state in the same store
          // transaction as the session replacement. Abort is deliberately
          // performed after the pure Zustand updater has committed.
          ...(ownerChanged
            ? {
                messages: [],
                chatMessages: [],
                chatLoading: false,
                chatDelegationActive: false,
                chatDelegationPlan: null,
                chatAbortController: null,
                agentErrors: [],
                pendingChatFollowup: null,
                reportGenerated: false,
                runtimeRestoreStatus: 'complete',
                runtimeRestoreWarning: null,
                stateVersion: 0,
                wsConnected: false,
                annotationGateData: null,
                annotationGateOwnerKey: null,
                speciesInputInfo: null,
                speciesInputOwnerKey: null,
                showSpeciesInputDialog: false,
                explorationFailedInfo: null,
                explorationFailedOwnerKey: null,
                showExplorationFailedDialog: false,
              }
            : {}),
        }
      })
      retired.chatController?.abort()
    },
    updateSession: updates => {
      const retired = { chatController: null as AbortController | null }
      set(state => {
        const previousOwner = chatOwnerKey(
          state.currentSession?.id, state.currentSession?.restoreBranchId,
        )
        const provisionalSessionId = Object.prototype.hasOwnProperty.call(updates, 'id')
          ? updates.id
          : state.currentSession?.id
        const provisionalOwnerKey = chatOwnerKey(
          provisionalSessionId,
          Object.prototype.hasOwnProperty.call(updates, 'restoreBranchId')
            ? updates.restoreBranchId
            : state.currentSession?.restoreBranchId,
        )
        const ownerChanged = previousOwner !== provisionalOwnerKey
        const nextSession = state.currentSession
          ? {
              ...state.currentSession,
              // A branch-only change carries the old branch's artifact arrays
              // by reference. Clear them before the incoming update, so an
              // explicit new-owner payload can still replace them in the same
              // transaction and no source artifact ever appears under the new
              // owner.
              ...(ownerChanged
                ? {
                    publicationFigures: [],
                    publicationTables: [],
                    roundResults: [],
                    finalSummary: undefined,
                    figureRevisionTasks: [],
                    bulkInputProfiles: [],
                    methodDecisions: [],
                    integratedReport: undefined,
                    integratedReportPublication: undefined,
                    graphicAbstract: undefined,
                    explorationReportUrl: undefined,
                    integrationReportUrl: undefined,
                    multiDirectionCorrelation: undefined,
                    runtimePersistence: undefined,
                  }
                : {}),
              ...updates,
            }
          : null
        const nextOwner = chatOwnerKey(
          nextSession?.id, nextSession?.restoreBranchId,
        )
        const finalOwnerChanged = previousOwner !== nextOwner
        if (finalOwnerChanged) retired.chatController = state.chatAbortController
        return {
          currentSession: nextSession,
          ...(finalOwnerChanged
            ? {
                messages: [],
                chatMessages: [],
                chatLoading: false,
                chatDelegationActive: false,
                chatDelegationPlan: null,
                chatAbortController: null,
                agentErrors: [],
                pendingChatFollowup: null,
                realtimeHistoryError: null,
                reportGenerated: false,
                runtimeRestoreStatus: 'complete',
                runtimeRestoreWarning: null,
                stateVersion: 0,
                wsConnected: false,
                llmFailureInfo: null,
                showLlmTimeoutDialog: false,
                llmModifyPending: false,
                llmModifySaved: false,
                annotationGateData: null,
                annotationGateOwnerKey: null,
                speciesInputInfo: null,
                speciesInputOwnerKey: null,
                showSpeciesInputDialog: false,
                explorationFailedInfo: null,
                explorationFailedOwnerKey: null,
                showExplorationFailedDialog: false,
              }
            : {}),
        }
      })
      retired.chatController?.abort()
    },
    pendingChatFollowup: null,
    setPendingChatFollowup: pendingChatFollowup => set({ pendingChatFollowup }),
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
  | 'speciesInputInfo' | 'speciesInputOwnerKey' | 'showSpeciesInputDialog' | 'setSpeciesInputInfo'
  | 'setShowSpeciesInputDialog' | 'clearSpeciesInputState'
  | 'apiConfigured' | 'setApiConfigured'
  | 'explorationFailedInfo' | 'explorationFailedOwnerKey' | 'showExplorationFailedDialog' | 'setExplorationFailedInfo'
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
    speciesInputOwnerKey: null,
    showSpeciesInputDialog: false,
    setSpeciesInputInfo: speciesInputInfo => set(state => ({
      speciesInputInfo,
      speciesInputOwnerKey: speciesInputInfo
        ? chatOwnerKey(
            state.currentSession?.id,
            state.currentSession?.restoreBranchId,
          )
        : null,
    })),
    setShowSpeciesInputDialog: showSpeciesInputDialog => set({ showSpeciesInputDialog }),
    clearSpeciesInputState: () => set({
      speciesInputInfo: null,
      speciesInputOwnerKey: null,
      showSpeciesInputDialog: false,
    }),
    apiConfigured: false,
    setApiConfigured: apiConfigured => set({ apiConfigured }),
    explorationFailedInfo: null,
    explorationFailedOwnerKey: null,
    showExplorationFailedDialog: false,
    setExplorationFailedInfo: explorationFailedInfo => set(state => ({
      explorationFailedInfo,
      explorationFailedOwnerKey: explorationFailedInfo
        ? chatOwnerKey(
            state.currentSession?.id,
            state.currentSession?.restoreBranchId,
          )
        : null,
    })),
    setShowExplorationFailedDialog: showExplorationFailedDialog => set({ showExplorationFailedDialog }),
    clearExplorationFailedState: () => set({
      explorationFailedInfo: null,
      explorationFailedOwnerKey: null,
      showExplorationFailedDialog: false,
    }),
  }
}
