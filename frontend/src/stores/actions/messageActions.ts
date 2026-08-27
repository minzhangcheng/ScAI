import type { StoreApi } from 'zustand'

import type { AppState } from '../appState'

type SetState = StoreApi<AppState>['setState']
type GetState = StoreApi<AppState>['getState']

let fallbackChatIdentity = 0

function createChatIdentity(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(bytes)
  } else {
    fallbackChatIdentity += 1
    const value = `${Date.now().toString(16)}${fallbackChatIdentity.toString(16)}`
    for (let index = 0; index < value.length && index < bytes.length * 2; index += 2) {
      bytes[index / 2] = Number.parseInt(value.slice(index, index + 2).padStart(2, '0'), 16)
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

type RealtimeMessageKeys = 'messages' | 'addMessage' | 'setMessages' | 'clearMessages'

export function createRealtimeMessageActions(set: SetState): Pick<AppState, RealtimeMessageKeys> {
  return {
    messages: [],
    addMessage: message => set(state => {
      if (message.type === 'progress') {
        let lastProgressIndex = -1
        for (let index = state.messages.length - 1; index >= 0; index -= 1) {
          if (state.messages[index].type === 'progress' && state.messages[index].agent === message.agent) {
            lastProgressIndex = index
            break
          }
        }
        if (lastProgressIndex !== -1) {
          const messages = [...state.messages]
          messages[lastProgressIndex] = message
          return { messages }
        }
      }
      return { messages: [...state.messages, message] }
    }),
    setMessages: messages => set({ messages }),
    clearMessages: () => set({ messages: [] }),
  }
}

type ChatMessageKeys =
  | 'chatMessages' | 'addChatMessage' | 'setChatMessages'
  | 'chatLoading' | 'setChatLoading'
  | 'chatDelegationActive' | 'setChatDelegationActive'
  | 'chatDelegationPlan' | 'setChatDelegationPlan'
  | 'chatAbortController' | 'setChatAbortController'
  | 'realtimeHistoryError' | 'setRealtimeHistoryError'

export function createChatMessageActions(set: SetState): Pick<AppState, ChatMessageKeys> {
  return {
    chatMessages: [],
    addChatMessage: message => set(state => ({
      chatMessages: [...state.chatMessages, { ...message, id: message.id || createChatIdentity() }],
    })),
    setChatMessages: messages => set({
      chatMessages: messages.every(message => Boolean(message.id))
        ? messages
        : messages.map(message => message.id ? message : ({ ...message, id: createChatIdentity() })),
    }),
    chatLoading: false,
    setChatLoading: chatLoading => set({ chatLoading }),
    chatDelegationActive: false,
    setChatDelegationActive: chatDelegationActive => set({ chatDelegationActive }),
    chatDelegationPlan: null,
    setChatDelegationPlan: chatDelegationPlan => set({ chatDelegationPlan }),
    chatAbortController: null,
    setChatAbortController: chatAbortController => set({ chatAbortController }),
    realtimeHistoryError: null,
    setRealtimeHistoryError: realtimeHistoryError => set({ realtimeHistoryError }),
  }
}

type AgentErrorKeys =
  | 'agentErrors' | 'addAgentError' | 'clearAgentErrors' | 'resolveError'
  | 'getUnresolvedErrors' | 'getErrorsByAgent'

export function createAgentErrorActions(set: SetState, get: GetState): Pick<AppState, AgentErrorKeys> {
  return {
    agentErrors: [],
    addAgentError: error => set(state => ({ agentErrors: [...state.agentErrors, error] })),
    clearAgentErrors: () => set({ agentErrors: [] }),
    resolveError: errorId => set(state => ({
      agentErrors: state.agentErrors.map(error => (
        error.id === errorId ? { ...error, resolved: true } : error
      )),
    })),
    getUnresolvedErrors: () => get().agentErrors.filter(error => !error.resolved),
    getErrorsByAgent: agent => get().agentErrors.filter(error => error.agent === agent && !error.resolved),
  }
}
