import type { StoreApi } from 'zustand'

import { clearToken, getToken, setToken } from '../../utils/authStorage'
import type { AppState } from '../appState'

type SetState = StoreApi<AppState>['setState']
type GetState = StoreApi<AppState>['getState']
type AuthActionKeys =
  | 'authReady' | '_hydrated' | 'authUser' | 'login' | 'register'
  | 'updateProfileEmail' | 'updateProfilePassword' | 'updateProfileUsername' | 'logout' | 'loadAuthFromStorage'

function clearSignedOutSecrets(state: AppState, authReady: boolean) {
  return {
    agentServerConfigs: {},
    apiConfig: { ...state.apiConfig, apiKey: '' },
    deepResearchApiKey: '',
    googleApiKey: '',
    authUser: null,
    authReady,
  }
}

export function createAuthActions(set: SetState, get: GetState): Pick<AppState, AuthActionKeys> {
  return {
    authReady: false,
    _hydrated: false,
    authUser: null,
    login: async (username, password) => {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: 'Invalid credentials' }))
        throw new Error(body.detail || 'Invalid credentials')
      }
      const { access_token, id, username: authenticatedUsername, email, role, status } = await response.json()
      setToken(access_token)
      set({
        authUser: id ? { id, username: authenticatedUsername, email, role, status } : null,
      })
    },
    register: async (username, password, email) => {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, email }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: 'Registration failed' }))
        throw new Error(body.detail || 'Registration failed')
      }
      const data = await response.json()
      return { id: data.id, username: data.username, status: data.status }
    },
    updateProfileEmail: async (currentPassword, newEmail) => {
      const token = getToken()
      if (!token) throw new Error('Not authenticated')
      const response = await fetch('/api/auth/profile/email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ current_password: currentPassword, new_email: newEmail }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: 'Update failed' }))
        throw new Error(body.detail || 'Update failed')
      }
      set(state => state.authUser ? { authUser: { ...state.authUser, email: newEmail } } : {})
    },
    updateProfilePassword: async (currentPassword, newPassword) => {
      const token = getToken()
      if (!token) throw new Error('Not authenticated')
      const response = await fetch('/api/auth/profile/password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: 'Update failed' }))
        throw new Error(body.detail || 'Update failed')
      }
    },
    updateProfileUsername: async (currentPassword, newUsername) => {
      const token = getToken()
      const userId = get().authUser?.id
      if (!token || !userId) throw new Error('Not authenticated')
      const response = await fetch('/api/auth/profile/username', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ current_password: currentPassword, new_username: newUsername }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: 'Update failed' }))
        throw new Error(body.detail || 'Update failed')
      }
      const data = await response.json()
      if (get().authUser?.id !== userId || getToken() !== token) return
      setToken(data.access_token)
      set(state => state.authUser ? { authUser: { ...state.authUser, username: data.username } } : {})
    },
    logout: () => {
      clearToken()
      set(state => ({
        ...clearSignedOutSecrets(state, state.authReady),
        currentSession: null,
        messages: [],
        chatMessages: [],
      }))
    },
    loadAuthFromStorage: async () => {
      if (get().authReady) return
      const token = getToken()
      if (!token) {
        set(state => clearSignedOutSecrets(state, true))
        return
      }
      try {
        const response = await fetch('/api/auth/me', {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (response.ok) {
          set({ authUser: await response.json(), authReady: true })
        } else {
          clearToken()
          set(state => clearSignedOutSecrets(state, true))
        }
      } catch {
        clearToken()
        set(state => clearSignedOutSecrets(state, true))
      }
    },
  }
}
