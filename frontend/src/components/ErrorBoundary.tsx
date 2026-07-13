import React from 'react'
import { PERSIST_STORAGE_KEY } from '../constants/storage'
import { JWT_STORAGE_KEY } from '../constants/auth'

type State = { hasError: boolean }

export class ErrorBoundary extends React.Component<React.PropsWithChildren, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(err: unknown) {
    console.error('UI crash:', err)
  }

  private reload = () => {
    window.location.reload()
  }

  // Self-recovery when a reload isn't enough — e.g. corrupted/legacy persisted
  // state that crashes rendering on every load. Clears app-owned localStorage
  // (persisted store + auth token) and reloads. The user must sign in again.
  private resetLocalData = () => {
    try {
      localStorage.removeItem(PERSIST_STORAGE_KEY)
      localStorage.removeItem(JWT_STORAGE_KEY)
    } catch { /* ignore storage errors */ }
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div role="alert" style={{ maxWidth: 480, margin: '15vh auto', padding: 24, textAlign: 'center', fontFamily: 'system-ui, sans-serif' }}>
          <p style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>Something went wrong. Please reload.</p>
          <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
            If reloading doesn&apos;t help, reset local data to clear cached app state (you&apos;ll need to sign in again).
          </p>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
            <button
              onClick={this.reload}
              style={{ padding: '8px 16px', borderRadius: 8, border: '1px solid #cbd5e1', background: '#fff', cursor: 'pointer' }}
            >
              Reload
            </button>
            <button
              onClick={this.resetLocalData}
              style={{ padding: '8px 16px', borderRadius: 8, border: 'none', background: '#dc2626', color: '#fff', cursor: 'pointer' }}
            >
              Reset local data
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
