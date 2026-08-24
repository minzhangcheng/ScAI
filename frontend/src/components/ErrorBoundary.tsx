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
        <div role="alert" className="mx-auto mt-[15vh] max-w-[480px] p-6 text-center font-sans text-slate-900 dark:text-slate-100">
          <p className="mb-2 text-base font-semibold">Something went wrong. Please reload.</p>
          <p className="mb-4 text-[13px] text-slate-500 dark:text-slate-400">
            If reloading doesn&apos;t help, reset local data to clear cached app state (you&apos;ll need to sign in again).
          </p>
          <div className="flex justify-center gap-2">
            <button
              onClick={this.reload}
              className="cursor-pointer rounded-lg border border-slate-300 bg-white px-4 py-2 dark:border-slate-600 dark:bg-slate-900"
            >
              Reload
            </button>
            <button
              onClick={this.resetLocalData}
              className="cursor-pointer rounded-lg border-0 bg-red-600 px-4 py-2 text-white hover:bg-red-700"
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
