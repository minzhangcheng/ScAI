import { ReactNode, useEffect } from 'react'
import { useAppStore } from '../stores/appStore'
import { LoadingScreen } from './LoadingScreen'

export function AuthInitializer({ children }: { children: ReactNode }) {
  const authReady = useAppStore(s => s.authReady)
  const loadAuthFromStorage = useAppStore(s => s.loadAuthFromStorage)

  useEffect(() => {
    loadAuthFromStorage()
  }, [loadAuthFromStorage])

  if (!authReady) return <LoadingScreen />

  return <>{children}</>
}