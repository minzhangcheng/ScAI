import { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAppStore } from '../stores/appStore'
import { LoadingScreen } from './LoadingScreen'

export function ProtectedAdminRoute({ children }: { children: ReactNode }) {
  const authUser = useAppStore(s => s.authUser)
  const authReady = useAppStore(s => s.authReady)
  if (!authReady) return <LoadingScreen />
  if (!authUser) return <Navigate to="/login" replace />
  if (authUser.role !== 'admin') return <Navigate to="/" replace />
  return <>{children}</>
}
