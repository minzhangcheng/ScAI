import { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAppStore } from '../stores/appStore'

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const authUser = useAppStore(s => s.authUser)
  if (!authUser) return <Navigate to="/login" replace />
  return <>{children}</>
}
