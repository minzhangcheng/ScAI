import { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAppStore } from '../stores/appStore'

export function ProtectedAdminRoute({ children }: { children: ReactNode }) {
  const authUser = useAppStore(s => s.authUser)
  if (!authUser) return <Navigate to="/login" replace />
  if (authUser.role !== 'admin') return <Navigate to="/" replace />
  return <>{children}</>
}
