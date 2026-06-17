import { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAppStore } from '../stores/appStore'
import { LoadingScreen } from './LoadingScreen'

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const authUser = useAppStore(s => s.authUser)
  const authReady = useAppStore(s => s.authReady)
  if (!authReady) return <LoadingScreen />
  if (!authUser) return <Navigate to="/login" replace />
  // Admins use the normal analysis app too; they reach the admin panel via the
  // sidebar "User Admin" link (see Sidebar). Do not force-redirect them to /admin.
  return <>{children}</>
}
