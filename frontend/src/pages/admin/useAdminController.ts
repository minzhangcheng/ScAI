import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { apiFetch } from '../../hooks/useApi'
import { useAppStore } from '../../stores/appStore'

export interface AdminUser {
  id: string
  username: string
  email?: string
  role: 'admin' | 'user'
  status: 'active' | 'pending' | 'readonly' | 'deleting'
  created_at: string
}

export interface CreateUserData {
  username: string
  password: string
  email?: string
}

export function useAdminController() {
  const logout = useAppStore(s => s.logout)
  const language = useAppStore(s => s.language)
  const authUser = useAppStore(s => s.authUser)
  const navigate = useNavigate()
  const [users, setUsers] = useState<AdminUser[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showPasswordPrompt, setShowPasswordPrompt] = useState(false)
  const [showCreateUser, setShowCreateUser] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<AdminUser | null>(null)
  const [changePasswordTarget, setChangePasswordTarget] = useState<AdminUser | null>(null)
  const zh = language === 'zh'

  const fetchUsers = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setUsers(await apiFetch<AdminUser[]>('/admin/users'))
    } catch (e: any) {
      setError(e.message || 'Failed to load users')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchUsers() }, [fetchUsers])

  const handleSignOut = () => {
    logout()
    navigate('/login')
  }
  const handleChangePassword = async (currentPassword: string, newPassword: string) => {
    await apiFetch('/auth/profile/password', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    })
  }
  const handleCreateUser = async (data: CreateUserData) => {
    const body: Record<string, string> = { username: data.username, password: data.password }
    if (data.email) body.email = data.email
    await apiFetch('/admin/users', { method: 'POST', body: JSON.stringify(body) })
    await fetchUsers()
  }
  const handleDeleteUser = async () => {
    if (!deleteTarget) return
    let deletionError = ''
    try {
      await apiFetch(`/admin/users/${deleteTarget.id}`, { method: 'DELETE' })
    } catch (e: any) {
      deletionError = e.message || (zh ? '删除清理尚未完成' : 'Deletion cleanup is not complete')
    }
    setDeleteTarget(null)
    await fetchUsers()
    if (deletionError) setError(deletionError)
  }
  const handleStatusChange = async (userId: string, status: string) => {
    await apiFetch(`/admin/users/${userId}/status`, {
      method: 'PATCH', body: JSON.stringify({ status }),
    })
    await fetchUsers()
  }
  const handleRoleChange = async (userId: string, role: string) => {
    await apiFetch(`/admin/users/${userId}/role`, {
      method: 'PATCH', body: JSON.stringify({ role }),
    })
    await fetchUsers()
  }
  const handleAdminChangePassword = async (userId: string, newPassword: string) => {
    await apiFetch(`/admin/users/${userId}/password`, {
      method: 'POST', body: JSON.stringify({ new_password: newPassword }),
    })
  }
  const formatDate = (dateStr: string) => new Date(dateStr).toLocaleDateString(zh ? 'zh-CN' : 'en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
  })

  return {
    authUser, changePasswordTarget, deleteTarget, error, formatDate,
    handleAdminChangePassword, handleChangePassword, handleCreateUser,
    handleDeleteUser, handleRoleChange, handleSignOut, handleStatusChange,
    loading, navigate, setChangePasswordTarget, setDeleteTarget, setShowCreateUser,
    setShowPasswordPrompt, showCreateUser, showPasswordPrompt, users, zh,
  }
}

export type AdminController = ReturnType<typeof useAdminController>
