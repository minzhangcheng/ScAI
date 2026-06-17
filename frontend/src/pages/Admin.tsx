import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { createPortal } from 'react-dom'
import { UserPlus, Key, LogOut, X, Loader2, User } from 'lucide-react'
import { useAppStore } from '../stores/appStore'
import { apiFetch } from '../hooks/useApi'
import { ConfirmDialog } from '../components/ConfirmDialog'

interface AdminUser {
  id: string
  username: string
  email?: string
  role: 'admin' | 'user'
  status: 'active' | 'pending' | 'readonly'
  created_at: string
}

// Password Prompt Dialog (for changing own password)
interface PasswordPromptProps {
  open: boolean
  onClose: () => void
  onConfirm: (currentPassword: string, newPassword: string) => Promise<void>
}

function PasswordPrompt({ open, onClose, onConfirm }: PasswordPromptProps) {
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const language = useAppStore(s => s.language)
  const zh = language === 'zh'

  const reset = () => {
    setCurrentPassword('')
    setNewPassword('')
    setConfirmPassword('')
    setError('')
  }

  const handleClose = () => {
    reset()
    onClose()
  }

  const handleConfirm = async () => {
    if (!currentPassword) {
      setError(zh ? '请输入当前密码' : 'Current password is required')
      return
    }
    if (!newPassword || newPassword.length < 8) {
      setError(zh ? '密码长度至少8位' : 'Password must be at least 8 characters')
      return
    }
    if (newPassword !== confirmPassword) {
      setError(zh ? '两次密码不一致' : 'Passwords do not match')
      return
    }
    setLoading(true)
    setError('')
    try {
      await onConfirm(currentPassword, newPassword)
      handleClose()
    } catch (e: any) {
      setError(e.message || (zh ? '更新失败' : 'Update failed'))
    } finally {
      setLoading(false)
    }
  }

  if (!open) return null

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) handleClose() }}
    >
      <div className="bg-white rounded-2xl shadow-modal w-full max-w-sm mx-4 overflow-hidden animate-scale-in">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200">
          <h2 className="text-lg font-display font-semibold text-surface-900">
            {zh ? '修改密码' : 'Change My Password'}
          </h2>
          <button onClick={handleClose} className="p-1 hover:bg-surface-100 rounded-lg transition-colors">
            <X className="w-5 h-5 text-surface-400" />
          </button>
        </div>
        <div className="px-6 py-5 space-y-4">
          <div>
            <label className="block text-sm font-medium text-surface-700 mb-1.5">
              {zh ? '当前密码' : 'Current Password'}
            </label>
            <input
              type="password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              placeholder={zh ? '输入当前密码' : 'Enter current password'}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-surface-700 mb-1.5">
              {zh ? '新密码' : 'New Password'}
            </label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder={zh ? '至少8位' : 'Min 8 characters'}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-surface-700 mb-1.5">
              {zh ? '确认新密码' : 'Confirm New Password'}
            </label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder={zh ? '再次输入新密码' : 'Confirm new password'}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
          </div>
          {error && (
            <div className="text-sm text-red-600 py-2 px-3 bg-red-50 rounded-lg">{error}</div>
          )}
        </div>
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-surface-200 bg-surface-50">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm font-medium text-surface-600 hover:bg-surface-100 rounded-lg transition-colors"
          >
            {zh ? '取消' : 'Cancel'}
          </button>
          <button
            onClick={handleConfirm}
            disabled={loading || !currentPassword || !newPassword || !confirmPassword}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-accent-600 hover:bg-accent-700 rounded-lg transition-colors disabled:opacity-50"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {zh ? '保存' : 'Save'}
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}

// Create User Dialog
interface CreateUserPromptProps {
  open: boolean
  onClose: () => void
  onConfirm: (data: { username: string; password: string; email?: string }) => Promise<void>
}

function CreateUserPrompt({ open, onClose, onConfirm }: CreateUserPromptProps) {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const language = useAppStore(s => s.language)
  const zh = language === 'zh'

  const reset = () => {
    setUsername('')
    setEmail('')
    setPassword('')
    setError('')
  }

  const handleClose = () => {
    reset()
    onClose()
  }

  const handleConfirm = async () => {
    if (!username.trim()) {
      setError(zh ? '请输入用户名' : 'Username is required')
      return
    }
    if (!password || password.length < 8) {
      setError(zh ? '密码长度至少8位' : 'Password must be at least 8 characters')
      return
    }
    setLoading(true)
    setError('')
    try {
      await onConfirm({ username: username.trim(), password, email: email.trim() || undefined })
      handleClose()
    } catch (e: any) {
      setError(e.message || (zh ? '创建失败' : 'Create failed'))
    } finally {
      setLoading(false)
    }
  }

  if (!open) return null

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) handleClose() }}
    >
      <div className="bg-white rounded-2xl shadow-modal w-full max-w-sm mx-4 overflow-hidden animate-scale-in">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200">
          <h2 className="text-lg font-display font-semibold text-surface-900">
            {zh ? '创建用户' : 'Create User'}
          </h2>
          <button onClick={handleClose} className="p-1 hover:bg-surface-100 rounded-lg transition-colors">
            <X className="w-5 h-5 text-surface-400" />
          </button>
        </div>
        <div className="px-6 py-5 space-y-4">
          <div>
            <label className="block text-sm font-medium text-surface-700 mb-1.5">
              {zh ? '用户名' : 'Username'}
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder={zh ? '输入用户名' : 'Enter username'}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-surface-700 mb-1.5">
              {zh ? '邮箱（可选）' : 'Email (optional)'}
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="user@example.com"
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-surface-700 mb-1.5">
              {zh ? '密码' : 'Password'}
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={zh ? '至少8位' : 'Min 8 characters'}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
          </div>
          {error && (
            <div className="text-sm text-red-600 py-2 px-3 bg-red-50 rounded-lg">{error}</div>
          )}
        </div>
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-surface-200 bg-surface-50">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm font-medium text-surface-600 hover:bg-surface-100 rounded-lg transition-colors"
          >
            {zh ? '取消' : 'Cancel'}
          </button>
          <button
            onClick={handleConfirm}
            disabled={loading || !username.trim() || !password}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-accent-600 hover:bg-accent-700 rounded-lg transition-colors disabled:opacity-50"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {zh ? '创建' : 'Create'}
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}

// Editable enum cell: a read-only badge for the current user's own row, an
// inline <select> for everyone else.
function EnumCell({ isSelf, value, options, onChange }: {
  isSelf: boolean
  value: string
  options: string[]
  onChange: (value: string) => void
}) {
  if (isSelf) {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-700">
        {value}
      </span>
    )
  }
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="text-sm border border-surface-200 rounded-lg px-2 py-1 bg-white focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
    >
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  )
}

export function Admin() {
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
      const data = await apiFetch<AdminUser[]>('/admin/users')
      setUsers(data)
    } catch (e: any) {
      setError(e.message || 'Failed to load users')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

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

  const handleCreateUser = async (data: { username: string; password: string; email?: string }) => {
    const body: Record<string, string> = { username: data.username, password: data.password }
    if (data.email) body.email = data.email
    await apiFetch('/admin/users', {
      method: 'POST',
      body: JSON.stringify(body),
    })
    await fetchUsers()
  }

  const handleDeleteUser = async () => {
    if (!deleteTarget) return
    await apiFetch(`/admin/users/${deleteTarget.id}`, {
      method: 'DELETE',
    })
    setDeleteTarget(null)
    await fetchUsers()
  }

  const handleStatusChange = async (userId: string, newStatus: string) => {
    await apiFetch(`/admin/users/${userId}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ status: newStatus }),
    })
    await fetchUsers()
  }

  const handleRoleChange = async (userId: string, newRole: string) => {
    await apiFetch(`/admin/users/${userId}/role`, {
      method: 'PATCH',
      body: JSON.stringify({ role: newRole }),
    })
    await fetchUsers()
  }

  const handleAdminChangePassword = async (userId: string, newPassword: string) => {
    await apiFetch(`/admin/users/${userId}/password`, {
      method: 'POST',
      body: JSON.stringify({ new_password: newPassword }),
    })
  }

  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr)
    return d.toLocaleDateString(zh ? 'zh-CN' : 'en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  return (
    <div className="min-h-screen bg-surface-50">
      {/* Header */}
      <header className="bg-white border-b border-surface-200 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent-500 to-accent-700 flex items-center justify-center">
              <User className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-display font-semibold text-surface-900">
              {zh ? '管理员面板' : 'Admin Panel'}
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/')}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-surface-600 hover:bg-surface-100 rounded-lg transition-colors"
            >
              {zh ? '返回' : 'Back'}
            </button>
            <button
              onClick={() => setShowPasswordPrompt(true)}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-surface-600 hover:bg-surface-100 rounded-lg transition-colors"
            >
              <Key className="w-4 h-4" />
              {zh ? '修改我的密码' : 'Change My Password'}
            </button>
            <button
              onClick={handleSignOut}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <LogOut className="w-4 h-4" />
              {zh ? '退出登录' : 'Sign Out'}
            </button>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-surface-800">
            {zh ? '用户管理' : 'User Management'}
          </h2>
          <button
            onClick={() => setShowCreateUser(true)}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-accent-600 hover:bg-accent-700 rounded-lg transition-colors"
          >
            <UserPlus className="w-4 h-4" />
            {zh ? '创建用户' : 'Create User'}
          </button>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
            {error}
          </div>
        )}

        {/* Users Table */}
        <div className="bg-white rounded-xl border border-surface-200 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-surface-50 border-b border-surface-200">
                <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 uppercase tracking-wider">
                  {zh ? '用户名' : 'Username'}
                </th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 uppercase tracking-wider">
                  Role
                </th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 uppercase tracking-wider">
                  {zh ? '创建时间' : 'Created'}
                </th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 uppercase tracking-wider">
                  {zh ? '操作' : 'Actions'}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-100">
              {loading ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-surface-400">
                    <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                    <p className="text-sm">{zh ? '加载中...' : 'Loading...'}</p>
                  </td>
                </tr>
              ) : users.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-surface-400 text-sm">
                    {zh ? '暂无用户' : 'No users found'}
                  </td>
                </tr>
              ) : (
                users.map((user) => {
                  const isSelf = user.id === authUser?.id
                  return (
                    <tr key={user.id} className="hover:bg-surface-50 transition-colors">
                      <td className="px-6 py-4">
                        <div>
                          <p className="text-sm font-medium text-surface-900">{user.username}</p>
                          {user.email && (
                            <p className="text-xs text-surface-500">{user.email}</p>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <EnumCell
                          isSelf={isSelf}
                          value={user.role}
                          options={['user', 'admin']}
                          onChange={(role) => handleRoleChange(user.id, role)}
                        />
                      </td>
                      <td className="px-6 py-4">
                        <EnumCell
                          isSelf={isSelf}
                          value={user.status}
                          options={user.role === 'admin' ? ['readonly', 'active'] : ['pending', 'readonly', 'active']}
                          onChange={(status) => handleStatusChange(user.id, status)}
                        />
                      </td>
                      <td className="px-6 py-4 text-sm text-surface-600">
                        {formatDate(user.created_at)}
                      </td>
                      <td className="px-6 py-4">
                        {isSelf ? (
                          <span className="text-xs text-surface-400 italic">
                            {zh ? '本人' : 'You'}
                          </span>
                        ) : (
                          <div className="flex items-center gap-4">
                            <button
                              onClick={() => setChangePasswordTarget(user)}
                              className="text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors"
                            >
                              {zh ? '改密' : 'Password'}
                            </button>
                            <button
                              onClick={() => setDeleteTarget(user)}
                              className="text-sm text-red-600 hover:text-red-700 font-medium transition-colors"
                            >
                              {zh ? '删除' : 'Delete'}
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </main>

      {/* Dialogs */}
      <PasswordPrompt
        open={showPasswordPrompt}
        onClose={() => setShowPasswordPrompt(false)}
        onConfirm={handleChangePassword}
      />

      <CreateUserPrompt
        open={showCreateUser}
        onClose={() => setShowCreateUser(false)}
        onConfirm={handleCreateUser}
      />

      <ConfirmDialog
        open={deleteTarget !== null}
        title={zh ? '确认删除' : 'Confirm Delete'}
        message={deleteTarget ? `${zh ? '确定要删除用户' : 'Are you sure you want to delete user'} "${deleteTarget.username}"?` : ''}
        confirmLabel={zh ? '删除' : 'Delete'}
        variant="danger"
        onConfirm={handleDeleteUser}
        onCancel={() => setDeleteTarget(null)}
      />

      <AdminPasswordPrompt
        open={changePasswordTarget !== null}
        user={changePasswordTarget}
        onClose={() => setChangePasswordTarget(null)}
        onConfirm={handleAdminChangePassword}
      />
    </div>
  )
}

// Admin Password Prompt Dialog (for changing any user's password)
interface AdminPasswordPromptProps {
  open: boolean
  user: AdminUser | null
  onClose: () => void
  onConfirm: (userId: string, newPassword: string) => Promise<void>
}

function AdminPasswordPrompt({ open, user, onClose, onConfirm }: AdminPasswordPromptProps) {
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const language = useAppStore(s => s.language)
  const zh = language === 'zh'

  const reset = () => {
    setNewPassword('')
    setConfirmPassword('')
    setError('')
  }

  const handleClose = () => {
    reset()
    onClose()
  }

  const handleConfirm = async () => {
    if (!user) return
    if (!newPassword || newPassword.length < 8) {
      setError(zh ? '密码长度至少8位' : 'Password must be at least 8 characters')
      return
    }
    if (newPassword !== confirmPassword) {
      setError(zh ? '两次密码不一致' : 'Passwords do not match')
      return
    }
    setLoading(true)
    setError('')
    try {
      await onConfirm(user.id, newPassword)
      handleClose()
    } catch (e: any) {
      setError(e.message || (zh ? '更新失败' : 'Update failed'))
    } finally {
      setLoading(false)
    }
  }

  if (!open || !user) return null

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) handleClose() }}
    >
      <div className="bg-white rounded-2xl shadow-modal w-full max-w-sm mx-4 overflow-hidden animate-scale-in">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200">
          <h2 className="text-lg font-display font-semibold text-surface-900">
            {zh ? `修改 ${user.username} 的密码` : `Change Password for ${user.username}`}
          </h2>
          <button onClick={handleClose} className="p-1 hover:bg-surface-100 rounded-lg transition-colors">
            <X className="w-5 h-5 text-surface-400" />
          </button>
        </div>
        <div className="px-6 py-5 space-y-4">
          <div>
            <label className="block text-sm font-medium text-surface-700 mb-1.5">
              {zh ? '新密码' : 'New Password'}
            </label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder={zh ? '至少8位' : 'Min 8 characters'}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-surface-700 mb-1.5">
              {zh ? '确认新密码' : 'Confirm New Password'}
            </label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder={zh ? '再次输入新密码' : 'Confirm new password'}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
          </div>
          {error && (
            <div className="text-sm text-red-600 py-2 px-3 bg-red-50 rounded-lg">{error}</div>
          )}
        </div>
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-surface-200 bg-surface-50">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm font-medium text-surface-600 hover:bg-surface-100 rounded-lg transition-colors"
          >
            {zh ? '取消' : 'Cancel'}
          </button>
          <button
            onClick={handleConfirm}
            disabled={loading || !newPassword || !confirmPassword}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-accent-600 hover:bg-accent-700 rounded-lg transition-colors disabled:opacity-50"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {zh ? '保存' : 'Save'}
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}
