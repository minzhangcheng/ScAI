import { useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, X } from 'lucide-react'

import { ConfirmDialog } from '../../components/ConfirmDialog'
import { useAppStore } from '../../stores/appStore'
import type { AdminController, AdminUser, CreateUserData } from './useAdminController'

function DialogShell({ children, onClose, title }: {
  children: ReactNode
  onClose: () => void
  title: string
}) {
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal w-full max-w-sm mx-4 overflow-hidden animate-scale-in">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200 dark:border-surface-700">
          <h2 className="text-lg font-display font-semibold text-surface-900 dark:text-surface-100">{title}</h2>
          <button onClick={onClose} className="p-1 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg transition-colors"><X className="w-5 h-5 text-surface-400" /></button>
        </div>
        {children}
      </div>
    </div>,
    document.body,
  )
}

const inputClass = 'w-full px-3 py-2 text-sm border border-surface-200 dark:border-surface-700 rounded-lg bg-white dark:bg-surface-900 text-surface-800 dark:text-surface-100 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500'

function Field({ label, onChange, placeholder, type = 'text', value }: {
  label: string
  onChange: (value: string) => void
  placeholder: string
  type?: string
  value: string
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-surface-700 dark:text-surface-200 mb-1.5">{label}</label>
      <input type={type} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} className={inputClass} />
    </div>
  )
}

function DialogActions({ disabled, label, loading, onCancel, onConfirm }: {
  disabled: boolean
  label: string
  loading: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  const zh = useAppStore(s => s.language) === 'zh'
  return (
    <div className="flex justify-end gap-3 px-6 py-4 border-t border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-950">
      <button onClick={onCancel} className="px-4 py-2 text-sm font-medium text-surface-600 dark:text-surface-300 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg transition-colors">{zh ? '取消' : 'Cancel'}</button>
      <button onClick={onConfirm} disabled={disabled || loading} className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-accent-600 hover:bg-accent-700 rounded-lg transition-colors disabled:opacity-50">
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}{label}
      </button>
    </div>
  )
}

function PasswordPrompt({ open, onClose, onConfirm }: {
  open: boolean
  onClose: () => void
  onConfirm: (currentPassword: string, newPassword: string) => Promise<void>
}) {
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const zh = useAppStore(s => s.language) === 'zh'
  const close = () => {
    setCurrentPassword(''); setNewPassword(''); setConfirmPassword(''); setError(''); onClose()
  }
  const confirm = async () => {
    if (!currentPassword) return setError(zh ? '请输入当前密码' : 'Current password is required')
    if (!newPassword || newPassword.length < 8) return setError(zh ? '密码长度至少8位' : 'Password must be at least 8 characters')
    if (newPassword !== confirmPassword) return setError(zh ? '两次密码不一致' : 'Passwords do not match')
    setLoading(true); setError('')
    try { await onConfirm(currentPassword, newPassword); close() }
    catch (e: any) { setError(e.message || (zh ? '更新失败' : 'Update failed')) }
    finally { setLoading(false) }
  }
  if (!open) return null
  return (
    <DialogShell title={zh ? '修改密码' : 'Change My Password'} onClose={close}>
      <div className="px-6 py-5 space-y-4">
        <Field type="password" label={zh ? '当前密码' : 'Current Password'} value={currentPassword} onChange={setCurrentPassword} placeholder={zh ? '输入当前密码' : 'Enter current password'} />
        <Field type="password" label={zh ? '新密码' : 'New Password'} value={newPassword} onChange={setNewPassword} placeholder={zh ? '至少8位' : 'Min 8 characters'} />
        <Field type="password" label={zh ? '确认新密码' : 'Confirm New Password'} value={confirmPassword} onChange={setConfirmPassword} placeholder={zh ? '再次输入新密码' : 'Confirm new password'} />
        {error && <div className="text-sm text-red-600 dark:text-red-400 py-2 px-3 bg-red-50 dark:bg-red-950/40 rounded-lg">{error}</div>}
      </div>
      <DialogActions disabled={!currentPassword || !newPassword || !confirmPassword} label={zh ? '保存' : 'Save'} loading={loading} onCancel={close} onConfirm={confirm} />
    </DialogShell>
  )
}

function CreateUserPrompt({ open, onClose, onConfirm }: {
  open: boolean
  onClose: () => void
  onConfirm: (data: CreateUserData) => Promise<void>
}) {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const zh = useAppStore(s => s.language) === 'zh'
  const close = () => { setUsername(''); setEmail(''); setPassword(''); setError(''); onClose() }
  const confirm = async () => {
    if (!username.trim()) return setError(zh ? '请输入用户名' : 'Username is required')
    if (!password || password.length < 8) return setError(zh ? '密码长度至少8位' : 'Password must be at least 8 characters')
    setLoading(true); setError('')
    try { await onConfirm({ username: username.trim(), password, email: email.trim() || undefined }); close() }
    catch (e: any) { setError(e.message || (zh ? '创建失败' : 'Create failed')) }
    finally { setLoading(false) }
  }
  if (!open) return null
  return (
    <DialogShell title={zh ? '创建用户' : 'Create User'} onClose={close}>
      <div className="px-6 py-5 space-y-4">
        <Field label={zh ? '用户名' : 'Username'} value={username} onChange={setUsername} placeholder={zh ? '输入用户名' : 'Enter username'} />
        <Field type="email" label={zh ? '邮箱（可选）' : 'Email (optional)'} value={email} onChange={setEmail} placeholder="user@example.com" />
        <Field type="password" label={zh ? '密码' : 'Password'} value={password} onChange={setPassword} placeholder={zh ? '至少8位' : 'Min 8 characters'} />
        {error && <div className="text-sm text-red-600 dark:text-red-400 py-2 px-3 bg-red-50 dark:bg-red-950/40 rounded-lg">{error}</div>}
      </div>
      <DialogActions disabled={!username.trim() || !password} label={zh ? '创建' : 'Create'} loading={loading} onCancel={close} onConfirm={confirm} />
    </DialogShell>
  )
}

function AdminPasswordPrompt({ open, user, onClose, onConfirm }: {
  open: boolean
  user: AdminUser | null
  onClose: () => void
  onConfirm: (userId: string, newPassword: string) => Promise<void>
}) {
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const zh = useAppStore(s => s.language) === 'zh'
  const close = () => { setNewPassword(''); setConfirmPassword(''); setError(''); onClose() }
  const confirm = async () => {
    if (!user) return
    if (!newPassword || newPassword.length < 8) return setError(zh ? '密码长度至少8位' : 'Password must be at least 8 characters')
    if (newPassword !== confirmPassword) return setError(zh ? '两次密码不一致' : 'Passwords do not match')
    setLoading(true); setError('')
    try { await onConfirm(user.id, newPassword); close() }
    catch (e: any) { setError(e.message || (zh ? '更新失败' : 'Update failed')) }
    finally { setLoading(false) }
  }
  if (!open || !user) return null
  return (
    <DialogShell title={zh ? `修改 ${user.username} 的密码` : `Change Password for ${user.username}`} onClose={close}>
      <div className="px-6 py-5 space-y-4">
        <Field type="password" label={zh ? '新密码' : 'New Password'} value={newPassword} onChange={setNewPassword} placeholder={zh ? '至少8位' : 'Min 8 characters'} />
        <Field type="password" label={zh ? '确认新密码' : 'Confirm New Password'} value={confirmPassword} onChange={setConfirmPassword} placeholder={zh ? '再次输入新密码' : 'Confirm new password'} />
        {error && <div className="text-sm text-red-600 dark:text-red-400 py-2 px-3 bg-red-50 dark:bg-red-950/40 rounded-lg">{error}</div>}
      </div>
      <DialogActions disabled={!newPassword || !confirmPassword} label={zh ? '保存' : 'Save'} loading={loading} onCancel={close} onConfirm={confirm} />
    </DialogShell>
  )
}

export function AdminDialogs({ controller }: { controller: AdminController }) {
  const {
    changePasswordTarget, deleteTarget, handleAdminChangePassword,
    handleChangePassword, handleCreateUser, handleDeleteUser,
    setChangePasswordTarget, setDeleteTarget, setShowCreateUser,
    setShowPasswordPrompt, showCreateUser, showPasswordPrompt, zh,
  } = controller
  return (
    <>
      <PasswordPrompt open={showPasswordPrompt} onClose={() => setShowPasswordPrompt(false)} onConfirm={handleChangePassword} />
      <CreateUserPrompt open={showCreateUser} onClose={() => setShowCreateUser(false)} onConfirm={handleCreateUser} />
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
    </>
  )
}
