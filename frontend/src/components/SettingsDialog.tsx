import { useState } from 'react'
import { createPortal } from 'react-dom'
import { X, Mail, Lock, Loader2 } from 'lucide-react'
import { useAppStore } from '../stores/appStore'

interface SettingsDialogProps {
  open: boolean
  onClose: () => void
}

export function SettingsDialog({ open, onClose }: SettingsDialogProps) {
  const { authUser, updateProfileEmail, updateProfilePassword, language } = useAppStore()
  const [tab, setTab] = useState<'email' | 'password'>('email')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  // Email tab state
  const [email, setEmail] = useState(authUser?.email ?? '')
  const [newEmail, setNewEmail] = useState('')
  const [currentEmailPassword, setCurrentEmailPassword] = useState('')

  // Password tab state
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')

  const isReadOnly = authUser?.status === 'readonly'
  const zh = language === 'zh'

  const resetForms = () => {
    setEmail(authUser?.email ?? '')
    setNewEmail('')
    setCurrentEmailPassword('')
    setCurrentPassword('')
    setNewPassword('')
    setConfirmPassword('')
    setMessage(null)
  }

  const handleClose = () => {
    resetForms()
    onClose()
  }

  const handleUpdateEmail = async () => {
    if (!newEmail.trim() || !currentEmailPassword) return
    setLoading(true)
    setMessage(null)
    try {
      await updateProfileEmail(currentEmailPassword, newEmail.trim())
      setMessage({ type: 'success', text: zh ? '邮箱更新成功' : 'Email updated successfully' })
      setNewEmail('')
      setCurrentEmailPassword('')
    } catch (e: any) {
      setMessage({ type: 'error', text: e.message || (zh ? '更新失败' : 'Update failed') })
    } finally {
      setLoading(false)
    }
  }

  const handleUpdatePassword = async () => {
    if (!currentPassword || !newPassword) return
    if (newPassword !== confirmPassword) {
      setMessage({ type: 'error', text: zh ? '两次密码不一致' : 'Passwords do not match' })
      return
    }
    if (newPassword.length < 8) {
      setMessage({ type: 'error', text: zh ? '密码长度至少8位' : 'Password must be at least 8 characters' })
      return
    }
    setLoading(true)
    setMessage(null)
    try {
      await updateProfilePassword(currentPassword, newPassword)
      setMessage({ type: 'success', text: zh ? '密码更新成功' : 'Password updated successfully' })
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (e: any) {
      setMessage({ type: 'error', text: e.message || (zh ? '更新失败' : 'Update failed') })
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
      <div className="bg-white rounded-2xl shadow-modal w-full max-w-md mx-4 max-h-[85vh] flex flex-col overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200 bg-gradient-to-r from-accent-50 to-accent-100">
          <h2 className="text-lg font-display font-semibold text-surface-900">
            {zh ? '设置' : 'Settings'}
          </h2>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-surface-200 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-surface-500" />
          </button>
        </div>

        {/* Tabs */}
        {!isReadOnly && (
          <div className="flex border-b border-surface-200">
            <button
              onClick={() => { setTab('email'); setMessage(null) }}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                tab === 'email'
                  ? 'text-accent-600 border-b-2 border-accent-500'
                  : 'text-surface-500 hover:text-surface-700'
              }`}
            >
              <Mail className="w-4 h-4" />
              {zh ? '邮箱' : 'Email'}
            </button>
            <button
              onClick={() => { setTab('password'); setMessage(null) }}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                tab === 'password'
                  ? 'text-accent-600 border-b-2 border-accent-500'
                  : 'text-surface-500 hover:text-surface-700'
              }`}
            >
              <Lock className="w-4 h-4" />
              {zh ? '密码' : 'Password'}
            </button>
          </div>
        )}

        {/* Content */}
        <div className="px-6 py-5 space-y-4 overflow-y-auto">
          {/* Email Tab */}
          {tab === 'email' && !isReadOnly && (
            <>
              <div>
                <label className="block text-sm font-medium text-surface-700 mb-1.5">
                  {zh ? '当前邮箱' : 'Current Email'}
                </label>
                <input
                  type="email"
                  value={email}
                  readOnly
                  className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-surface-50 text-surface-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-surface-700 mb-1.5">
                  {zh ? '当前密码' : 'Current Password'}
                </label>
                <input
                  type="password"
                  value={currentEmailPassword}
                  onChange={(e) => setCurrentEmailPassword(e.target.value)}
                  placeholder={zh ? '输入当前密码' : 'Enter current password'}
                  className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-surface-700 mb-1.5">
                  {zh ? '新邮箱' : 'New Email'}
                </label>
                <input
                  type="email"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  placeholder={zh ? '输入新邮箱地址' : 'Enter new email address'}
                  className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
                />
              </div>
              <button
                onClick={handleUpdateEmail}
                disabled={loading || !newEmail.trim() || !currentEmailPassword}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent-600 text-white text-sm font-medium rounded-lg hover:bg-accent-700 disabled:opacity-50 transition-colors"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                {zh ? '更新邮箱' : 'Update Email'}
              </button>
            </>
          )}

          {/* Password Tab */}
          {tab === 'password' && (
            <>
              {!isReadOnly && (
                <div>
                  <label className="block text-sm font-medium text-surface-700 mb-1.5">
                    {zh ? '当前邮箱' : 'Current Email'}
                  </label>
                  <input
                    type="email"
                    value={email}
                    readOnly
                    className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-surface-50 text-surface-500"
                  />
                </div>
              )}
              {isReadOnly && (
                <p className="text-sm text-surface-500 mb-2">
                  {zh ? 'readonly 用户请通过管理员修改邮箱。' : 'readonly users please contact admin to change email.'}
                </p>
              )}
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
                  placeholder={zh ? '输入新密码（至少8位）' : 'Enter new password (min 8 characters)'}
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
              <button
                onClick={handleUpdatePassword}
                disabled={loading || !currentPassword || !newPassword || !confirmPassword}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent-600 text-white text-sm font-medium rounded-lg hover:bg-accent-700 disabled:opacity-50 transition-colors"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                {zh ? '更新密码' : 'Update Password'}
              </button>
            </>
          )}

          {/* Message */}
          {message && (
            <div className={`text-sm text-center py-2 px-3 rounded-lg ${
              message.type === 'success'
                ? 'bg-green-50 text-green-700'
                : 'bg-red-50 text-red-600'
            }`}>
              {message.text}
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body
  )
}
