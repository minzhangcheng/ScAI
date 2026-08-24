import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../stores/appStore'
import { AuthLayout } from '../components/AuthLayout'
import { apiFetch } from '../hooks/useApi'

export function Setup() {
  const nav = useNavigate()
  const language = useAppStore(s => s.language)
  const zh = language === 'zh'
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    apiFetch<{ initialized: boolean }>('/auth/setup/status')
      .then(data => {
        if (data?.initialized) {
          nav('/login', { replace: true })
        }
      })
      .catch(() => {})
  }, [nav])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (password !== confirm) {
      setErr(zh ? '两次密码不一致' : 'Passwords do not match')
      return
    }
    setLoading(true)
    setErr('')
    try {
      const body: Record<string, string> = { username: username.trim(), password }
      if (email.trim()) {
        body.email = email.trim()
      }
      await apiFetch('/auth/setup/admin', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      nav('/login', { replace: true })
    } catch (e) {
      setErr(e instanceof Error ? e.message : (zh ? '设置失败' : 'Setup failed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout subtitle={zh ? '初始化管理员账号' : 'Initialize Admin Account'} onSubmit={handleSubmit}>
      <h2 className="text-lg font-semibold text-surface-700 dark:text-surface-200">{zh ? '初始化' : 'Setup'}</h2>

      {err && (
        <div className="p-3 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300 text-sm">{err}</div>
      )}

      <div>
        <label className="block text-sm font-medium text-surface-600 dark:text-surface-300 mb-1">{zh ? '用户名' : 'Username'}</label>
        <input
          type="text"
          value={username}
          onChange={e => setUsername(e.target.value)}
          required
          minLength={3}
          maxLength={64}
          pattern="[a-zA-Z0-9_-]+"
          title={zh ? '只能包含字母、数字、下划线和连字符' : 'Only letters, numbers, underscore and hyphen are allowed'}
          className="w-full px-3 py-2 border border-surface-300 dark:border-surface-600 rounded-lg focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
          placeholder={zh ? '管理员用户名' : 'Admin username'}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-surface-600 dark:text-surface-300 mb-1">{zh ? '邮箱（可选）' : 'Email (optional)'}</label>
        <input
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          className="w-full px-3 py-2 border border-surface-300 dark:border-surface-600 rounded-lg focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
          placeholder="admin@example.com"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-surface-600 dark:text-surface-300 mb-1">{zh ? '密码' : 'Password'}</label>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          minLength={8}
          className="w-full px-3 py-2 border border-surface-300 dark:border-surface-600 rounded-lg focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
          placeholder="••••••••"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-surface-600 dark:text-surface-300 mb-1">{zh ? '确认密码' : 'Confirm Password'}</label>
        <input
          type="password"
          value={confirm}
          onChange={e => setConfirm(e.target.value)}
          required
          minLength={8}
          className="w-full px-3 py-2 border border-surface-300 dark:border-surface-600 rounded-lg focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
          placeholder="••••••••"
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2.5 bg-accent-600 hover:bg-accent-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
      >
        {loading ? (zh ? '设置中...' : 'Setting up...') : (zh ? '创建管理员账号' : 'Create Admin Account')}
      </button>
    </AuthLayout>
  )
}
