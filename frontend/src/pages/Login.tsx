import { useState, useEffect } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAppStore } from '../stores/appStore'
import { AuthLayout } from '../components/AuthLayout'

export function Login() {
  const nav = useNavigate()
  const login = useAppStore(s => s.login)
  const language = useAppStore(s => s.language)
  const zh = language === 'zh'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)
  const [searchParams] = useSearchParams()
  const registered = searchParams.get('registered')

  useEffect(() => {
    fetch('/api/auth/setup/status')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data?.initialized) {
          nav('/setup', { replace: true })
        }
      })
      .catch(() => {})
  }, [nav])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setErr('')
    try {
      await login(username.trim(), password)
      nav('/')
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout subtitle="Single Cell Sequencing Analysis Driven by AI" onSubmit={handleSubmit}>
      <h2 className="text-lg font-semibold text-surface-700">{zh ? '登录' : 'Sign In'}</h2>

      {err && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">{err}</div>
      )}

      {registered === 'pending' && (
        <div className="p-3 bg-green-50 border border-green-200 rounded-lg text-green-700 text-sm">
          {zh ? '账号创建成功，请登录。' : 'Account created successfully. Please sign in.'}
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-surface-600 mb-1">{zh ? '用户名' : 'Username'}</label>
        <input
          type="text"
          name="username"
          value={username}
          onChange={e => setUsername(e.target.value)}
          required
          className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
          placeholder={zh ? '输入用户名' : 'Enter your username'}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-surface-600 mb-1">{zh ? '密码' : 'Password'}</label>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          minLength={8}
          className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
          placeholder="••••••••"
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2.5 bg-accent-600 hover:bg-accent-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
      >
        {loading ? (zh ? '登录中...' : 'Signing in...') : (zh ? '登录' : 'Sign In')}
      </button>

      <p className="text-center text-sm text-surface-500">
        {zh ? '没有账号？' : "Don't have an account?"}{' '}
        <Link to="/register" className="text-accent-600 hover:text-accent-700 font-medium">
          {zh ? '注册' : 'Register'}
        </Link>
      </p>
    </AuthLayout>
  )
}
