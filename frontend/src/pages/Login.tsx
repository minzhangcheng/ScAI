import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAppStore } from '../stores/appStore'
import { AuthLayout } from '../components/AuthLayout'

export function Login() {
  const nav = useNavigate()
  const login = useAppStore(s => s.login)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setErr('')
    try {
      await login(email, password)
      nav('/')
    } catch {
      setErr('Invalid email or password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout subtitle="Single Cell Sequencing Analysis Driven by AI" onSubmit={handleSubmit}>
      <h2 className="text-lg font-semibold text-surface-700">Sign In</h2>

      {err && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">{err}</div>
      )}

      <div>
        <label className="block text-sm font-medium text-surface-600 mb-1">Email</label>
        <input
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
          placeholder="you@example.com"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-surface-600 mb-1">Password</label>
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
        {loading ? 'Signing in...' : 'Sign In'}
      </button>

      <p className="text-center text-sm text-surface-500">
        Don't have an account?{' '}
        <Link to="/register" className="text-accent-600 hover:text-accent-700 font-medium">
          Register
        </Link>
      </p>
    </AuthLayout>
  )
}
