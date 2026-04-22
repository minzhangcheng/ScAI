import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAppStore } from '../stores/appStore'
import { AuthLayout } from '../components/AuthLayout'
import { DisclaimerContent } from '../components/DisclaimerDialog'

export function Register() {
  const nav = useNavigate()
  const register = useAppStore(s => s.register)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [accepted, setAccepted] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (password !== confirm) {
      setErr('Passwords do not match')
      return
    }
    setLoading(true)
    setErr('')
    try {
      await register(email, password)
      nav('/login?registered=pending', { replace: true })
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout subtitle="Create an account" onSubmit={handleSubmit}>
      <h2 className="text-lg font-semibold text-surface-700">Register</h2>

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

      <div>
        <label className="block text-sm font-medium text-surface-600 mb-1">Confirm Password</label>
        <input
          type="password"
          value={confirm}
          onChange={e => setConfirm(e.target.value)}
          required
          minLength={8}
          className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-accent-500 focus:border-accent-500"
          placeholder="••••••••"
        />
      </div>

      <DisclaimerContent />

      <label className="flex items-start gap-2 text-sm text-surface-600">
        <input
          type="checkbox"
          checked={accepted}
          onChange={e => setAccepted(e.target.checked)}
          className="mt-0.5 rounded border-surface-300 text-accent-600 focus:ring-accent-500"
        />
        <span>I have read and agree to the AI risk warning above</span>
      </label>

      <button
        type="submit"
        disabled={loading || !accepted || password !== confirm}
        className="w-full py-2.5 bg-accent-600 hover:bg-accent-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
      >
        {loading ? 'Creating account...' : 'Create Account'}
      </button>

      <p className="text-center text-sm text-surface-500">
        Already have an account?{' '}
        <Link to="/login" className="text-accent-600 hover:text-accent-700 font-medium">
          Sign In
        </Link>
      </p>
    </AuthLayout>
  )
}
