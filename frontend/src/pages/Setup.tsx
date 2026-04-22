import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { AuthLayout } from '../components/AuthLayout'

export function Setup() {
  const nav = useNavigate()
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch('/api/auth/setup/status')
      .then(r => r.ok ? r.json() : null)
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
      setErr('Passwords do not match')
      return
    }
    setLoading(true)
    setErr('')
    try {
      const r = await fetch('/api/auth/setup/admin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      })
      if (!r.ok) {
        const body = await r.json().catch(() => ({ detail: 'Setup failed' }))
        throw new Error(body.detail || 'Setup failed')
      }
      nav('/login', { replace: true })
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Setup failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout subtitle="Initialize Admin Account" onSubmit={handleSubmit}>
      <h2 className="text-lg font-semibold text-surface-700">Setup</h2>

      {err && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">{err}</div>
      )}

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

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2.5 bg-accent-600 hover:bg-accent-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
      >
        {loading ? 'Setting up...' : 'Create Admin Account'}
      </button>
    </AuthLayout>
  )
}
