import { useState, useEffect } from 'react'
import { useAppStore } from '../stores/appStore'
import { apiFetch } from '../hooks/useApi'
import { Key, Plus, Trash2 } from 'lucide-react'

interface ApiKeyItem {
  id: string
  label: string
  created_at: string
  last_used_at: string | null
}

export function ApiKeys() {
  const logout = useAppStore(s => s.logout)
  const authUser = useAppStore(s => s.authUser)
  const language = useAppStore(s => s.language)
  const zh = language === 'zh'
  const [keys, setKeys] = useState<ApiKeyItem[]>([])
  const [loading, setLoading] = useState(false)
  const [showCreate, setShowCreate] = useState(false)
  const [newLabel, setNewLabel] = useState('')
  const [newKey, setNewKey] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)

  const loadKeys = async () => {
    setLoading(true)
    try {
      const data = await apiFetch<ApiKeyItem[]>('/auth/api-keys')
      setKeys(data)
      setLoaded(true)
    } catch (e) {
      console.error('Failed to load API keys:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadKeys() }, [])

  if (!loaded) {
    return <div className="p-6 text-surface-500">Loading...</div>
  }

  const handleCreate = async () => {
    if (!newLabel.trim()) return
    try {
      const result = await apiFetch<{ id: string; key: string; label: string; created_at: string }>('/auth/api-keys', {
        method: 'POST',
        body: JSON.stringify({ label: newLabel }),
      })
      setNewKey(result.key)
      setNewLabel('')
      loadKeys()
    } catch (e) {
      console.error('Failed to create API key:', e)
    }
  }

  const handleRevoke = async (id: string) => {
    try {
      await apiFetch(`/auth/api-keys/${id}`, { method: 'DELETE' })
      loadKeys()
    } catch (e) {
      console.error('Failed to revoke API key:', e)
    }
  }

  return (
    <div className="min-h-screen bg-surface-50 p-6">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-surface-800">{zh ? 'API 密钥' : 'API Keys'}</h1>
            <p className="text-surface-500 mt-1">{zh ? '管理用于程序化访问的 API 密钥' : 'Manage API keys for programmatic access'}</p>
            <p className="text-surface-400 text-sm">{zh ? '登录用户：' : 'Logged in as '}{authUser?.username}</p>
          </div>
          <button
            onClick={() => (window.location.href = '/')}
            className="px-4 py-2 text-sm text-surface-600 hover:text-surface-800"
          >
            {zh ? '返回应用' : 'Back to App'}
          </button>
        </div>

        {newKey && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
            <p className="text-amber-800 font-medium mb-1">{zh ? '⚠️ 请立即复制您的 API 密钥 — 之后将不再显示！' : '⚠️ Copy your API key now — it will not be shown again!'}</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 p-2 bg-white rounded border text-sm font-mono break-all select-all">{newKey}</code>
            </div>
            <button onClick={() => setNewKey(null)} className="mt-2 text-sm text-amber-700 hover:text-amber-900">
              {zh ? '我已保存密钥' : "I've saved my key"}
            </button>
          </div>
        )}

        <div className="bg-white rounded-xl shadow-sm border border-surface-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-surface-700">Your API Keys</h2>
            <button
              onClick={() => setShowCreate(!showCreate)}
              className="flex items-center gap-1 px-3 py-1.5 bg-accent-600 text-white text-sm rounded-lg hover:bg-accent-700"
            >
              <Plus className="w-4 h-4" /> New Key
            </button>
          </div>

          {showCreate && (
            <div className="flex items-center gap-2 mb-4 p-3 bg-surface-50 rounded-lg">
              <input
                type="text"
                value={newLabel}
                onChange={e => setNewLabel(e.target.value)}
                placeholder="Key label (e.g., 'my-script')"
                className="flex-1 px-3 py-2 border border-surface-300 rounded-lg text-sm"
              />
              <button
                onClick={handleCreate}
                disabled={!newLabel.trim()}
                className="px-4 py-2 bg-accent-600 text-white text-sm rounded-lg hover:bg-accent-700 disabled:opacity-50"
              >
                Create
              </button>
            </div>
          )}

          {loading ? (
            <p className="text-surface-400">Loading...</p>
          ) : keys.length === 0 ? (
            <p className="text-surface-400 text-center py-8">{zh ? '暂无 API 密钥' : 'No API keys yet. Create one to get started.'}</p>
          ) : (
            <div className="divide-y divide-surface-100">
              {keys.map(k => (
                <div key={k.id} className="flex items-center justify-between py-3">
                  <div className="flex items-center gap-3">
                    <Key className="w-4 h-4 text-surface-400" />
                    <div>
                      <p className="font-medium text-surface-700">{k.label}</p>
                      <p className="text-xs text-surface-400">Created {new Date(k.created_at).toLocaleDateString()}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleRevoke(k.id)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded-lg"
                    title="Revoke key"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="mt-4">
          <button onClick={logout} className="text-sm text-surface-500 hover:text-red-600">
            {zh ? '退出登录' : 'Sign out'}
          </button>
        </div>
      </div>
    </div>
  )
}
