import { useState, useEffect, useCallback } from 'react'
import { Camera, RotateCcw, Trash2, X } from 'lucide-react'
import { Language } from '../i18n/translations'

interface Snapshot {
  id: string
  name: string
  description: string
  phase: string
  round_number: number
  timestamp: string
  size_bytes: number
}

interface SnapshotPanelProps {
  sessionId: string
  apiBase: string
  language: Language
  onClose: () => void
  onRestore: () => void
}

export function SnapshotPanel({ sessionId, apiBase, language, onClose, onRestore }: SnapshotPanelProps) {
  const isZh = language === 'zh'
  const [snapshots, setSnapshots] = useState<Snapshot[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [restoring, setRestoring] = useState<string | null>(null)
  const [confirmRestore, setConfirmRestore] = useState<string | null>(null)
  const [snapshotName, setSnapshotName] = useState('')

  const fetchSnapshots = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/session/${sessionId}/snapshots`)
      const data = await res.json()
      if (data.success) setSnapshots(data.data?.snapshots || [])
    } catch (e) {
      console.error('Failed to fetch snapshots:', e)
    } finally {
      setLoading(false)
    }
  }, [sessionId, apiBase])

  useEffect(() => { fetchSnapshots() }, [fetchSnapshots])

  const handleCreate = async () => {
    setCreating(true)
    try {
      const res = await fetch(`${apiBase}/session/${sessionId}/snapshots`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: snapshotName || (isZh ? '手动快照' : 'Manual snapshot') }),
      })
      if (res.ok) {
        setSnapshotName('')
        await fetchSnapshots()
      }
    } finally {
      setCreating(false)
    }
  }

  const handleRestore = async (snapshotId: string) => {
    setRestoring(snapshotId)
    try {
      const res = await fetch(`${apiBase}/session/${sessionId}/snapshots/${snapshotId}/restore`, {
        method: 'POST',
      })
      if (res.ok) {
        onRestore()
      }
    } finally {
      setRestoring(null)
      setConfirmRestore(null)
    }
  }

  const handleDelete = async (snapshotId: string) => {
    await fetch(`${apiBase}/session/${sessionId}/snapshots/${snapshotId}`, {
      method: 'DELETE',
    })
    await fetchSnapshots()
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const phaseLabel = (phase: string) => {
    const labels: Record<string, string> = isZh
      ? { pre_annotation: '注释前', post_annotation: '注释后', pre_direction: '选方向前', post_direction: '选方向后', manual: '手动' }
      : { pre_annotation: 'Pre-Annotation', post_annotation: 'Post-Annotation', pre_direction: 'Pre-Direction', post_direction: 'Post-Direction', manual: 'Manual' }
    return labels[phase] || phase
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-xl mx-4 max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-indigo-600" />
            <h2 className="text-lg font-semibold text-slate-900">
              {isZh ? '分析快照' : 'Analysis Snapshots'}
            </h2>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-lg">
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Create snapshot */}
        <div className="px-6 py-3 border-b bg-slate-50 flex items-center gap-2">
          <input
            type="text"
            value={snapshotName}
            onChange={e => setSnapshotName(e.target.value)}
            placeholder={isZh ? '快照名称（可选）' : 'Snapshot name (optional)'}
            className="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            onKeyDown={e => { if (e.key === 'Enter') handleCreate() }}
          />
          <button
            onClick={handleCreate}
            disabled={creating}
            className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-1.5 whitespace-nowrap"
          >
            <Camera className="w-4 h-4" />
            {creating ? (isZh ? '创建中...' : 'Creating...') : (isZh ? '创建快照' : 'Create')}
          </button>
        </div>

        {/* Snapshot list */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {loading ? (
            <div className="text-center text-slate-400 py-8">{isZh ? '加载中...' : 'Loading...'}</div>
          ) : snapshots.length === 0 ? (
            <div className="text-center text-slate-400 py-8">{isZh ? '暂无快照' : 'No snapshots'}</div>
          ) : (
            [...snapshots].reverse().map(snap => (
              <div key={snap.id} className="border border-slate-200 rounded-xl p-3 hover:border-indigo-300 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="font-medium text-sm text-slate-900 truncate">{snap.name}</div>
                    <div className="flex items-center gap-3 mt-1 text-xs text-slate-500 flex-wrap">
                      <span className="px-2 py-0.5 bg-slate-100 rounded-full">{phaseLabel(snap.phase)}</span>
                      <span>Round {snap.round_number}</span>
                      <span>{new Date(snap.timestamp).toLocaleString()}</span>
                      <span>{formatSize(snap.size_bytes)}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 ml-2 flex-shrink-0">
                    {confirmRestore === snap.id ? (
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-amber-600 mr-1">
                          {isZh ? '确认？' : 'Sure?'}
                        </span>
                        <button
                          onClick={() => handleRestore(snap.id)}
                          disabled={restoring === snap.id}
                          className="px-2 py-1 text-xs bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-50"
                        >
                          {restoring === snap.id ? '...' : (isZh ? '确认' : 'Yes')}
                        </button>
                        <button
                          onClick={() => setConfirmRestore(null)}
                          className="px-2 py-1 text-xs text-slate-500 hover:bg-slate-100 rounded-lg"
                        >
                          {isZh ? '取消' : 'No'}
                        </button>
                      </div>
                    ) : (
                      <>
                        <button
                          onClick={() => setConfirmRestore(snap.id)}
                          className="p-1.5 text-indigo-600 hover:bg-indigo-50 rounded-lg"
                          title={isZh ? '恢复' : 'Restore'}
                        >
                          <RotateCcw className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(snap.id)}
                          className="p-1.5 text-red-500 hover:bg-red-50 rounded-lg"
                          title={isZh ? '删除' : 'Delete'}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
