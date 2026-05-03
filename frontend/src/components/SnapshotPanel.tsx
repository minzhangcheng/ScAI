import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Camera, RotateCcw, Trash2, X, GitBranch } from 'lucide-react'
import { Language } from '../i18n/translations'
import { apiFetch } from '../hooks/useApi'

interface Snapshot {
  id: string
  name: string
  description: string
  phase: string
  round_number: number
  timestamp: string
  size_bytes: number
  parent_id: string | null
  branch_label: string | null
  notebook_file: string
  workspace_manifest: Record<string, string[]>
}

interface TreeNode {
  snapshot: Snapshot
  depth: number
  isBranch: boolean
}

interface SnapshotPanelProps {
  sessionId: string
  language: Language
  onClose: () => void
  onRestore: () => void
}

export function SnapshotPanel({ sessionId, language, onClose, onRestore }: SnapshotPanelProps) {
  const isZh = language === 'zh'
  const [snapshots, setSnapshots] = useState<Snapshot[]>([])
  const [loading, setLoading] = useState(true)
  const [restoring, setRestoring] = useState<string | null>(null)
  const [confirmRestore, setConfirmRestore] = useState<string | null>(null)
  const [rollbackPreview, setRollbackPreview] = useState<{
    snapshotId: string
    filesCount: number
    cellsRemoved: number
    descendantSnapshots: string[]
  } | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const fetchSnapshots = useCallback(async () => {
    try {
      const data = await apiFetch<any>(`/session/${sessionId}/snapshots`)
      if (data.success) setSnapshots(data.data?.snapshots || [])
    } catch (e) {
      console.error('Failed to fetch snapshots:', e)
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  useEffect(() => { fetchSnapshots() }, [fetchSnapshots])

  // Abort in-flight rollback preview request on unmount
  useEffect(() => {
    return () => { abortControllerRef.current?.abort() }
  }, [])

  const handleRestore = async (snapshotId: string) => {
    setRestoring(snapshotId)
    try {
      await apiFetch(`/session/${sessionId}/snapshots/${snapshotId}/restore`, {
        method: 'POST',
      })
      onRestore()
    } finally {
      setRestoring(null)
      setConfirmRestore(null)
      setRollbackPreview(null)
    }
  }

  const fetchRollbackPreview = async (snapshotId: string) => {
    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller
    try {
      const data = await apiFetch<any>(`/session/${sessionId}/snapshots/${snapshotId}/rollback-preview`, { signal: controller.signal })
      if (data.success && !controller.signal.aborted) {
        setRollbackPreview({
          snapshotId,
          filesCount: data.data.files_to_delete_count,
          cellsRemoved: data.data.notebook_cells_removed,
          descendantSnapshots: data.data.descendant_snapshots,
        })
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      console.error('Failed to fetch rollback preview', e)
    }
  }

  const handleRestoreClick = (snapshotId: string) => {
    setConfirmRestore(snapshotId)
    fetchRollbackPreview(snapshotId)
  }

  const handleDelete = async (snapshotId: string) => {
    await apiFetch(`/session/${sessionId}/snapshots/${snapshotId}`, {
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
      ? { pre_annotation: '注释前', post_annotation: '注释后', pre_direction: '选方向前', post_direction: '选方向后', pre_round_wait: '轮次结束', manual: '手动' }
      : { pre_annotation: 'Pre-Annotation', post_annotation: 'Post-Annotation', pre_direction: 'Pre-Direction', post_direction: 'Post-Direction', pre_round_wait: 'Round Complete', manual: 'Manual' }
    return labels[phase] || phase
  }

  // Build tree structure from flat snapshot list using parent_id relationships
  const treeNodes = useMemo(() => {
    if (snapshots.length === 0) return []

    const sorted = [...snapshots].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    )

    const idSet = new Set(sorted.map(s => s.id))

    // Map each parent to its children (ordered by timestamp via sorted)
    const childrenMap = new Map<string | null, Snapshot[]>()
    for (const snap of sorted) {
      const parentKey = snap.parent_id && idSet.has(snap.parent_id) ? snap.parent_id : null
      if (!childrenMap.has(parentKey)) childrenMap.set(parentKey, [])
      childrenMap.get(parentKey)!.push(snap)
    }

    // DFS to flatten the tree in parent-first order
    const result: TreeNode[] = []
    const visited = new Set<string>()

    function dfs(parentId: string | null) {
      for (const snap of childrenMap.get(parentId) || []) {
        if (visited.has(snap.id)) continue
        visited.add(snap.id)
        result.push({ snapshot: snap, depth: 0, isBranch: false })
        dfs(snap.id)
      }
    }
    dfs(null)

    // Add orphans (e.g., legacy snapshots without parent_id)
    for (const snap of sorted) {
      if (!visited.has(snap.id)) {
        result.push({ snapshot: snap, depth: 0, isBranch: false })
      }
    }

    // Compute depth and detect branches: when a parent has multiple children,
    // the second+ child is a branch (created after a restore)
    const parentChildCount = new Map<string | null, number>()
    const depthMap = new Map<string, number>()

    for (const node of result) {
      const parentKey = node.snapshot.parent_id && idSet.has(node.snapshot.parent_id)
        ? node.snapshot.parent_id : null
      const count = (parentChildCount.get(parentKey) || 0) + 1
      parentChildCount.set(parentKey, count)

      let depth = 0
      if (parentKey && depthMap.has(parentKey)) {
        depth = depthMap.get(parentKey)!
        if (count > 1) {
          depth += 1
          node.isBranch = true
        }
      }
      node.depth = depth
      depthMap.set(node.snapshot.id, depth)
    }

    return result
  }, [snapshots])

  const renderSnapshotCard = (node: TreeNode) => {
    const snap = node.snapshot
    const indent = node.depth * 20

    return (
      <div key={snap.id} style={{ marginLeft: indent }} className="relative">
        {/* Branch indicator */}
        {node.isBranch && (
          <div className="flex items-center gap-1 mb-1 text-xs text-amber-600">
            <GitBranch className="w-3 h-3" />
            <span>{isZh ? '从此分支' : 'Branched here'}</span>
          </div>
        )}
        {/* Connector line for indented items */}
        {node.depth > 0 && (
          <div
            className="absolute left-0 top-0 bottom-0 border-l-2 border-accent-200"
            style={{ left: -10 }}
          />
        )}
        <div className="border border-slate-200 rounded-xl p-3 hover:border-accent-300 transition-colors bg-white">
          <div className="flex items-center justify-between">
            <div className="min-w-0 flex-1">
              <div className="font-medium text-sm text-slate-900 truncate">{snap.name}</div>
              <div className="flex items-center gap-3 mt-1 text-xs text-slate-500 flex-wrap">
                <span className="px-2 py-0.5 bg-slate-100 rounded-full">{phaseLabel(snap.phase)}</span>
                {snap.branch_label && snap.branch_label !== 'main' && (
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300 ml-1">
                    {snap.branch_label}
                  </span>
                )}
                <span>Round {snap.round_number}</span>
                <span>{new Date(snap.timestamp).toLocaleString()}</span>
                <span>{formatSize(snap.size_bytes)}</span>
              </div>
            </div>
            <div className="flex items-center gap-1 ml-2 flex-shrink-0">
              {confirmRestore === snap.id ? (
                <div className="flex flex-col gap-1">
                  {rollbackPreview?.snapshotId === snap.id && (
                    <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded text-xs">
                      <p className="font-medium text-yellow-800 dark:text-yellow-200">
                        {isZh ? '回滚将删除以下内容：' : 'Restoring will delete:'}
                      </p>
                      <ul className="list-disc list-inside text-yellow-700 dark:text-yellow-300">
                        {rollbackPreview.filesCount > 0 && (
                          <li>{rollbackPreview.filesCount} {isZh ? '个工作区文件' : ' workspace file(s)'}</li>
                        )}
                        {rollbackPreview.cellsRemoved > 0 && (
                          <li>{rollbackPreview.cellsRemoved} {isZh ? '个notebook单元格' : ' notebook cell(s)'}</li>
                        )}
                        {rollbackPreview.descendantSnapshots.length > 0 && (
                          <li>{rollbackPreview.descendantSnapshots.length} {isZh ? '个后续快照' : ' descendant snapshot(s)'}</li>
                        )}
                      </ul>
                    </div>
                  )}
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
                      onClick={() => { setConfirmRestore(null); setRollbackPreview(null) }}
                      className="px-2 py-1 text-xs text-slate-500 hover:bg-slate-100 rounded-lg"
                    >
                      {isZh ? '取消' : 'No'}
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <button
                    onClick={() => handleRestoreClick(snap.id)}
                    className="p-1.5 text-accent-600 hover:bg-accent-50 rounded-lg"
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
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-xl mx-4 max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-accent-600" />
            <h2 className="text-lg font-semibold text-slate-900">
              {isZh ? '分析快照' : 'Analysis Snapshots'}
            </h2>
            {snapshots.length > 0 && (
              <span className="text-xs text-slate-400 ml-1">({snapshots.length})</span>
            )}
          </div>
          <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-lg">
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Snapshot tree */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {loading ? (
            <div className="text-center text-slate-400 py-8">{isZh ? '加载中...' : 'Loading...'}</div>
          ) : treeNodes.length === 0 ? (
            <div className="text-center text-slate-400 py-8">
              {isZh ? '暂无快照（快照会在分析关键节点自动创建）' : 'No snapshots (snapshots are created automatically at key analysis points)'}
            </div>
          ) : (
            treeNodes.map(node => renderSnapshotCard(node))
          )}
        </div>
      </div>
    </div>
  )
}
