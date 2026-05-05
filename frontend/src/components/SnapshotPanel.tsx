import { useState, useEffect, useLayoutEffect, useCallback, useMemo, useRef } from 'react'
import { Camera, RotateCcw, Trash2, X } from 'lucide-react'
import { Language } from '../i18n/translations'
import { apiFetch } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'

const SNAP_ROW_GAP = 24
const SNAP_COL_GAP = 16
const SNAP_GUTTER_W = 32
const SNAP_CARD_W = 260
const SNAP_NODE_R = 5
const SNAP_COL_STRIDE = SNAP_GUTTER_W + SNAP_CARD_W + 2 * SNAP_COL_GAP

const gutterCenterX = (ci: number) => ci * SNAP_COL_STRIDE + SNAP_GUTTER_W / 2
const cardGridCol = (ci: number) => 2 * ci + 2

type ForkPath = { d: string; archived: boolean }
type Rail = { x: number; y1: number; y2: number; archived: boolean }
type GraphNode = { cx: number; cy: number; archived: boolean; latest: boolean }
type GraphLayout = { nodes: GraphNode[]; rails: Rail[]; forkPaths: ForkPath[] }

const nodeClass = (n: GraphNode) =>
  n.archived
    ? 'stroke-slate-300 fill-white'
    : n.latest
      ? 'stroke-accent-500 fill-accent-500'
      : 'stroke-accent-500 fill-white'

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

interface Column {
  snapshots: Snapshot[]
  isArchived: boolean
}

interface Edge {
  fromId: string
  toId: string
  archived: boolean
}

interface GraphData {
  columns: Column[]
  edges: Edge[]
  rowOf: Map<string, number>
  activeLineIds: Set<string>
  rowCount: number
  latestId: string | null
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
      if (data.success) {
        const list = data.data?.snapshots || []
        setSnapshots(list)
        useAppStore.getState().setSnapshotCount(list.length)
      }
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
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  }

  const phaseLabel = (phase: string) => {
    const labels: Record<string, string> = isZh
      ? { pre_annotation: '注释前', post_annotation: '注释后', pre_direction: '选方向前', post_direction: '选方向后', pre_round_wait: '轮次结束', manual: '手动' }
      : { pre_annotation: 'Pre-Annotation', post_annotation: 'Post-Annotation', pre_direction: 'Pre-Direction', post_direction: 'Post-Direction', pre_round_wait: 'Round Complete', manual: 'Manual' }
    return labels[phase] || phase
  }

  const graph = useMemo<GraphData>(() => {
    if (snapshots.length === 0) {
      return { columns: [], edges: [], rowOf: new Map(), activeLineIds: new Set(), rowCount: 0, latestId: null }
    }

    const sorted = [...snapshots].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    )
    const idSet = new Set(sorted.map(s => s.id))
    const byId = new Map(sorted.map(s => [s.id, s]))
    const rowOf = new Map<string, number>()

    const latest = sorted[sorted.length - 1]
    const activeLineIds = new Set<string>()
    {
      let cursor: Snapshot | undefined = latest
      while (cursor) {
        activeLineIds.add(cursor.id)
        cursor = cursor.parent_id && idSet.has(cursor.parent_id)
          ? byId.get(cursor.parent_id)
          : undefined
      }
    }

    const childrenMap = new Map<string | null, Snapshot[]>()
    for (const snap of sorted) {
      const parentKey = snap.parent_id && idSet.has(snap.parent_id) ? snap.parent_id : null
      if (!childrenMap.has(parentKey)) childrenMap.set(parentKey, [])
      childrenMap.get(parentKey)!.push(snap)
    }
    for (const [, kids] of childrenMap) {
      kids.sort((a, b) => {
        const aActive = activeLineIds.has(a.id) ? 0 : 1
        const bActive = activeLineIds.has(b.id) ? 0 : 1
        if (aActive !== bActive) return aActive - bActive
        return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      })
    }

    const columns: Column[] = []
    const edges: Edge[] = []
    const visited = new Set<string>()

    let maxRow = -1

    function walk(node: Snapshot, colIndex: number, parentRow: number | null) {
      if (visited.has(node.id)) return
      visited.add(node.id)
      const col = columns[colIndex]
      const colNextRow = col.snapshots.length === 0
        ? 0
        : (rowOf.get(col.snapshots[col.snapshots.length - 1].id)! + 1)
      const row = parentRow !== null ? Math.max(colNextRow, parentRow + 1) : colNextRow
      rowOf.set(node.id, row)
      if (row > maxRow) maxRow = row
      col.snapshots.push(node)

      const kids = childrenMap.get(node.id) || []
      kids.forEach((child, i) => {
        const archived = !activeLineIds.has(child.id)
        if (i === 0) {
          walk(child, colIndex, row)
        } else {
          const newCol = columns.length
          columns.push({ snapshots: [], isArchived: archived })
          edges.push({ fromId: node.id, toId: child.id, archived })
          walk(child, newCol, row)
        }
      })
    }

    const roots = childrenMap.get(null) || []
    roots.forEach((root) => {
      const colIdx = columns.length
      columns.push({ snapshots: [], isArchived: !activeLineIds.has(root.id) })
      walk(root, colIdx, null)
    })

    for (const snap of sorted) {
      if (!visited.has(snap.id)) {
        columns.push({ snapshots: [snap], isArchived: !activeLineIds.has(snap.id) })
        rowOf.set(snap.id, 0)
        if (maxRow < 0) maxRow = 0
        visited.add(snap.id)
      }
    }

    return { columns, edges, rowOf, activeLineIds, rowCount: maxRow + 1, latestId: latest.id }
  }, [snapshots])

  const cardRefsRef = useRef<Map<string, HTMLDivElement>>(new Map())
  const setCardRef = useCallback((id: string) => (el: HTMLDivElement | null) => {
    if (el) cardRefsRef.current.set(id, el)
    else cardRefsRef.current.delete(id)
  }, [])

  const graphContainerRef = useRef<HTMLDivElement | null>(null)

  const [layout, setLayout] = useState<GraphLayout>({ nodes: [], rails: [], forkPaths: [] })
  const prevSigRef = useRef<string>('')

  const colOf = useMemo(() => {
    const m = new Map<string, number>()
    graph.columns.forEach((col, ci) => {
      col.snapshots.forEach(s => m.set(s.id, ci))
    })
    return m
  }, [graph])

  useLayoutEffect(() => {
    const container = graphContainerRef.current
    if (!container) return

    const recompute = () => {
      const cTop = container.getBoundingClientRect().top

      const cardCenterY = (id: string): number | null => {
        const el = cardRefsRef.current.get(id)
        if (!el) return null
        const r = el.getBoundingClientRect()
        return r.top + r.height / 2 - cTop
      }
      const cardBottomY = (id: string): number | null => {
        const el = cardRefsRef.current.get(id)
        if (!el) return null
        return el.getBoundingClientRect().bottom - cTop
      }

      const newNodes: GraphNode[] = []
      const newRails: Rail[] = []
      const newForks: ForkPath[] = []

      for (const [id, ci] of colOf) {
        const cy = cardCenterY(id)
        if (cy == null) continue
        newNodes.push({
          cx: gutterCenterX(ci),
          cy,
          archived: !graph.activeLineIds.has(id),
          latest: id === graph.latestId,
        })
      }

      graph.columns.forEach((col, ci) => {
        if (col.snapshots.length < 2) return
        const y1 = cardCenterY(col.snapshots[0].id)
        const y2 = cardCenterY(col.snapshots[col.snapshots.length - 1].id)
        if (y1 == null || y2 == null) return
        newRails.push({ x: gutterCenterX(ci), y1, y2, archived: col.isArchived })
      })

      for (const e of graph.edges) {
        const fromCi = colOf.get(e.fromId)
        const toCi = colOf.get(e.toId)
        if (fromCi == null || toCi == null) continue
        const py = cardCenterY(e.fromId)
        const cy = cardCenterY(e.toId)
        const fBottom = cardBottomY(e.fromId)
        if (py == null || cy == null || fBottom == null) continue
        const px = gutterCenterX(fromCi)
        const cx = gutterCenterX(toCi)
        const turnY = fBottom + SNAP_ROW_GAP / 2
        newForks.push({
          d: `M ${px} ${py} L ${px} ${turnY} L ${cx} ${turnY} L ${cx} ${cy}`,
          archived: e.archived,
        })
      }

      const next: GraphLayout = { nodes: newNodes, rails: newRails, forkPaths: newForks }
      const sig = JSON.stringify(next)
      if (sig === prevSigRef.current) return
      prevSigRef.current = sig
      setLayout(next)
    }

    recompute()
    const ro = new ResizeObserver(recompute)
    ro.observe(container)
    cardRefsRef.current.forEach(el => ro.observe(el))
    window.addEventListener('resize', recompute)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', recompute)
    }
  }, [graph, colOf])

  const renderCard = (snap: Snapshot, isArchived: boolean) => {
    return (
      <div
        ref={setCardRef(snap.id)}
        className={`border border-slate-200 rounded-xl p-3 hover:border-accent-300 transition-colors bg-white ${isArchived ? 'opacity-60' : ''}`}
        style={{ width: SNAP_CARD_W }}
      >
        <div className="flex items-center justify-between">
          <div className="min-w-0 flex-1">
            <div className={`font-medium text-sm truncate ${isArchived ? 'text-slate-500' : 'text-slate-900'}`}>{snap.name}</div>
            <div className="flex items-center gap-2 mt-1 text-xs text-slate-500 flex-wrap">
              <span className="px-2 py-0.5 bg-slate-100 rounded-full">{phaseLabel(snap.phase)}</span>
              {snap.branch_label && snap.branch_label !== 'main' && (
                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
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
                  <div className="p-2 bg-yellow-50 rounded text-xs">
                    <p className="font-medium text-yellow-800">
                      {isZh ? '回滚将删除以下内容：' : 'Restoring will delete:'}
                    </p>
                    <ul className="list-disc list-inside text-yellow-700">
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

        {/* Snapshot graph */}
        <div className="flex-1 overflow-auto p-4 relative">
          {loading ? (
            <div className="text-center text-slate-400 py-8">{isZh ? '加载中...' : 'Loading...'}</div>
          ) : graph.columns.length === 0 ? (
            <div className="text-center text-slate-400 py-8">
              {isZh ? '暂无快照（快照会在分析关键节点自动创建）' : 'No snapshots (snapshots are created automatically at key analysis points)'}
            </div>
          ) : (
            <div
              ref={graphContainerRef}
              className="relative"
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${graph.columns.length}, ${SNAP_GUTTER_W}px ${SNAP_CARD_W}px)`,
                columnGap: SNAP_COL_GAP,
                rowGap: SNAP_ROW_GAP,
                gridTemplateRows: `repeat(${graph.rowCount}, auto)`,
                width: 'max-content',
              }}
            >
              <svg
                className="absolute inset-0 pointer-events-none"
                style={{ width: '100%', height: '100%', overflow: 'visible' }}
              >
                {layout.rails.map((r, i) => (
                  <line
                    key={`rail-${i}`}
                    x1={r.x}
                    y1={r.y1}
                    x2={r.x}
                    y2={r.y2}
                    strokeWidth={2}
                    className={r.archived ? 'stroke-slate-300' : 'stroke-accent-500'}
                  />
                ))}
                {layout.forkPaths.map((p, i) => (
                  <path
                    key={`fork-${i}`}
                    d={p.d}
                    fill="none"
                    strokeWidth={2}
                    className={p.archived ? 'stroke-slate-300' : 'stroke-accent-500'}
                  />
                ))}
                {layout.nodes.map((n, i) => (
                  <circle
                    key={`node-${i}`}
                    cx={n.cx}
                    cy={n.cy}
                    r={SNAP_NODE_R}
                    strokeWidth={2}
                    className={nodeClass(n)}
                  />
                ))}
              </svg>
              {graph.columns.flatMap((col, ci) =>
                col.snapshots.map(snap => (
                  <div
                    key={snap.id}
                    style={{ gridColumn: cardGridCol(ci), gridRow: (graph.rowOf.get(snap.id) ?? 0) + 1 }}
                  >
                    {renderCard(snap, col.isArchived)}
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
