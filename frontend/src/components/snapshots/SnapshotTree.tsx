import type { SnapshotRecordV2 } from '../../types/snapshotV2'

function capsuleLabel(kind: string): string {
  const labels: Record<string, string> = {
    idle: 'Idle',
    graph_node: 'Graph Node',
    terminal: 'Terminal',
    semantic_gate: 'Semantic Gate',
  }
  return labels[kind] || kind
}

interface SnapshotTreeProps {
  snapshots: SnapshotRecordV2[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export function SnapshotTree({ snapshots, selectedId, onSelect }: SnapshotTreeProps) {
  const sorted = [...snapshots].sort(
    (a, b) => new Date(a.manifest.captured_at).getTime() - new Date(b.manifest.captured_at).getTime(),
  )

  return (
    <div className="space-y-2">
      {sorted.map((record) => {
        const m = record.manifest
        const isSelected = record.snapshot_id === selectedId
        const workspaceCount = m.workspace ? m.workspace.length : 0

        return (
          <div
            key={record.snapshot_id}
            onClick={() => onSelect(record.snapshot_id)}
            className={`border rounded-xl p-3 cursor-pointer transition-colors ${
              isSelected
                ? 'border-accent-500 bg-accent-50'
                : 'border-slate-200 hover:border-accent-300 bg-white'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="min-w-0 flex-1">
                <div className="font-medium text-sm text-slate-900 truncate">{record.snapshot_id}</div>
                <div className="flex items-center gap-2 mt-1 text-xs text-slate-500 flex-wrap">
                  <span className="px-2 py-0.5 bg-slate-100 rounded-full">{capsuleLabel(m.capsule.kind)}</span>
                  <span>{new Date(m.captured_at).toLocaleString()}</span>
                  <span>
                    {workspaceCount} {workspaceCount === 1 ? 'file' : 'files'}
                  </span>
                  {m.capsule.kind === 'graph_node' && (
                    <span className="text-accent-600">Next: {m.capsule.next_node}</span>
                  )}
                  {m.runtime_dependencies?.python && (
                    <span>{m.runtime_dependencies.python.version}</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
