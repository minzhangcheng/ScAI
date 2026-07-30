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

/**
 * A record with no usable manifest sorts last rather than nowhere.
 *
 * `normalizeSnapshotRecordV2` gives a corrupt entry `captured_at: ''`, whose
 * timestamp is NaN; a NaN comparator result is treated as "not greater than",
 * so the row silently kept whatever position the sort happened to visit it in.
 */
function orderKey(record: SnapshotRecordV2): number {
  if (record.failure) return Number.MAX_SAFE_INTEGER
  const captured = new Date(record.manifest.captured_at).getTime()
  return Number.isNaN(captured) ? Number.MAX_SAFE_INTEGER : captured
}

interface SnapshotTreeProps {
  snapshots: SnapshotRecordV2[]
  selectedId: string | null
  onSelect: (id: string) => void
  isZh?: boolean
}

export function SnapshotTree({ snapshots, selectedId, onSelect, isZh = false }: SnapshotTreeProps) {
  const sorted = [...snapshots].sort((a, b) => orderKey(a) - orderKey(b))

  return (
    <div className="space-y-2">
      {sorted.map((record) => {
        const m = record.manifest
        const isSelected = record.snapshot_id === selectedId
        const workspaceCount = m.workspace ? m.workspace.length : 0
        // `list_snapshots` keeps a manifest that fails HMAC/schema/root-hash
        // verification in the listing on purpose, and the normalizer parks its
        // diagnostics on `record.failure` while synthesising a placeholder
        // manifest so this render cannot crash on a missing field. Every field
        // below is therefore a fabrication for exactly those entries — "Idle",
        // "Invalid Date", "0 files" — i.e. a corrupt snapshot was indis-
        // tinguishable from a healthy idle one. Show the diagnostics instead.
        const failure = record.failure

        return (
          <div
            key={record.snapshot_id}
            onClick={() => onSelect(record.snapshot_id)}
            className={`border rounded-xl p-3 cursor-pointer transition-colors ${
              failure
                ? (isSelected ? 'border-red-500 bg-red-100' : 'border-red-200 hover:border-red-300 bg-red-50')
                : isSelected
                  ? 'border-accent-500 bg-accent-50'
                  : 'border-slate-200 hover:border-accent-300 bg-white'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="min-w-0 flex-1">
                <div className="font-medium text-sm text-slate-900 truncate">{record.snapshot_id}</div>
                {failure ? (
                  <div className="flex items-center gap-2 mt-1 text-xs text-red-700 flex-wrap">
                    <span className="px-2 py-0.5 bg-red-100 rounded-full">
                      {isZh ? '已损坏' : 'Corrupt'}
                    </span>
                    <span className="font-mono">{failure.code}</span>
                    {failure.message && <span className="truncate">{failure.message}</span>}
                  </div>
                ) : (
                <div className="flex items-center gap-2 mt-1 text-xs text-slate-500 flex-wrap">
                  <span className="px-2 py-0.5 bg-slate-100 rounded-full">{capsuleLabel(m.capsule.kind)}</span>
                  <span>{new Date(m.captured_at).toLocaleString()}</span>
                  <span>
                    {workspaceCount} {workspaceCount === 1 ? 'file' : 'files'}
                  </span>
                  {m.capsule.kind === 'graph_node' && (
                    <span className="text-accent-600">Next: {m.capsule.next_node}</span>
                  )}
                  {/* The runtime status is what decides whether restoring this
                      snapshot bounces the user into the reset/keep dialog, so
                      it belongs in the row rather than behind a click. */}
                  {m.runtime?.status && (
                    <span
                      className={
                        m.runtime.status === 'complete'
                          ? 'text-emerald-600'
                          : 'text-amber-600'
                      }
                    >
                      runtime: {m.runtime.status}
                    </span>
                  )}
                </div>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
