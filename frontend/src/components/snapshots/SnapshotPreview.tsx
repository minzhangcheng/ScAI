import { formatBytes } from '../../utils/snapshotV2'
import type { SnapshotRecordV2, ContinuationCapsule } from '../../types/snapshotV2'

function capsuleDetail(capsule: ContinuationCapsule): string {
  if (capsule.kind === 'graph_node') return `Next: ${capsule.next_node} (${capsule.planning_stage})`
  if (capsule.kind === 'semantic_gate') return `Gate: ${capsule.gate}${capsule.wait_reason ? ` — ${capsule.wait_reason}` : ''}`
  if (capsule.kind === 'terminal') return `Status: ${capsule.status}`
  return ''
}

interface SnapshotPreviewProps {
  record: SnapshotRecordV2
}

export function SnapshotPreview({ record }: SnapshotPreviewProps) {
  const m = record.manifest

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
        Snapshot {record.snapshot_id}
      </div>

      <div className="bg-slate-50 rounded-xl p-3 space-y-2">
        <div className="text-xs text-slate-600">
          <span className="font-medium text-slate-800">Captured: </span>
          {new Date(m.captured_at).toLocaleString()}
        </div>
        <div className="text-xs text-slate-600">
          <span className="font-medium text-slate-800">Branch: </span>
          {m.branch}
        </div>
        <div className="text-xs text-slate-600">
          <span className="font-medium text-slate-800">Generation: </span>
          {m.generation}
        </div>
        <div className="text-xs text-slate-600">
          <span className="font-medium text-slate-800">Root Hash: </span>
          <code className="text-[10px]">{m.root_hash}</code>
        </div>
      </div>

      <div className="bg-slate-50 rounded-xl p-3">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Capsule</div>
        <div className="text-xs text-slate-700">
          <div>
            <span className="font-medium text-slate-800">Kind: </span>
            {m.capsule.kind}
          </div>
          {capsuleDetail(m.capsule) && (
            <div className="mt-1 text-accent-700">{capsuleDetail(m.capsule)}</div>
          )}
          {m.capsule.kind === 'graph_node' && (
            <div className="mt-1 text-xs text-slate-500">Graph v{m.capsule.graph_version}</div>
          )}
        </div>
      </div>

      {m.workspace && m.workspace.length > 0 && (
        <div className="bg-slate-50 rounded-xl p-3">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            Workspace ({m.workspace.length})
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {m.workspace.map((entry) => (
              <div key={entry.path} className="flex items-center justify-between text-xs text-slate-600">
                <div className="flex items-center gap-1 min-w-0">
                  <span className="truncate font-mono text-[10px]">{entry.path}</span>
                  <span className="text-slate-400">({formatBytes(entry.size)})</span>
                </div>
                <code className="text-[10px] text-slate-400 ml-2 shrink-0">{entry.object_hash}</code>
              </div>
            ))}
          </div>
        </div>
      )}

      {m.objects && Object.keys(m.objects).length > 0 && (
        <div className="bg-slate-50 rounded-xl p-3">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            Objects ({Object.keys(m.objects).length})
          </div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {Object.entries(m.objects).map(([hash, obj]) => (
              <div key={hash} className="flex items-center justify-between text-xs text-slate-600">
                <code className="text-[10px] truncate">{hash}</code>
                <span className="shrink-0 ml-2">{formatBytes(obj.size)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {m.runtime_dependencies && (
        <div className="bg-slate-50 rounded-xl p-3">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Runtime Dependencies</div>
          <div className="space-y-1 text-xs text-slate-600">
            <div>
              <span className="font-medium text-slate-800">{m.runtime_dependencies.python.name}: </span>
              {m.runtime_dependencies.python.version}
            </div>
            {Object.entries(m.runtime_dependencies.additional).map(([name, dep]) => (
              <div key={name}>
                <span className="font-medium text-slate-800">{dep.name}: </span>
                {dep.version}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
