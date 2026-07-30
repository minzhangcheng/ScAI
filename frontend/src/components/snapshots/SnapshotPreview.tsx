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
  isZh?: boolean
}

export function SnapshotPreview({ record, isZh = false }: SnapshotPreviewProps) {
  const m = record.manifest

  // For a corrupt entry the manifest is a placeholder the normalizer invented
  // so this component cannot crash on a missing field — rendering it would
  // report a captured time, branch, generation and root hash that were never
  // read off disk. The verifier's diagnostics are the only real information
  // available, so show those and nothing else.
  if (record.failure) {
    return (
      <div className="space-y-3">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
          Snapshot {record.snapshot_id}
        </div>
        <div className="bg-red-50 border border-red-200 rounded-xl p-3 space-y-2">
          <div className="text-xs font-semibold text-red-700 uppercase tracking-wide">
            {isZh ? '快照已损坏' : 'Corrupt Snapshot'}
          </div>
          <div className="text-xs text-red-700">
            <span className="font-medium">{isZh ? '错误码：' : 'Code: '}</span>
            <code className="text-[10px]">{record.failure.code}</code>
          </div>
          {record.failure.message && (
            <div className="text-xs text-red-700 break-words">{record.failure.message}</div>
          )}
          <div className="text-xs text-red-600">
            {isZh
              ? '完整性校验未通过，该快照无法恢复；可将其删除后使用其他快照。'
              : 'Integrity verification failed. This snapshot cannot be restored; delete it and use another one.'}
          </div>
        </div>
      </div>
    )
  }

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

      {m.runtime && (
        <div className="bg-slate-50 rounded-xl p-3">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Runtime</div>
          <div className="space-y-1 text-xs text-slate-600">
            <div>
              <span className="font-medium text-slate-800">Status: </span>
              {m.runtime.status}
            </div>
            {Object.entries(m.runtime.dependencies?.required ?? {}).map(([name, dep]) => (
              <div key={name}>
                <span className="font-medium text-slate-800">{name}: </span>
                {dep.serializer}
                {dep.type_name ? ` (${dep.type_name})` : ''}
              </div>
            ))}
            {(m.runtime.dependencies?.incidental_skips ?? []).map((skip) => (
              <div key={`skip-${skip.name}`} className="text-amber-700">
                <span className="font-medium">{skip.name}: </span>
                skipped{skip.reason ? ` — ${skip.reason}` : ''}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
