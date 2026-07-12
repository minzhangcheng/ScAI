import type { SnapshotStoreStatus } from '../../types/snapshotV2'

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

interface StorageSummaryProps {
  status: SnapshotStoreStatus
}

export function StorageSummary({ status }: StorageSummaryProps) {
  const savingsPct = status.logical_bytes > 0
    ? Math.round((1 - status.physical_bytes / status.logical_bytes) * 100)
    : 0

  return (
    <div className="bg-slate-50 rounded-xl p-4 space-y-3">
      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Store Status</div>
      <div className="flex items-center gap-4 text-xs text-slate-600">
        <div>
          <span className="font-medium text-slate-800">Logical: </span>
          {formatSize(status.logical_bytes)}
        </div>
        <div>
          <span className="font-medium text-slate-800">Physical: </span>
          {formatSize(status.physical_bytes)}
        </div>
        <div>
          <span className="font-medium text-slate-800">Saved: </span>
          {formatSize(status.dedup_savings)}
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex-1 bg-slate-200 rounded-full h-2 overflow-hidden">
          <div
            role="progressbar"
            className="bg-accent-500 h-2 rounded-full transition-all"
            style={{ width: `${Math.min(savingsPct, 100)}%` }}
          />
        </div>
        <span className="text-xs text-slate-500">{savingsPct}% dedup</span>
      </div>
      <div className="flex items-center gap-4 text-xs text-slate-600">
        <div>
          <span className="font-medium text-slate-800">Objects: </span>
          {status.object_count}
        </div>
        <div>
          <span className="font-medium text-slate-800">Manifests: </span>
          {status.manifest_count}
        </div>
      </div>
    </div>
  )
}
