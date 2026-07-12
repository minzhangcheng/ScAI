/**
 * Read-only presentation of unsupported V1 snapshots.
 * Actions: view metadata, export, delete — never restore.
 */

export interface LegacySnapshotItem {
  id: string
  name: string
  phase?: string
  format_version?: number | null
  restorable?: boolean
  unsupported?: boolean
}

export interface LegacySnapshotInventoryProps {
  snapshots: LegacySnapshotItem[]
  warningVersion?: string
  onViewMetadata?: (id: string) => void
  onExport?: (id: string) => void
  onDelete?: (id: string) => void
  /** Intentionally unsupported — restore is never offered after V1 cutover */
  onRestore?: never
}

export function isLegacyRestorable(_item: LegacySnapshotItem): false {
  return false
}

export function legacyActionsFor(_item: LegacySnapshotItem): ReadonlyArray<'metadata' | 'export' | 'delete'> {
  return ['metadata', 'export', 'delete'] as const
}

export function LegacySnapshotInventory({
  snapshots,
  warningVersion,
  onViewMetadata,
  onExport,
  onDelete,
}: LegacySnapshotInventoryProps) {
  if (!snapshots.length) {
    return (
      <div data-testid="legacy-inventory-empty" className="text-sm text-slate-500">
        No legacy snapshots
      </div>
    )
  }

  return (
    <div data-testid="legacy-inventory" className="space-y-2">
      {warningVersion && (
        <p data-testid="legacy-warning" className="text-xs text-amber-700 bg-amber-50 rounded-lg px-3 py-2">
          Legacy V1 snapshots are unsupported for restore since {warningVersion}. Metadata, export, and
          delete remain available.
        </p>
      )}
      <ul className="space-y-2">
        {snapshots.map((snap) => (
          <li
            key={snap.id}
            data-testid={`legacy-snap-${snap.id}`}
            className="border border-slate-200 rounded-xl p-3 bg-white"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="font-medium text-sm text-slate-900 truncate">{snap.name || snap.id}</div>
                <div className="mt-1 flex flex-wrap gap-2 text-xs text-slate-500">
                  <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-800">unsupported</span>
                  {snap.phase && <span>{snap.phase}</span>}
                  <span>not restorable</span>
                </div>
              </div>
              <div className="flex flex-wrap gap-1 shrink-0">
                <button
                  type="button"
                  data-testid={`legacy-metadata-${snap.id}`}
                  className="text-xs px-2 py-1 rounded-lg border border-slate-200 hover:bg-slate-50"
                  onClick={() => onViewMetadata?.(snap.id)}
                >
                  Metadata
                </button>
                <button
                  type="button"
                  data-testid={`legacy-export-${snap.id}`}
                  className="text-xs px-2 py-1 rounded-lg border border-slate-200 hover:bg-slate-50"
                  onClick={() => onExport?.(snap.id)}
                >
                  Export
                </button>
                <button
                  type="button"
                  data-testid={`legacy-delete-${snap.id}`}
                  className="text-xs px-2 py-1 rounded-lg border border-red-200 text-red-700 hover:bg-red-50"
                  onClick={() => onDelete?.(snap.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
