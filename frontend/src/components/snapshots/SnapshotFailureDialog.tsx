import type { SnapshotFailureInfo } from '../../types/snapshotV2'
import { useTranslation } from '../../i18n/translations'

export interface SnapshotFailureRetryOptions {
  /** Retry re-enters capture only; never regenerates completed reports. */
  regenerateReports: false
}

interface SnapshotFailureDialogProps {
  failure: SnapshotFailureInfo
  isZh?: boolean
  onRetry: (options: SnapshotFailureRetryOptions) => void
  onProceed: () => void
  onSkip: () => void
  onManage: () => void
  onTerminate: () => void
  onClose?: () => void
}

function extractNamedVariables(details?: Record<string, unknown>): string[] {
  if (!details) return []
  const names = new Set<string>()
  const raw = details.required_variables
  if (Array.isArray(raw)) {
    for (const item of raw) {
      if (typeof item === 'string' && item.trim()) names.add(item.trim())
    }
  }
  if (typeof details.name === 'string' && details.name.trim()) {
    names.add(details.name.trim())
  }
  return Array.from(names)
}

function phaseOf(details?: Record<string, unknown>): string | null {
  if (!details) return null
  if (typeof details.phase === 'string') return details.phase
  return null
}

function formatBytes(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '—'
  if (value < 1024) return `${Math.round(value)} B`
  const units = ['KiB', 'MiB', 'GiB', 'TiB']
  let amount = value / 1024
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[unit]}`
}

export function SnapshotFailureDialog({
  failure,
  isZh = false,
  onRetry,
  onProceed,
  onSkip,
  onManage,
  onTerminate,
  onClose,
}: SnapshotFailureDialogProps) {
  const { t } = useTranslation(isZh ? 'zh' : 'en')
  const record = failure.failure
  const actions = new Set(failure.allowed_actions || [])
  const variables = extractNamedVariables(record.details)
  const phase = phaseOf(record.details)
  const isPreRestore = phase === 'pre_restore' || /pre_restore/i.test(record.code || '')
  const isPostReport =
    phase === 'post_report' || /post_report/i.test(record.code || '')
  // The memory preflight refused before allocating anything. There is no Retry
  // on this gate: the check is deterministic, so it would refuse again.
  const isMemoryRefusal = record.code === 'snapshot_memory_insufficient'
  const isRuntimePersistence = (record.code || '').startsWith('runtime_persistence_')
  const runtimeDetails = record.details || {}
  const budgetViolations = Array.isArray(runtimeDetails.budget_violations)
    ? runtimeDetails.budget_violations.filter((item): item is Record<string, unknown> =>
        Boolean(item) && typeof item === 'object',
      )
    : []
  const topOffenders = Array.isArray(runtimeDetails.top_offenders)
    ? runtimeDetails.top_offenders.filter((item): item is Record<string, unknown> =>
        Boolean(item) && typeof item === 'object',
      )
    : []

  // The backend message can be a short sentence or a multi-thousand-character
  // kernel traceback. Split off the first line as a human summary and keep the
  // full text in a bounded, scrollable box so the dialog never outgrows the
  // viewport (which would push the action buttons off-screen).
  const fullMessage = (record.message || '').trim()
  const firstLine = fullMessage.split('\n', 1)[0]
  const hasMoreDetail = fullMessage.length > firstLine.length

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-label={t('snapshotFailureAria')}
      aria-modal="true"
    >
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md max-h-[85vh] flex flex-col">
        <div className="flex items-start justify-between gap-3 p-5 pb-3 shrink-0 border-b border-slate-100">
          <h3 className="text-base font-semibold text-red-700">
            {isRuntimePersistence
              ? t('snapshotFailureRuntimeTitle')
              : t('snapshotFailureTitle')}
          </h3>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="text-xs text-slate-400 hover:text-slate-600 shrink-0"
            >
              {t('snapshotFailureClose')}
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-5 py-4 space-y-4">
          <div className="space-y-2 text-sm text-slate-700">
            <div>
              <span className="font-medium text-slate-500">{t('snapshotFailureCode')}: </span>
              <code className="text-red-700 font-mono text-xs break-all">{record.code}</code>
            </div>
            {firstLine && (
              <p className="text-slate-600 break-words">{firstLine}</p>
            )}
            {hasMoreDetail && (
              <details className="group">
                <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-700 select-none">
                  {t('snapshotFailureTechnicalDetails')}
                </summary>
                <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words bg-slate-50 border border-slate-200 rounded-lg p-2 font-mono text-[11px] leading-snug text-slate-600">
                  {fullMessage}
                </pre>
              </details>
            )}
          </div>

          {variables.length > 0 && (
            <div className="bg-amber-50 border border-amber-100 rounded-lg p-3">
              <div className="text-xs font-semibold text-amber-800 uppercase tracking-wide mb-1">
                {t('snapshotFailureRequiredVariables')}
              </div>
              <ul className="list-disc list-inside text-sm text-amber-900 space-y-0.5">
                {variables.map((name) => (
                  <li key={name}>
                    <code className="font-mono text-xs break-all">{name}</code>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {isPreRestore && (
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-600">
              <div className="font-semibold text-slate-700 mb-1">pre_restore</div>
              {t('snapshotFailurePreRestore')}
            </div>
          )}

          {isPostReport && (
            <p className="text-xs text-slate-500">
              {t('snapshotFailurePostReportRetry')}
            </p>
          )}

          {!isPostReport && actions.has('retry') && (
            <p className="text-xs text-slate-500">
              {t('snapshotFailureRetryCaptureOnly')}
            </p>
          )}
          {isMemoryRefusal && (
            <p className="text-xs text-slate-500">
              {t('snapshotFailureMemoryRefusal')}
            </p>
          )}
          {isRuntimePersistence && (
            <div className="space-y-3">
              {runtimeDetails.degraded_recovery === true && (
                <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3">
                  {t('snapshotFailureDegradedRecovery', {
                    committed: String(runtimeDetails.committed_revision ?? '—'),
                    attempted: String(runtimeDetails.attempted_revision ?? '—'),
                  })}
                </p>
              )}
              {budgetViolations.length > 0 && (
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                  <div className="text-xs font-semibold text-slate-700 mb-2">
                    {t('snapshotFailureResourceLimits')}
                  </div>
                  <ul className="space-y-1 text-xs text-slate-600">
                    {budgetViolations.map((item, index) => (
                      <li key={`${String(item.resource)}-${index}`}>
                        <code>{String(item.resource || 'resource')}</code>: {' '}
                        {typeof item.observed === 'number' ? formatBytes(item.observed) : String(item.observed ?? '—')}
                        {' / '}
                        {typeof item.limit === 'number' ? formatBytes(item.limit) : String(item.limit ?? '—')}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {topOffenders.length > 0 && (
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                  <div className="text-xs font-semibold text-slate-700 mb-2">
                    {t('snapshotFailureLargestObjects')}
                  </div>
                  <ul className="space-y-1 text-xs text-slate-600">
                    {topOffenders.map((item, index) => (
                      <li key={`${String(item.name)}-${index}`} className="flex justify-between gap-3">
                        <span className="truncate">
                          <code>{String(item.name || 'object')}</code>
                          {item.type ? ` (${String(item.type)})` : ''}
                        </span>
                        <span className="shrink-0">{formatBytes(item.bytes)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2 p-5 pt-3 shrink-0 border-t border-slate-100">
          {actions.has('retry') && (
            <button
              type="button"
              onClick={() => onRetry({ regenerateReports: false })}
              className="w-full px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium"
            >
              {isRuntimePersistence
                ? t('snapshotFailureRetryPersistence')
                : t('snapshotFailureRetry')}
            </button>
          )}
          {actions.has('proceed') && (
            <button
              type="button"
              onClick={onProceed}
              className="w-full px-3 py-2 text-sm bg-orange-50 hover:bg-orange-100 text-orange-800 border border-orange-300 rounded-lg font-medium"
              title={
                t('snapshotFailureCaptureRiskTitle')
              }
            >
              {t('snapshotFailureCaptureRisk')}
            </button>
          )}
          {actions.has('skip') && !isPreRestore && (
            <button
              type="button"
              onClick={onSkip}
              className="w-full px-3 py-2 text-sm bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-200 rounded-lg font-medium"
              title={
                t('snapshotFailureSkipTitle')
              }
            >
              {t('snapshotFailureSkip')}
            </button>
          )}
          {actions.has('manage') && (
            <button
              type="button"
              onClick={onManage}
              className="w-full px-3 py-2 text-sm bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg font-medium"
            >
              {isRuntimePersistence
                ? t('snapshotFailureRecomputeRuntime')
                : t('snapshotFailureManage')}
            </button>
          )}
          {actions.has('terminate') && (
            <button
              type="button"
              onClick={onTerminate}
              className="w-full px-3 py-2 text-sm bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 rounded-lg font-medium"
            >
              {t('snapshotFailureTerminate')}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
