import { useState } from 'react'

export const DISCARD_CONFIRM_PHRASE = 'DISCARD CURRENT STATE'

interface DiscardCurrentDialogProps {
  isZh?: boolean
  message?: string
  onAbort: () => void
  onConfirm: () => void
}

export function DiscardCurrentDialog({
  isZh = false,
  message,
  onAbort,
  onConfirm,
}: DiscardCurrentDialogProps) {
  const [phrase, setPhrase] = useState('')
  const canConfirm = phrase.trim() === DISCARD_CONFIRM_PHRASE

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40" role="dialog" aria-label="Discard current">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4 p-5 space-y-4">
        <h3 className="text-base font-semibold text-red-700">
          {isZh ? '丢弃当前状态' : 'Discard Current'}
        </h3>
        <p className="text-sm text-slate-600">
          {message ||
            (isZh
              ? '预恢复快照不可用。继续将以仅前进模式丢弃当前状态，无法回滚。'
              : 'Pre-restore capture is unavailable. Continuing discards current state in forward-only mode and cannot roll back.')}
        </p>
        <label className="block text-xs text-slate-500" htmlFor="discard-confirm-phrase">
          {isZh ? '请输入确认短语' : 'Type confirmation phrase'}
          <span className="ml-1 font-mono text-slate-700">{DISCARD_CONFIRM_PHRASE}</span>
        </label>
        <input
          id="discard-confirm-phrase"
          aria-label="Type confirmation phrase"
          className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm"
          value={phrase}
          onChange={(e) => setPhrase(e.target.value)}
          autoComplete="off"
        />
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onAbort}
            className="px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-100 rounded-lg border border-slate-200"
          >
            {isZh ? '中止' : 'Abort'}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={!canConfirm}
            className="px-3 py-1.5 text-xs bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isZh ? '确认丢弃' : 'Confirm Discard'}
          </button>
        </div>
      </div>
    </div>
  )
}
