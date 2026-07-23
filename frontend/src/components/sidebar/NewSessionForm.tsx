// New-session creation form. Split out of Sidebar.tsx; JSX copied verbatim.
import { Clock } from 'lucide-react'
import { ROUND_WAIT_OPTIONS } from '../../constants/roundWait'
import { useTranslation } from '../../i18n/translations'

interface NewSessionFormProps {
  show: boolean
  sessionName: string
  setSessionName: (value: string) => void
  instruction: string
  setInstruction: (value: string) => void
  roundWaitSeconds: number
  setRoundWaitSeconds: (value: number) => void
  onCancel: () => void
  onCreate: () => void
  t: ReturnType<typeof useTranslation>['t']
  language: string
}

export function NewSessionForm({
  show,
  sessionName,
  setSessionName,
  instruction,
  setInstruction,
  roundWaitSeconds,
  setRoundWaitSeconds,
  onCancel,
  onCreate,
  t,
  language,
}: NewSessionFormProps) {
  if (!show) return null

  return (
    <div className="px-3 pb-3">
      <div className="p-4 bg-surface-50 rounded-xl space-y-3 border border-surface-200">
        <input
          type="text"
          value={sessionName}
          onChange={(e) => setSessionName(e.target.value)}
          placeholder={t('sessionNamePlaceholder')}
          className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
        />
        <textarea
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          placeholder={t('researchGoalPlaceholder')}
          rows={12}
          className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500 resize-y min-h-[200px]"
        />

        {/* Round Wait Time Setting */}
        <div className="p-3 bg-surface-50 rounded-lg border border-surface-200">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-accent-600" />
            <label className="text-sm font-medium text-surface-700">
              {language === 'zh' ? '轮次等待时间' : 'Round Wait Time'}
            </label>
          </div>
          <select
            value={roundWaitSeconds}
            onChange={(e) => setRoundWaitSeconds(parseInt(e.target.value))}
            className="w-full text-sm font-mono text-surface-700 bg-white px-3 py-1.5 rounded border border-surface-200 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
          >
            {ROUND_WAIT_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>
                {language === 'zh' ? opt.labelZh : opt.labelEn}
              </option>
            ))}
          </select>
          <p className="text-xs text-surface-500 mt-2 leading-relaxed">
            {language === 'zh'
              ? '💡 每轮分析后等待用户输入的时间，超时后自动继续'
              : '💡 Wait time for user input between rounds, auto-continues on timeout'}
          </p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={onCancel}
            className="flex-1 px-3 py-2 text-sm text-surface-500 hover:bg-surface-100 rounded-lg"
          >
            {t('cancel')}
          </button>
          <button
            onClick={onCreate}
            disabled={!instruction.trim()}
            className="flex-1 px-3 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50 font-medium"
          >
            {t('create')}
          </button>
        </div>
      </div>
    </div>
  )
}
