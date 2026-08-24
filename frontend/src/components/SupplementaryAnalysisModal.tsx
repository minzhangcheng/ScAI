import { useState } from 'react'
import { Language } from '../i18n/translations'

interface SupplementaryAnalysisModalProps {
  language: Language
  placeholder: string
  onSubmit: (requestText: string) => Promise<void>
  onClose: () => void
}

export function SupplementaryAnalysisModal({
  language,
  placeholder,
  onSubmit,
  onClose,
}: SupplementaryAnalysisModalProps) {
  const isZh = language === 'zh'
  const [text, setText] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async () => {
    if (!text.trim()) return
    setSubmitting(true)
    try {
      await onSubmit(text.trim())
      onClose()
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/30">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-xl w-full max-w-lg mx-4">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-3">
            {isZh ? '请求补充分析' : 'Request Supplementary Analysis'}
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
            {isZh
              ? '请描述您希望进行的补充分析，PI Agent将组织各Agent执行分析后更新报告。'
              : 'Describe the additional analysis you need. The PI Agent will coordinate agents to run the analysis and update the report.'}
          </p>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder={placeholder}
            className="w-full h-32 px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-xl bg-white dark:bg-surface-900 text-slate-800 dark:text-slate-100 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-accent-500"
            autoFocus
          />
          <div className="flex justify-end gap-3 mt-4">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-xl"
            >
              {isZh ? '取消' : 'Cancel'}
            </button>
            <button
              onClick={handleSubmit}
              disabled={!text.trim() || submitting}
              className="px-5 py-2 text-sm font-medium text-white bg-accent-600 hover:bg-accent-700 rounded-xl disabled:opacity-50"
            >
              {submitting
                ? (isZh ? '提交中...' : 'Submitting...')
                : (isZh ? '提交' : 'Submit')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
