import { useState } from 'react'
import { useAppStore } from '../../stores/appStore'
import { useTranslation } from '../../i18n/translations'
import { downloadBlob } from '../../utils/download'
import { authHeaders } from '../../hooks/useApi'
import { reportBranchQuery } from '../../utils/reportBranch'
import ReactMarkdown from 'react-markdown'
import {
  FileText,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Download
} from 'lucide-react'

const API_BASE = '/api'

interface IntegratedReportProps {
  report: string
}

export function IntegratedReport({ report }: IntegratedReportProps) {
  const { currentSession, language } = useAppStore()
  const { t } = useTranslation(language)
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)

  if (!report) {
    return null
  }

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(report)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (e) {
      console.error('Failed to copy:', e)
    }
  }

  const downloadAsMarkdown = () => {
    const blob = new Blob([report], { type: 'text/markdown' })
    downloadBlob(blob, `integrated_report_${currentSession?.id || 'report'}.md`)
  }

  const downloadAsHTML = async () => {
    if (!currentSession) return
    try {
      const response = await fetch(`${API_BASE}/session/${currentSession.id}/report/html${reportBranchQuery(currentSession.restoreBranchId)}`, {
        headers: authHeaders(),
      })
      if (response.ok) {
        const html = await response.text()
        downloadBlob(new Blob([html], { type: 'text/html' }), `integrated_report_${currentSession.id}.html`)
      }
    } catch (e) {
      console.error('Failed to download HTML:', e)
    }
  }

  return (
    <div className="mt-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
        >
          <FileText className="w-4 h-4" />
          {t('integratedReport')}
          {expanded ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        <div className="flex items-center gap-2">
          <button
            onClick={copyToClipboard}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 rounded transition-colors"
            title={language === 'zh' ? '复制到剪贴板' : 'Copy to clipboard'}
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-green-500" />
                {language === 'zh' ? '已复制' : 'Copied'}
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                {language === 'zh' ? '复制' : 'Copy'}
              </>
            )}
          </button>
          <button
            onClick={downloadAsMarkdown}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 rounded transition-colors"
          >
            <Download className="w-3 h-3" />
            MD
          </button>
          <button
            onClick={downloadAsHTML}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-violet-100 dark:bg-violet-900/60 hover:bg-violet-200 dark:hover:bg-violet-800/70 text-violet-600 dark:text-violet-400 rounded transition-colors"
          >
            <Download className="w-3 h-3" />
            HTML
          </button>
        </div>
      </div>

      {/* Report Content */}
      {expanded && (
        <div className="bg-white dark:bg-surface-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 prose prose-slate prose-sm max-w-none">
          <ReactMarkdown
            components={{
              // Custom rendering for figure references
              a: ({ href, children }) => {
                // Check if it's a figure reference like [Figure 1](#figure-1)
                if (href?.startsWith('#figure-')) {
                  return (
                    <a
                      href={href}
                      className="text-violet-600 dark:text-violet-400 hover:text-violet-800 dark:hover:text-violet-200 font-medium no-underline hover:underline"
                      onClick={(e) => {
                        e.preventDefault()
                        // Could scroll to figure in gallery
                        const figureId = href.replace('#figure-', '')
                        const el = document.getElementById(`figure-${figureId}`)
                        if (el) {
                          el.scrollIntoView({ behavior: 'smooth' })
                        }
                      }}
                    >
                      {children}
                    </a>
                  )
                }
                return (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200"
                  >
                    {children}
                  </a>
                )
              },
              // Style tables
              table: ({ children }) => (
                <div className="overflow-x-auto my-4">
                  <table className="min-w-full border-collapse border border-slate-200 dark:border-slate-700">
                    {children}
                  </table>
                </div>
              ),
              th: ({ children }) => (
                <th className="border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 px-3 py-2 text-left text-sm font-medium">
                  {children}
                </th>
              ),
              td: ({ children }) => (
                <td className="border border-slate-200 dark:border-slate-700 px-3 py-2 text-sm">
                  {children}
                </td>
              ),
              // Style code blocks
              code: ({ className, children }) => {
                const isInline = !className
                if (isInline) {
                  return (
                    <code className="bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-100 px-1 py-0.5 rounded text-sm">
                      {children}
                    </code>
                  )
                }
                return (
                  <code className={className}>
                    {children}
                  </code>
                )
              },
              // Style headings
              h1: ({ children }) => (
                <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-6 mb-4 pb-2 border-b border-slate-200 dark:border-slate-700">
                  {children}
                </h1>
              ),
              h2: ({ children }) => (
                <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mt-5 mb-3">
                  {children}
                </h2>
              ),
              h3: ({ children }) => (
                <h3 className="text-lg font-medium text-slate-800 dark:text-slate-100 mt-4 mb-2">
                  {children}
                </h3>
              ),
              // Style lists
              ul: ({ children }) => (
                <ul className="list-disc list-inside space-y-1 my-2">
                  {children}
                </ul>
              ),
              ol: ({ children }) => (
                <ol className="list-decimal list-inside space-y-1 my-2">
                  {children}
                </ol>
              ),
              // Style blockquotes
              blockquote: ({ children }) => (
                <blockquote className="border-l-4 border-violet-300 dark:border-violet-700 pl-4 my-4 italic text-slate-600 dark:text-slate-300">
                  {children}
                </blockquote>
              ),
            }}
          >
            {report}
          </ReactMarkdown>
        </div>
      )}
    </div>
  )
}
