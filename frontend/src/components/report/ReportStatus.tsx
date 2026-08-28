import { AlertCircle, FileText, Loader2, RefreshCw } from 'lucide-react'

import type { ReportController } from './useReportController'

interface ReportStatusProps {
  controller: ReportController
}
export function ReportStatus({ controller }: ReportStatusProps) {
  const { error, generateReport, generating, language, loading, preview, report } = controller
  return (
      <div className="flex-1 flex flex-col min-h-0">
        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <Loader2 className="w-8 h-8 animate-spin text-accent-600" />
          </div>
        ) : error ? (
          <div className="flex-1 overflow-auto p-4">
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                <span>{error}</span>
              </div>
            </div>
          </div>
        ) : !report || report.status === 'pending' ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
            <FileText className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg mb-2">
              {language === 'zh' ? '详细报告尚未生成' : 'Detailed report not yet generated'}
            </p>
            <p className="text-sm mb-4">
              {language === 'zh'
                ? '报告可能正在生成中，或者生成过程中出现了问题'
                : 'Report may be generating, or there was an issue during generation'}
            </p>
            <button
              onClick={generateReport}
              disabled={generating}
              className="flex items-center gap-1.5 px-4 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50"
            >
              {generating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              {generating
                ? (language === 'zh' ? '生成中...' : 'Generating...')
                : (language === 'zh' ? '生成报告' : 'Generate Report')}
            </button>
          </div>
        ) : report.status === 'generating' ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
            <Loader2 className="w-16 h-16 mb-4 animate-spin text-accent-500" />
            <p className="text-lg mb-2">
              {language === 'zh' ? '正在生成详细报告...' : 'Generating detailed report...'}
            </p>
            <p className="text-sm text-accent-600">
              {report.current_step || (language === 'zh' ? '准备中...' : 'Preparing...')}
            </p>
            {report.progress > 0 && (
              <div className="w-64 mt-4">
                <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-accent-600 transition-all duration-300"
                    style={{ width: `${report.progress}%` }}
                  />
                </div>
                <p className="text-xs text-center mt-1">{Math.round(report.progress)}%</p>
              </div>
            )}
          </div>
        ) : preview?.has_html ? (
          // HTML Report - use src instead of srcDoc so relative paths work
          <iframe
            src={preview.html_url}
            className="publication-canvas w-full flex-1 border-0 min-h-0"
            title="Report"
          />
        ) : (
          // Report record exists but the rendered HTML is not available
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
            <FileText className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg mb-2">
              {language === 'zh' ? 'HTML 报告暂不可用' : 'HTML report not available'}
            </p>
            <p className="text-sm mb-4">
              {language === 'zh'
                ? '报告记录已存在，但渲染后的 HTML 文件缺失，请尝试重新生成。'
                : 'A report record exists, but the rendered HTML file is missing. Try regenerating.'}
            </p>
            <button
              onClick={generateReport}
              disabled={generating}
              className="flex items-center gap-1.5 px-4 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50"
            >
              {generating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              {generating
                ? (language === 'zh' ? '生成中...' : 'Generating...')
                : (language === 'zh' ? '重新生成' : 'Regenerate')}
            </button>
          </div>
        )}
      </div>

  )
}
