import { ExternalLink, FileText, Image, Loader2, MessageCircle, Package, RefreshCw, Table } from 'lucide-react'

import { API_BASE } from '../../constants/api'
import { ErrorBanner } from '../shared/ErrorBanner'
import { ValidationDialog } from '../ValidationDialog'
import { ReportStatus } from './ReportStatus'
import { type ReportController } from './useReportController'

interface ReportViewProps {
  controller: ReportController
}
export function ReportView({ controller }: ReportViewProps) {
  const {
    archiveWarning, currentSession, doGenerateReport, downloadError, downloadReportArtifact,
    generating, language, openHtmlInNewTab, preview, report, reportArchives, safeDownload,
    setDownloadError, setShowValidationDialog, showConfigPanel, showValidationDialog, t,
    validating, validationMessage, validationUncertain, generateReport, continueFollowupInChat,
  } = controller
  if (!currentSession) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>{language === 'zh' ? '请先创建或选择一个分析会话' : 'Please create or select an analysis session first'}</p>
        </div>
      </div>
    )
  }

  if (currentSession.status !== 'completed') {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <Loader2 className="w-16 h-16 mx-auto mb-4 animate-spin text-accent-400" />
          <p className="text-lg font-medium mb-2">
            {language === 'zh' ? '分析进行中...' : 'Analysis in progress...'}
          </p>
          <p className="text-sm">
            {language === 'zh'
              ? '分析完成后将自动生成详细报告'
              : 'Detailed report will be generated automatically when analysis completes'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Validating spinner overlay */}
      {validating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-lg p-6 flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-accent-600" />
            <span className="text-slate-700 font-medium">
              {t('validatingApiKey')}
            </span>
          </div>
        </div>
      )}

      {/* Validation failure dialog */}
      {showValidationDialog && (
        <ValidationDialog
          message={validationMessage}
          uncertain={validationUncertain}
          onIgnore={() => { setShowValidationDialog(false); doGenerateReport() }}
          onReconfigure={() => { setShowValidationDialog(false); if (showConfigPanel) showConfigPanel() }}
          language={language}
        />
      )}

      {/* Header */}
      <div className="p-4 border-b border-slate-200 bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-accent-600" />
            <h2 className="font-display text-xl font-semibold text-slate-900">
              {language === 'zh' ? '详细研究报告' : 'Detailed Research Report'}
            </h2>
          </div>

          <div className="flex items-center gap-2">
            {report && (report.status === 'completed' || report.status === 'partial') && (
              <>
                {preview?.has_html && (
                  <button
                    onClick={openHtmlInNewTab}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-accent-50 text-accent-700 rounded-lg hover:bg-accent-100"
                  >
                    <ExternalLink className="w-4 h-4" />
                    {language === 'zh' ? '新窗口打开' : 'Open in Tab'}
                  </button>
                )}

                <button
                  onClick={() => downloadReportArtifact('download-all-figures', 'figures')}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-pink-50 text-pink-700 rounded-lg hover:bg-pink-100"
                >
                  <Image className="w-4 h-4" />
                  {language === 'zh' ? '下载图片' : 'Figures'}
                </button>
                <button
                  onClick={() => downloadReportArtifact('download-all-tables', 'tables')}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-emerald-50 text-emerald-700 rounded-lg hover:bg-emerald-100"
                >
                  <Table className="w-4 h-4" />
                  {language === 'zh' ? '下载表格' : 'Tables'}
                </button>
                <button
                  onClick={() => downloadReportArtifact('download-full', 'full_report')}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-accent-50 text-accent-700 rounded-lg hover:bg-accent-100"
                >
                  <Package className="w-4 h-4" />
                  {language === 'zh' ? '打包下载' : 'Download All'}
                </button>
              </>
            )}

            {/* Only show regenerate button when report exists */}
            {report && (report.status === 'completed' || report.status === 'partial') && (
              <button
                onClick={generateReport}
                disabled={generating}
                className="flex items-center gap-1.5 px-4 py-2 text-sm bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 disabled:opacity-50"
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
            )}
          </div>
        </div>

        {downloadError && (
          <ErrorBanner message={downloadError} onDismiss={() => setDownloadError(null)} />
        )}

        {archiveWarning && (
          <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-800">
            {language === 'zh' ? '归档历史报告失败：' : 'Previous report archive failed: '}{archiveWarning}
          </div>
        )}

        {reportArchives.length > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-200">
            <h4 className="text-sm font-medium text-slate-600 mb-2">
              {language === 'zh' ? '历史报告' : 'Previous Reports'}
            </h4>
            <div className="space-y-1">
              {reportArchives.map(archive => (
                <button
                  key={archive.filename}
                  onClick={() => {
                    safeDownload(`${API_BASE}/${archive.download_url}`, archive.filename)
                  }}
                  className="block text-sm text-blue-600 hover:text-blue-800 hover:underline text-left cursor-pointer"
                >
                  v{archive.version} — {new Date(archive.created_at).toLocaleString()} ({(archive.size_bytes / 1024 / 1024).toFixed(1)} MB)
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Progress bar */}
        {generating && report && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm text-slate-600 mb-2">
              <span>{report.current_step}</span>
              <span>{Math.round(report.progress)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-accent-600 transition-all duration-300"
                style={{ width: `${report.progress}%` }}
              />
            </div>
          </div>
        )}

        {report?.followups && report.followups.length > 0 && (
          <section className="mt-4 pt-4 border-t border-slate-200" aria-labelledby="report-followups-heading">
            <h3 id="report-followups-heading" className="text-sm font-semibold text-slate-700 mb-2">
              {language === 'zh' ? '后续问题' : 'Follow-ups'}
            </h3>
            <div className="space-y-2">
              {report.followups.map((followup) => {
                const disposition = String(followup.disposition || '').toLowerCase()
                const needsNewData = disposition === 'needs_new_data'
                const canContinue = disposition !== 'closed'
                  && Boolean(followup.suggested_instruction?.trim())
                const dispositionLabel = needsNewData
                  ? (language === 'zh' ? '需要新数据' : 'Needs new data')
                  : disposition === 'offer_in_chat'
                    ? (language === 'zh' ? '在 Chat 中继续' : 'Continue in Chat')
                    : disposition === 'deepen_now'
                      ? (language === 'zh' ? '现在深入' : 'Deepen now')
                      : (language === 'zh' ? 'PI 已记录' : 'Recorded by PI')
                return (
                  <article key={followup.followup_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <p className="text-sm font-medium text-slate-800">{followup.question}</p>
                      <span className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-medium ${
                        needsNewData
                          ? 'bg-amber-100 text-amber-800'
                          : 'bg-accent-100 text-accent-800'
                      }`}>
                        {dispositionLabel}
                      </span>
                    </div>
                    {followup.why_it_matters && (
                      <p className="mt-1 text-xs text-slate-600">{followup.why_it_matters}</p>
                    )}
                    {followup.reason && (
                      <p className="mt-1 text-xs text-slate-500">{followup.reason}</p>
                    )}
                    {followup.suggested_instruction && (
                      <p className="mt-1 text-xs text-slate-700">
                        <span className="font-medium">{language === 'zh' ? '建议指令：' : 'Suggested instruction: '}</span>
                        {followup.suggested_instruction}
                      </p>
                    )}
                    {followup.required_input && (
                      <p className="mt-1 text-xs text-amber-800">
                        <span className="font-medium">{language === 'zh' ? '所需输入：' : 'Required input: '}</span>
                        {followup.required_input}
                      </p>
                    )}
                    {followup.execution_outcome && (
                      <p className="mt-1 text-xs text-slate-600">
                        <span className="font-medium">{language === 'zh' ? '尝试结果：' : 'Attempted outcome: '}</span>
                        {followup.execution_outcome}
                      </p>
                    )}
                    {followup.source_round && (
                      <p className="mt-1 text-[11px] text-slate-400">
                        {language === 'zh' ? `来源：第 ${followup.source_round} 轮` : `Source: Round ${followup.source_round}`}
                        {followup.direction ? ` · ${followup.direction}` : ''}
                      </p>
                    )}
                    {canContinue && (
                      <button
                        type="button"
                        onClick={() => continueFollowupInChat(followup)}
                        className="mt-2 inline-flex items-center gap-1.5 rounded-md bg-accent-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-accent-700"
                      >
                        <MessageCircle className="h-3.5 w-3.5" />
                        {language === 'zh' ? '在 Chat 中继续' : 'Continue in Chat'}
                      </button>
                    )}
                  </article>
                )
              })}
            </div>
          </section>
        )}
      </div>

      <ReportStatus controller={controller} />
    </div>
  )
}
