import { useState, type ReactNode } from 'react'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  AlertCircle, AlertTriangle, BookOpen, CheckCircle2, ChevronDown, ChevronRight,
  Clock, Eye, FileText, HelpCircle, Image as ImageIcon, Loader2, Reply, Send,
  Sparkles, Terminal, TrendingUp, Users, Wrench,
} from 'lucide-react'

import { STRIPPED_FIGURE_PLACEHOLDER } from '../../constants/storage'
import type { GraphicAbstractContent } from '../../types'
import { base64ToDataUri } from '../../utils/image'
import type {
  CompletedContent, FigureContent, MessageRenderContext, ReportContent, ResultContent,
} from './types'

SyntaxHighlighter.registerLanguage('python', python)

function TruncatedText({ text, limit, className, language }: { text: string; limit: number; className?: string; language?: string }) {
  const [expanded, setExpanded] = useState(false)
  if (text.length <= limit) return <>{text}</>
  return (
    <>
      {expanded ? text : text.slice(0, limit) + '...'}
      <button
        onClick={() => setExpanded(!expanded)}
        className={`ml-1 text-xs text-blue-500 hover:text-blue-700 dark:hover:text-blue-300 underline ${className || ''}`}
      >
        {expanded ? (language === 'zh' ? '收起' : 'Collapse') : (language === 'zh' ? '展开全文' : 'Expand')}
      </button>
    </>
  )
}

function renderBasicMessage(context: MessageRenderContext): ReactNode | undefined {
  const { message, language, expanded, setExpanded, isDark, bgColor } = context
  switch (message.type) {
      case 'status':
        return (
          <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
            <CheckCircle2 className="w-4 h-4 text-blue-500" />
            <span>{typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}</span>
          </div>
        )

      case 'step':
        return (
          <div className="flex items-center gap-2">
            <div className="px-3 py-1.5 rounded-full bg-gradient-to-r from-slate-100 dark:from-slate-800 to-slate-200 dark:to-slate-700 text-slate-700 dark:text-slate-200 text-sm font-medium">
              {typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}
            </div>
          </div>
        )

      case 'prompt':
        // 发送给LLM的指令
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
              <Send className="w-3 h-3" />
              <span>{language === 'zh' ? '发送给 LLM' : 'Sent to LLM'}</span>
            </div>
            <div className="bg-slate-800 rounded-lg p-3 text-sm text-slate-100 overflow-x-auto">
              <pre className="whitespace-pre-wrap font-mono text-xs">
                <TruncatedText text={typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)} limit={1000} language={language} />
              </pre>
            </div>
          </div>
        )

      case 'response':
        // LLM的回复
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
              <Reply className="w-3 h-3" />
              <span>{language === 'zh' ? 'LLM 回复' : 'LLM Response'}</span>
            </div>
            <div className={`${bgColor} border rounded-lg p-3 text-sm`}>
              <pre className="whitespace-pre-wrap text-slate-700 dark:text-slate-200 text-xs">
                <TruncatedText text={typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)} limit={1500} language={language} />
              </pre>
            </div>
          </div>
        )

      case 'thinking':
        // 思考过程
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-purple-600 dark:text-purple-400">
              <Sparkles className="w-3 h-3" />
              <span>{language === 'zh' ? '思考中...' : 'Thinking...'}</span>
            </div>
            <div className="bg-purple-50 dark:bg-purple-950/40 border border-purple-200 dark:border-purple-800 rounded-lg p-3 text-sm text-purple-800 dark:text-purple-200 italic">
              {typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}
            </div>
          </div>
        )

      case 'tool_call':
        // 工具调用（如function calling）
        const toolContent = (message.content !== null && typeof message.content === 'object' && !Array.isArray(message.content))
          ? message.content
          : { name: 'unknown', args: message.content }
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-orange-600 dark:text-orange-400">
              <Wrench className="w-3 h-3" />
              <span>{language === 'zh' ? '调用工具' : 'Tool Call'}: <code className="bg-orange-100 dark:bg-orange-900/60 px-1 rounded">{toolContent.name}</code></span>
            </div>
            {toolContent.args && (
              <div className="bg-orange-50 dark:bg-orange-950/40 border border-orange-200 dark:border-orange-800 rounded-lg p-2 text-xs">
                <pre className="whitespace-pre-wrap text-slate-700 dark:text-slate-200 font-mono">
                  {typeof toolContent.args === 'string' ? toolContent.args : JSON.stringify(toolContent.args, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )

      case 'code':
        return (
          <div className="space-y-2">
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-2 text-xs text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300"
            >
              {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Terminal className="w-3 h-3" />
              <span>{language === 'zh' ? '执行代码' : 'Execute Code'}</span>
            </button>
            {expanded && (
              <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
                <SyntaxHighlighter
                  language="python"
                  style={isDark ? oneDark : oneLight}
                  customStyle={{ margin: 0, padding: '0.75rem', fontSize: '0.7rem' }}
                >
                  {typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)}
                </SyntaxHighlighter>
              </div>
            )}
          </div>
        )

      default:
        return undefined
    }
}

function renderAnalysisMessage(context: MessageRenderContext): ReactNode | undefined {
  const { message, language } = context
  switch (message.type) {
      case 'figure': {
        // The backend replaces a figure's base64 with a sentinel before
        // persisting it, so a reloaded session carries the marker rather than
        // an image. Rendering that as a source produced a broken <img> for
        // every figure in the history; say what happened instead.
        const stripped =
          typeof message.content !== 'string' ||
          message.content === STRIPPED_FIGURE_PLACEHOLDER ||
          !message.content
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
              <ImageIcon className="w-3 h-3" />
              <span>{language === 'zh' ? '生成图表' : 'Generated Figure'}</span>
            </div>
            {stripped ? (
              <div className="flex items-center gap-2 px-3 py-2 text-xs text-slate-400 bg-slate-50 dark:bg-slate-800 border border-dashed border-slate-200 dark:border-slate-700 rounded-lg">
                <ImageIcon className="w-4 h-4" />
                {language === 'zh'
                  ? '图片未随会话保存，请在结果页或报告中查看'
                  : 'Image not retained with the session — see the Results tab or the report'}
              </div>
            ) : (
              <img
                src={base64ToDataUri(message.content as string)}
                alt="Analysis figure"
                className="publication-canvas max-w-full rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm"
                style={{ maxWidth: '800px', maxHeight: '600px' }}
              />
            )}
          </div>
        )
      }

      case 'result':
        const content: ResultContent = (message.content !== null && typeof message.content === 'object' && !Array.isArray(message.content))
          ? message.content as ResultContent
          : { text: message.content }
        return (
          <div className="space-y-3">
            {Array.isArray(content.findings) && content.findings.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-medium text-slate-600 dark:text-slate-300 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  {language === 'zh' ? '分析发现' : 'Findings'}
                </span>
                <ul className="space-y-1.5">
                  {content.findings.slice(0, 8).map((f: unknown, i: number) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-700 dark:text-slate-200 bg-slate-50 dark:bg-slate-800 p-2 rounded">
                      <span className="text-green-500 font-medium">•</span>
                      {typeof f === 'string' ? f : JSON.stringify(f, null, 2)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {content.summary && (
              <div className="bg-gradient-to-r from-slate-50 dark:from-slate-900 to-slate-100 dark:to-slate-800 p-3 rounded-lg text-sm text-slate-700 dark:text-slate-200 border-l-2 border-slate-300 dark:border-slate-600">
                {content.summary}
              </div>
            )}
            {content.message && (
              <p className="text-sm text-slate-600 dark:text-slate-300">{content.message}</p>
            )}
            {content.output && (
              <pre className="text-xs text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 rounded p-2 overflow-x-auto">
                {content.output.slice(0, 500)}{content.output.length > 500 ? '...' : ''}
              </pre>
            )}
          </div>
        )

      case 'error':
        return (
          <div className="flex items-start gap-2 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 rounded-lg p-3">
            <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
            <span className="text-sm text-red-700 dark:text-red-300">
              {typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}
            </span>
          </div>
        )

      case 'output':
        // 代码执行输出
        if (!message.content) return null
        return (
          <div className="space-y-1">
            <span className="text-xs text-slate-500 dark:text-slate-400">{language === 'zh' ? '输出' : 'Output'}:</span>
            <pre className="text-xs text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 rounded-lg p-2 overflow-x-auto whitespace-pre-wrap">
              <TruncatedText text={typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)} limit={1000} language={language} />
            </pre>
          </div>
        )

      case 'completed':
        // 分析完成
        const completedContent: CompletedContent = (message.content !== null && typeof message.content === 'object' && !Array.isArray(message.content))
          ? message.content as CompletedContent
          : { status: 'completed' }
        return (
          <div className="bg-green-50 dark:bg-green-950/40 border border-green-200 dark:border-green-800 rounded-lg p-3">
            <div className="flex items-center gap-2 text-green-700 dark:text-green-300 font-medium">
              <CheckCircle2 className="w-4 h-4" />
              <span>{language === 'zh' ? '分析完成' : 'Analysis Completed'}</span>
            </div>
            {completedContent.final_summary && (
              <p className="mt-2 text-sm text-green-800 dark:text-green-200">{completedContent.final_summary}</p>
            )}
          </div>
        )

      case 'progress': {
        // 进度更新 - content can be: string, number, or {step, progress, status}
        const progressContent = message.content
        let progressText = ''
        let progressPercent: number | null = null

        if (typeof progressContent === 'string') {
          progressText = progressContent
        } else if (typeof progressContent === 'number') {
          progressPercent = progressContent
        } else if (progressContent && typeof progressContent === 'object') {
          progressText = (progressContent as any).step || ''
          progressPercent = (progressContent as any).progress ?? null
        }

        const isComplete = progressPercent !== null && progressPercent >= 100

        return (
          <div className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400">
            {isComplete
              ? <CheckCircle2 className="w-4 h-4 text-green-500" />
              : <Loader2 className="w-4 h-4 animate-spin" />}
            <span>
              {progressText}
              {progressPercent !== null && ` (${Math.round(progressPercent)}%)`}
            </span>
          </div>
        )
      }

      default:
        return undefined
    }
}

function renderReportMessage(context: MessageRenderContext): ReactNode | undefined {
  const { message, language, t } = context
  switch (message.type) {
      case 'figure_review':
      case 'figure_approved':
      case 'figure_revision': {
        const figureContent: FigureContent = (message.content !== null && typeof message.content === 'object' && !Array.isArray(message.content))
          ? message.content as FigureContent
          : {}
        const isApproved = message.type === 'figure_approved' || Boolean(figureContent.approved)
        const isRevision = message.type === 'figure_revision' || Boolean(figureContent.revision_requested)

        return (
          <div className="space-y-2">
            <div className={`flex items-center gap-2 text-xs ${isApproved ? 'text-green-600 dark:text-green-400' : isRevision ? 'text-amber-600 dark:text-amber-400' : 'text-blue-600 dark:text-blue-400'}`}>
              {isApproved ? <CheckCircle2 className="w-3 h-3" /> : isRevision ? <AlertCircle className="w-3 h-3" /> : <Clock className="w-3 h-3" />}
              <span>
                {isApproved
                  ? (language === 'zh' ? '图片审核通过' : 'Figure Approved')
                  : isRevision
                  ? (language === 'zh' ? '图片需要修改' : 'Figure Revision Requested')
                  : (language === 'zh' ? '图片审核中' : 'Figure Under Review')}
              </span>
            </div>
            <div className={`border rounded-lg p-3 ${isApproved ? 'bg-green-50 dark:bg-green-950/40 border-green-200 dark:border-green-800' : isRevision ? 'bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800' : 'bg-blue-50 dark:bg-blue-950/40 border-blue-200 dark:border-blue-800'}`}>
              {figureContent.figure_name && (
                <p className="font-medium text-sm text-slate-800 dark:text-slate-100">{figureContent.figure_name}</p>
              )}
              {figureContent.feedback && (
                <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">{figureContent.feedback}</p>
              )}
              {figureContent.png_base64 && (
                <img
                  src={base64ToDataUri(figureContent.png_base64)}
                  alt={figureContent.figure_name || 'Figure'}
                  className="publication-canvas mt-2 max-w-xs rounded border border-slate-200 dark:border-slate-700"
                />
              )}
            </div>
          </div>
        )
      }

      case 'integrated_report': {
        const reportContent: ReportContent = (message.content !== null && typeof message.content === 'object' && !Array.isArray(message.content))
          ? message.content as ReportContent
          : { report: String(message.content) }
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-violet-600 dark:text-violet-400">
              <FileText className="w-3 h-3" />
              <span>{t('integratedReportGenerated')}</span>
            </div>
            <div className="bg-violet-50 dark:bg-violet-950/40 border border-violet-200 dark:border-violet-800 rounded-lg p-3">
              <p className="text-sm text-violet-800 dark:text-violet-200">
                {language === 'zh'
                  ? `报告已生成，包含 ${reportContent.figures?.length || 0} 张图片`
                  : `Report generated with ${reportContent.figures?.length || 0} figures`}
              </p>
              {reportContent.report && (
                <p className="text-xs text-violet-600 dark:text-violet-400 mt-1 line-clamp-3">
                  {reportContent.report.substring(0, 200)}...
                </p>
              )}
            </div>
          </div>
        )
      }

      case 'graphic_abstract': {
        const gaContent: GraphicAbstractContent = (message.content !== null && typeof message.content === 'object' && !Array.isArray(message.content))
          ? message.content as GraphicAbstractContent
          : {}
        const imageBase64 = gaContent.image_base64 || gaContent.imageBase64
        const textDiagram = gaContent.text_diagram || gaContent.textDiagram
        const isTextMode = gaContent.is_text_mode ?? gaContent.isTextMode ?? !!textDiagram
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-pink-600 dark:text-pink-400">
              <Sparkles className="w-3 h-3" />
              <span>
                {isTextMode ? t('graphicAbstractTextMode') : t('graphicAbstractGenerated')}
              </span>
            </div>
            <div className="bg-pink-50 dark:bg-pink-950/40 border border-pink-200 dark:border-pink-800 rounded-lg p-3">
              {gaContent.prompt && (
                <p className="text-xs text-pink-700 dark:text-pink-300 mb-2">{gaContent.prompt}</p>
              )}
              {isTextMode && textDiagram ? (
                <pre className="publication-canvas font-mono text-xs text-slate-800 bg-white border border-pink-200 dark:border-pink-800 rounded p-3 whitespace-pre overflow-auto leading-5">
                  {textDiagram}
                </pre>
              ) : imageBase64 ? (
                <img
                  src={base64ToDataUri(imageBase64)}
                  alt="Graphic Abstract"
                  className="publication-canvas max-w-full rounded border border-pink-200 dark:border-pink-800"
                />
              ) : null}
            </div>
          </div>
        )
      }

      default:
        return undefined
    }
}

function renderLegacyMessage(context: MessageRenderContext): ReactNode | undefined {
  const { message, language } = context
  switch (message.type) {
      case 'message':
      default:
        // Check for special new types first
        if (message.type === 'literature') {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-blue-600 dark:text-blue-400">
                <BookOpen className="w-3 h-3" />
                <span>{language === 'zh' ? '文献检索结果' : 'Literature Search Results'}</span>
              </div>
              <div className="bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded-lg p-3 text-sm text-blue-800 dark:text-blue-200">
                <pre className="whitespace-pre-wrap text-xs">
                  <TruncatedText text={typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)} limit={1500} language={language} />
                </pre>
              </div>
            </div>
          )
        }

        if (message.type === 'modality_warning') {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400">
                <AlertTriangle className="w-3 h-3" />
                <span>{language === 'zh' ? '模态提示' : 'Modality Warning'}</span>
              </div>
              <div className="bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded-lg p-3 text-sm text-amber-800 dark:text-amber-200">
                <pre className="whitespace-pre-wrap text-xs">{typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)}</pre>
              </div>
            </div>
          )
        }

        if (message.type === 'report_blocked') {
          const rb = (typeof message.content === 'object' && message.content) ? message.content as { reason?: string; partial_allowed?: boolean } : {}
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400">
                <AlertTriangle className="w-3 h-3" />
                <span>{language === 'zh' ? '报告生成受阻' : 'Report Blocked'}</span>
              </div>
              <div className="bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded-lg p-3 text-sm text-amber-800 dark:text-amber-200">
                <pre className="whitespace-pre-wrap text-xs">{rb.reason || (typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2))}</pre>
              </div>
            </div>
          )
        }

        if (message.type === 'direction_correlation') {
          const data = typeof message.content === 'object' ? message.content : {}
          const correlation = data.correlation || (typeof message.content === 'string' ? message.content : '')
          const directions = data.directions || []
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-violet-600 dark:text-violet-400">
                <TrendingUp className="w-3 h-3" />
                <span>{language === 'zh' ? '多方向相关性分析' : 'Multi-Direction Correlation Analysis'}</span>
              </div>
              {Array.isArray(directions) && directions.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-1">
                  {directions.map((d: string, i: number) => (
                    <span key={i} className="px-2 py-0.5 bg-violet-100 dark:bg-violet-900/60 text-violet-700 dark:text-violet-300 text-xs rounded-full">{d}</span>
                  ))}
                </div>
              )}
              <div className="bg-violet-50 dark:bg-violet-950/40 border border-violet-200 dark:border-violet-800 rounded-lg p-3 text-sm text-violet-800 dark:text-violet-200">
                <pre className="whitespace-pre-wrap text-xs">{correlation}</pre>
              </div>
            </div>
          )
        }

        if (message.type === 'supervisor_feedback') {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-cyan-600 dark:text-cyan-400">
                <Eye className="w-3 h-3" />
                <span>{language === 'zh' ? '监督反馈' : 'Supervisor Feedback'}</span>
              </div>
              <div className="bg-cyan-50 dark:bg-cyan-950/40 border border-cyan-200 dark:border-cyan-800 rounded-lg p-3 text-sm text-cyan-800 dark:text-cyan-200">
                <pre className="whitespace-pre-wrap text-xs">
                  {typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)}
                </pre>
              </div>
            </div>
          )
        }

        if (message.type === 'supervisor_summary') {
          let stats: Record<string, any> | string = message.content
          try {
            if (typeof message.content === 'string') {
              stats = JSON.parse(message.content)
            }
          } catch { console.debug('Failed to parse supervisor_summary'); stats = {} }
          const s = typeof stats === 'string' ? {} : stats

          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-cyan-600 dark:text-cyan-400">
                <Eye className="w-3 h-3" />
                <span>{language === 'zh' ? '监督摘要' : 'Supervision Summary'}</span>
              </div>
              <div className="bg-cyan-50 dark:bg-cyan-950/40 border border-cyan-200 dark:border-cyan-800 rounded-lg p-3">
                <div className="flex flex-wrap gap-3 text-sm">
                  <span className="text-cyan-700 dark:text-cyan-300">{language === 'zh' ? '监控步骤' : 'Steps'}: <strong>{s.total_steps_monitored ?? '-'}</strong></span>
                  <span className="text-cyan-700 dark:text-cyan-300">{language === 'zh' ? '技术问题' : 'Issues'}: <strong>{s.steps_with_technical_issues ?? '-'}</strong></span>
                  <span className="text-cyan-700 dark:text-cyan-300">{language === 'zh' ? 'PI介入' : 'PI Escalations'}: <strong>{s.pi_escalations ?? '-'}</strong></span>
                  <span className="text-cyan-700 dark:text-cyan-300">{language === 'zh' ? '严重问题' : 'Critical'}: <strong>{s.critical_issues ?? '-'}</strong></span>
                </div>
              </div>
            </div>
          )
        }

        if (message.type === 'supervisor_checkpoint') {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-cyan-600 dark:text-cyan-400">
                <Eye className="w-3 h-3" />
                <span>{language === 'zh' ? '检查点上报' : 'Checkpoint Escalation'}</span>
              </div>
              <div className="bg-cyan-50 dark:bg-cyan-950/40 border border-cyan-300 dark:border-cyan-700 rounded-lg p-3 text-sm text-cyan-800 dark:text-cyan-200">
                {typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}
              </div>
            </div>
          )
        }

        if (message.type === 'pi_question') {
          let questionData: Record<string, any> | string = message.content
          try {
            if (typeof message.content === 'string') {
              questionData = JSON.parse(message.content)
            }
          } catch { questionData = { question: message.content } }
          const qd = typeof questionData === 'string' ? {} : questionData

          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-violet-600 dark:text-violet-400">
                <HelpCircle className="w-3 h-3" />
                <span>{language === 'zh' ? '咨询 PI Agent' : 'Consulting PI Agent'}</span>
              </div>
              <div className="bg-violet-50 dark:bg-violet-950/40 border border-violet-200 dark:border-violet-800 rounded-lg p-3">
                <p className="text-sm text-violet-800 dark:text-violet-200 font-medium">{qd.question || (typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2))}</p>
                {qd.context && (
                  <pre className="mt-2 text-xs text-violet-600 dark:text-violet-400 whitespace-pre-wrap">{qd.context}</pre>
                )}
                {Array.isArray(qd.options) && qd.options.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {qd.options.map((opt: string, i: number) => (
                      <span key={i} className="text-xs bg-violet-100 dark:bg-violet-900/60 text-violet-700 dark:text-violet-300 px-2 py-0.5 rounded">{opt}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )
        }

        if (message.type === 'pi_response') {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-violet-600 dark:text-violet-400">
                <Users className="w-3 h-3" />
                <span>{language === 'zh' ? 'PI Agent 指导' : 'PI Agent Guidance'}</span>
              </div>
              <div className="bg-violet-50 dark:bg-violet-950/40 border border-violet-200 dark:border-violet-800 rounded-lg p-3 text-sm text-violet-800 dark:text-violet-200">
                <pre className="whitespace-pre-wrap text-xs">
                  {typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)}
                </pre>
              </div>
            </div>
          )
        }

        if (message.type === 'findings') {
          let findingsData: Record<string, any> | string = message.content
          try {
            if (typeof message.content === 'string') {
              findingsData = JSON.parse(message.content)
            }
          } catch { findingsData = { findings: [message.content] } }
          const fd = typeof findingsData === 'string' ? {} : findingsData

          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400">
                <CheckCircle2 className="w-3 h-3" />
                <span>{language === 'zh' ? '分析发现' : 'Analysis Findings'}</span>
              </div>
              <div className="bg-green-50 dark:bg-green-950/40 border border-green-200 dark:border-green-800 rounded-lg p-3">
                {fd.summary && (
                  <p className="text-sm text-green-800 dark:text-green-200 font-medium mb-2">{fd.summary}</p>
                )}
                {Array.isArray(fd.findings) && fd.findings.length > 0 && (
                  <ul className="space-y-1">
                    {fd.findings.map((f: unknown, i: number) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-green-700 dark:text-green-300">
                        <span className="text-green-500">•</span>
                        {typeof f === 'string' ? f : JSON.stringify(f, null, 2)}
                      </li>
                    ))}
                  </ul>
                )}
                {Array.isArray(fd.suggestions) && fd.suggestions.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-green-200 dark:border-green-800">
                    <span className="text-xs text-green-600 dark:text-green-400 font-medium">{language === 'zh' ? '建议:' : 'Suggestions:'}</span>
                    <ul className="mt-1 space-y-1">
                      {fd.suggestions.map((s: unknown, i: number) => (
                        <li key={i} className="text-xs text-green-600 dark:text-green-400">→ {typeof s === 'string' ? s : JSON.stringify(s, null, 2)}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )
        }

        // Chat delegation progress
        if (message.type === 'chat_delegation') {
          const delegContent = message.content as { status: string; plan?: string; agent?: string; agents?: string[]; error?: string; message?: string }
          if (delegContent.status === 'started') {
            return (
              <div className="flex items-center gap-2 text-sm text-violet-600 dark:text-violet-400 bg-violet-50 dark:bg-violet-950/40 rounded-lg p-3">
                <Sparkles className="w-4 h-4 shrink-0" />
                <div>
                  <div className="font-medium">{language === 'zh' ? 'Chat分析委派开始' : 'Chat delegation started'}</div>
                  {delegContent.plan && <div className="text-xs mt-0.5">{delegContent.plan}</div>}
                  {delegContent.agents && <div className="text-xs text-violet-400 mt-0.5">Agents: {delegContent.agents.join(', ')}</div>}
                </div>
              </div>
            )
          }
          if (delegContent.status === 'agent_complete') {
            return (
              <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/40 rounded-lg p-3">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>{delegContent.agent} {language === 'zh' ? '分析完成' : 'agent completed'}</span>
              </div>
            )
          }
          if (delegContent.status === 'agent_error') {
            return (
              <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/40 rounded-lg p-3">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{delegContent.agent} {language === 'zh' ? '分析失败' : 'agent error'}: {delegContent.error}</span>
              </div>
            )
          }
          if (delegContent.status === 'synthesizing') {
            return (
              <div className="flex items-center gap-2 text-sm text-violet-600 dark:text-violet-400 bg-violet-50 dark:bg-violet-950/40 rounded-lg p-3">
                <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                <span>{language === 'zh' ? 'PI正在汇总结果...' : 'PI synthesizing results...'}</span>
              </div>
            )
          }
          return (
            <div className="text-sm text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
              {delegContent.message || JSON.stringify(delegContent)}
            </div>
          )
        }

        // Default message rendering
        if (typeof message.content === 'string') {
          return (
            <div className="text-sm text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800 rounded-lg p-3">
              {message.content}
            </div>
          )
        }
        return (
          <pre className="text-xs text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800 rounded-lg p-3 overflow-x-auto">
            {JSON.stringify(message.content, null, 2)}
          </pre>
        )
    }
}

export function renderMessageContent(context: MessageRenderContext): ReactNode {
  return renderBasicMessage(context)
    ?? renderAnalysisMessage(context)
    ?? renderReportMessage(context)
    ?? renderLegacyMessage(context)
}
