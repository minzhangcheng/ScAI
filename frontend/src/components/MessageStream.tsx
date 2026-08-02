import { useEffect, useRef, useState, useMemo, memo } from 'react'
import { useAppStore, Message, AgentType } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import type { GraphicAbstractContent } from '../types'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  Users,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Image as ImageIcon,
  AlertCircle,
  CheckCircle2,
  MessageSquare,
  Sparkles,
  Terminal,
  Send,
  Reply,
  Wrench,
  HelpCircle,
  FileText,
  Clock,
  Loader2,
  Eye,
  TrendingUp,
  AlertTriangle
} from 'lucide-react'
import { base64ToDataUri } from '../utils/image'
import { STRIPPED_FIGURE_PLACEHOLDER } from '../constants/storage'
import { agentIcons, agentBgColors, agentGradients, FALLBACK_AGENT_ICON } from '../constants/agents'

SyntaxHighlighter.registerLanguage('python', python)

// Type interfaces for message content
interface ResultContent {
  findings?: string[]
  summary?: string
  message?: string
  output?: string
  text?: unknown
}

interface CompletedContent {
  status?: string
  final_summary?: string
}

interface FigureContent {
  approved?: boolean
  revision_requested?: boolean
  figure_name?: string
  feedback?: string
  png_base64?: string
}

interface ReportContent {
  report?: string
  figures?: unknown[]
}

// Reusable truncated text with expand/collapse
function TruncatedText({ text, limit, className, language }: { text: string; limit: number; className?: string; language?: string }) {
  const [expanded, setExpanded] = useState(false)
  if (text.length <= limit) return <>{text}</>
  return (
    <>
      {expanded ? text : text.slice(0, limit) + '...'}
      <button
        onClick={() => setExpanded(!expanded)}
        className={`ml-1 text-xs text-blue-500 hover:text-blue-700 underline ${className || ''}`}
      >
        {expanded ? (language === 'zh' ? '收起' : 'Collapse') : (language === 'zh' ? '展开全文' : 'Expand')}
      </button>
    </>
  )
}

const MessageItem = memo(function MessageItem({ message, language }: { message: Message; language: 'zh' | 'en' }) {
  const [expanded, setExpanded] = useState(true)
  const { t } = useTranslation(language)

  const agentNames: Record<AgentType, string> = {
    pi: 'PI Agent',
    pathology: language === 'zh' ? '病理 Agent' : 'Pathology Agent',
    bioinformatics: language === 'zh' ? '生信 Agent' : 'Bioinformatics Agent',
    biochemistry: language === 'zh' ? '生化 Agent' : 'Biochemistry Agent',
    literature: language === 'zh' ? '文献 Agent' : 'Literature Agent',
    painter: language === 'zh' ? 'Painter Agent' : 'Painter Agent',
    supervisor: language === 'zh' ? '监督 Agent' : 'Supervisor Agent',
    technique: language === 'zh' ? '技术 Agent' : 'Technique Agent',
  }

  const agent = message.agent
  const Icon = agent ? agentIcons[agent] : FALLBACK_AGENT_ICON
  const bgColor = agent ? agentBgColors[agent] : 'bg-slate-50 border-slate-200'
  const name = agent ? agentNames[agent] : (language === 'zh' ? '系统' : 'System')

  const renderContent = () => {
    switch (message.type) {
      case 'status':
        return (
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <CheckCircle2 className="w-4 h-4 text-blue-500" />
            <span>{typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}</span>
          </div>
        )
      
      case 'step':
        return (
          <div className="flex items-center gap-2">
            <div className="px-3 py-1.5 rounded-full bg-gradient-to-r from-slate-100 to-slate-200 text-slate-700 text-sm font-medium">
              {typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}
            </div>
          </div>
        )
      
      case 'prompt':
        // 发送给LLM的指令
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-slate-500">
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
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Reply className="w-3 h-3" />
              <span>{language === 'zh' ? 'LLM 回复' : 'LLM Response'}</span>
            </div>
            <div className={`${bgColor} border rounded-lg p-3 text-sm`}>
              <pre className="whitespace-pre-wrap text-slate-700 text-xs">
                <TruncatedText text={typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)} limit={1500} language={language} />
              </pre>
            </div>
          </div>
        )
      
      case 'thinking':
        // 思考过程
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-purple-600">
              <Sparkles className="w-3 h-3" />
              <span>{language === 'zh' ? '思考中...' : 'Thinking...'}</span>
            </div>
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 text-sm text-purple-800 italic">
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
            <div className="flex items-center gap-2 text-xs text-orange-600">
              <Wrench className="w-3 h-3" />
              <span>{language === 'zh' ? '调用工具' : 'Tool Call'}: <code className="bg-orange-100 px-1 rounded">{toolContent.name}</code></span>
            </div>
            {toolContent.args && (
              <div className="bg-orange-50 border border-orange-200 rounded-lg p-2 text-xs">
                <pre className="whitespace-pre-wrap text-slate-700 font-mono">
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
              className="flex items-center gap-2 text-xs text-emerald-600 hover:text-emerald-700"
            >
              {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Terminal className="w-3 h-3" />
              <span>{language === 'zh' ? '执行代码' : 'Execute Code'}</span>
            </button>
            {expanded && (
              <div className="rounded-lg overflow-hidden border border-slate-200">
                <SyntaxHighlighter
                  language="python"
                  style={oneLight}
                  customStyle={{ margin: 0, padding: '0.75rem', fontSize: '0.7rem' }}
                >
                  {typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)}
                </SyntaxHighlighter>
              </div>
            )}
          </div>
        )
      
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
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <ImageIcon className="w-3 h-3" />
              <span>{language === 'zh' ? '生成图表' : 'Generated Figure'}</span>
            </div>
            {stripped ? (
              <div className="flex items-center gap-2 px-3 py-2 text-xs text-slate-400 bg-slate-50 border border-dashed border-slate-200 rounded-lg">
                <ImageIcon className="w-4 h-4" />
                {language === 'zh'
                  ? '图片未随会话保存，请在结果页或报告中查看'
                  : 'Image not retained with the session — see the Results tab or the report'}
              </div>
            ) : (
              <img
                src={base64ToDataUri(message.content as string)}
                alt="Analysis figure"
                className="max-w-full rounded-lg border border-slate-200 shadow-sm"
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
                <span className="text-xs font-medium text-slate-600 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  {language === 'zh' ? '分析发现' : 'Findings'}
                </span>
                <ul className="space-y-1.5">
                  {content.findings.slice(0, 8).map((f: unknown, i: number) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-700 bg-slate-50 p-2 rounded">
                      <span className="text-green-500 font-medium">•</span>
                      {typeof f === 'string' ? f : JSON.stringify(f, null, 2)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {content.summary && (
              <div className="bg-gradient-to-r from-slate-50 to-slate-100 p-3 rounded-lg text-sm text-slate-700 border-l-2 border-slate-300">
                {content.summary}
              </div>
            )}
            {content.message && (
              <p className="text-sm text-slate-600">{content.message}</p>
            )}
            {content.output && (
              <pre className="text-xs text-slate-600 bg-slate-100 rounded p-2 overflow-x-auto">
                {content.output.slice(0, 500)}{content.output.length > 500 ? '...' : ''}
              </pre>
            )}
          </div>
        )
      
      case 'error':
        return (
          <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-3">
            <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
            <span className="text-sm text-red-700">
              {typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}
            </span>
          </div>
        )
      
      case 'output':
        // 代码执行输出
        if (!message.content) return null
        return (
          <div className="space-y-1">
            <span className="text-xs text-slate-500">{language === 'zh' ? '输出' : 'Output'}:</span>
            <pre className="text-xs text-slate-700 bg-slate-100 rounded-lg p-2 overflow-x-auto whitespace-pre-wrap">
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
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center gap-2 text-green-700 font-medium">
              <CheckCircle2 className="w-4 h-4" />
              <span>{language === 'zh' ? '分析完成' : 'Analysis Completed'}</span>
            </div>
            {completedContent.final_summary && (
              <p className="mt-2 text-sm text-green-800">{completedContent.final_summary}</p>
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
          <div className="flex items-center gap-2 text-sm text-blue-600">
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
            <div className={`flex items-center gap-2 text-xs ${isApproved ? 'text-green-600' : isRevision ? 'text-amber-600' : 'text-blue-600'}`}>
              {isApproved ? <CheckCircle2 className="w-3 h-3" /> : isRevision ? <AlertCircle className="w-3 h-3" /> : <Clock className="w-3 h-3" />}
              <span>
                {isApproved
                  ? (language === 'zh' ? '图片审核通过' : 'Figure Approved')
                  : isRevision
                  ? (language === 'zh' ? '图片需要修改' : 'Figure Revision Requested')
                  : (language === 'zh' ? '图片审核中' : 'Figure Under Review')}
              </span>
            </div>
            <div className={`border rounded-lg p-3 ${isApproved ? 'bg-green-50 border-green-200' : isRevision ? 'bg-amber-50 border-amber-200' : 'bg-blue-50 border-blue-200'}`}>
              {figureContent.figure_name && (
                <p className="font-medium text-sm text-slate-800">{figureContent.figure_name}</p>
              )}
              {figureContent.feedback && (
                <p className="text-sm text-slate-600 mt-1">{figureContent.feedback}</p>
              )}
              {figureContent.png_base64 && (
                <img
                  src={base64ToDataUri(figureContent.png_base64)}
                  alt={figureContent.figure_name || 'Figure'}
                  className="mt-2 max-w-xs rounded border border-slate-200"
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
            <div className="flex items-center gap-2 text-xs text-violet-600">
              <FileText className="w-3 h-3" />
              <span>{t('integratedReportGenerated')}</span>
            </div>
            <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
              <p className="text-sm text-violet-800">
                {language === 'zh'
                  ? `报告已生成，包含 ${reportContent.figures?.length || 0} 张图片`
                  : `Report generated with ${reportContent.figures?.length || 0} figures`}
              </p>
              {reportContent.report && (
                <p className="text-xs text-violet-600 mt-1 line-clamp-3">
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
            <div className="flex items-center gap-2 text-xs text-pink-600">
              <Sparkles className="w-3 h-3" />
              <span>
                {isTextMode ? t('graphicAbstractTextMode') : t('graphicAbstractGenerated')}
              </span>
            </div>
            <div className="bg-pink-50 border border-pink-200 rounded-lg p-3">
              {gaContent.prompt && (
                <p className="text-xs text-pink-700 mb-2">{gaContent.prompt}</p>
              )}
              {isTextMode && textDiagram ? (
                <pre className="font-mono text-xs text-slate-800 bg-white border border-pink-200 rounded p-3 whitespace-pre overflow-auto leading-5">
                  {textDiagram}
                </pre>
              ) : imageBase64 ? (
                <img
                  src={base64ToDataUri(imageBase64)}
                  alt="Graphic Abstract"
                  className="max-w-full rounded border border-pink-200"
                />
              ) : null}
            </div>
          </div>
        )
      }

      case 'message':
      default:
        // Check for special new types first
        if (message.type === 'literature') {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-blue-600">
                <BookOpen className="w-3 h-3" />
                <span>{language === 'zh' ? '文献检索结果' : 'Literature Search Results'}</span>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
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
              <div className="flex items-center gap-2 text-xs text-amber-600">
                <AlertTriangle className="w-3 h-3" />
                <span>{language === 'zh' ? '模态提示' : 'Modality Warning'}</span>
              </div>
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-800">
                <pre className="whitespace-pre-wrap text-xs">{typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)}</pre>
              </div>
            </div>
          )
        }

        if (message.type === 'report_blocked') {
          const rb = (typeof message.content === 'object' && message.content) ? message.content as { reason?: string; partial_allowed?: boolean } : {}
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-amber-600">
                <AlertTriangle className="w-3 h-3" />
                <span>{language === 'zh' ? '报告生成受阻' : 'Report Blocked'}</span>
              </div>
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-800">
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
              <div className="flex items-center gap-2 text-xs text-violet-600">
                <TrendingUp className="w-3 h-3" />
                <span>{language === 'zh' ? '多方向相关性分析' : 'Multi-Direction Correlation Analysis'}</span>
              </div>
              {Array.isArray(directions) && directions.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-1">
                  {directions.map((d: string, i: number) => (
                    <span key={i} className="px-2 py-0.5 bg-violet-100 text-violet-700 text-xs rounded-full">{d}</span>
                  ))}
                </div>
              )}
              <div className="bg-violet-50 border border-violet-200 rounded-lg p-3 text-sm text-violet-800">
                <pre className="whitespace-pre-wrap text-xs">{correlation}</pre>
              </div>
            </div>
          )
        }

        if (message.type === 'supervisor_feedback') {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-cyan-600">
                <Eye className="w-3 h-3" />
                <span>{language === 'zh' ? '监督反馈' : 'Supervisor Feedback'}</span>
              </div>
              <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3 text-sm text-cyan-800">
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
              <div className="flex items-center gap-2 text-xs text-cyan-600">
                <Eye className="w-3 h-3" />
                <span>{language === 'zh' ? '监督摘要' : 'Supervision Summary'}</span>
              </div>
              <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3">
                <div className="flex flex-wrap gap-3 text-sm">
                  <span className="text-cyan-700">{language === 'zh' ? '监控步骤' : 'Steps'}: <strong>{s.total_steps_monitored ?? '-'}</strong></span>
                  <span className="text-cyan-700">{language === 'zh' ? '技术问题' : 'Issues'}: <strong>{s.steps_with_technical_issues ?? '-'}</strong></span>
                  <span className="text-cyan-700">{language === 'zh' ? 'PI介入' : 'PI Escalations'}: <strong>{s.pi_escalations ?? '-'}</strong></span>
                  <span className="text-cyan-700">{language === 'zh' ? '严重问题' : 'Critical'}: <strong>{s.critical_issues ?? '-'}</strong></span>
                </div>
              </div>
            </div>
          )
        }

        if (message.type === 'supervisor_checkpoint') {
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-cyan-600">
                <Eye className="w-3 h-3" />
                <span>{language === 'zh' ? '检查点上报' : 'Checkpoint Escalation'}</span>
              </div>
              <div className="bg-cyan-50 border border-cyan-300 rounded-lg p-3 text-sm text-cyan-800">
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
              <div className="flex items-center gap-2 text-xs text-violet-600">
                <HelpCircle className="w-3 h-3" />
                <span>{language === 'zh' ? '咨询 PI Agent' : 'Consulting PI Agent'}</span>
              </div>
              <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
                <p className="text-sm text-violet-800 font-medium">{qd.question || (typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2))}</p>
                {qd.context && (
                  <pre className="mt-2 text-xs text-violet-600 whitespace-pre-wrap">{qd.context}</pre>
                )}
                {Array.isArray(qd.options) && qd.options.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {qd.options.map((opt: string, i: number) => (
                      <span key={i} className="text-xs bg-violet-100 text-violet-700 px-2 py-0.5 rounded">{opt}</span>
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
              <div className="flex items-center gap-2 text-xs text-violet-600">
                <Users className="w-3 h-3" />
                <span>{language === 'zh' ? 'PI Agent 指导' : 'PI Agent Guidance'}</span>
              </div>
              <div className="bg-violet-50 border border-violet-200 rounded-lg p-3 text-sm text-violet-800">
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
              <div className="flex items-center gap-2 text-xs text-green-600">
                <CheckCircle2 className="w-3 h-3" />
                <span>{language === 'zh' ? '分析发现' : 'Analysis Findings'}</span>
              </div>
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                {fd.summary && (
                  <p className="text-sm text-green-800 font-medium mb-2">{fd.summary}</p>
                )}
                {Array.isArray(fd.findings) && fd.findings.length > 0 && (
                  <ul className="space-y-1">
                    {fd.findings.map((f: unknown, i: number) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-green-700">
                        <span className="text-green-500">•</span>
                        {typeof f === 'string' ? f : JSON.stringify(f, null, 2)}
                      </li>
                    ))}
                  </ul>
                )}
                {Array.isArray(fd.suggestions) && fd.suggestions.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-green-200">
                    <span className="text-xs text-green-600 font-medium">{language === 'zh' ? '建议:' : 'Suggestions:'}</span>
                    <ul className="mt-1 space-y-1">
                      {fd.suggestions.map((s: unknown, i: number) => (
                        <li key={i} className="text-xs text-green-600">→ {typeof s === 'string' ? s : JSON.stringify(s, null, 2)}</li>
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
              <div className="flex items-center gap-2 text-sm text-violet-600 bg-violet-50 rounded-lg p-3">
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
              <div className="flex items-center gap-2 text-sm text-green-600 bg-green-50 rounded-lg p-3">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>{delegContent.agent} {language === 'zh' ? '分析完成' : 'agent completed'}</span>
              </div>
            )
          }
          if (delegContent.status === 'agent_error') {
            return (
              <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 rounded-lg p-3">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{delegContent.agent} {language === 'zh' ? '分析失败' : 'agent error'}: {delegContent.error}</span>
              </div>
            )
          }
          if (delegContent.status === 'synthesizing') {
            return (
              <div className="flex items-center gap-2 text-sm text-violet-600 bg-violet-50 rounded-lg p-3">
                <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                <span>{language === 'zh' ? 'PI正在汇总结果...' : 'PI synthesizing results...'}</span>
              </div>
            )
          }
          return (
            <div className="text-sm text-slate-500 bg-slate-50 rounded-lg p-3">
              {delegContent.message || JSON.stringify(delegContent)}
            </div>
          )
        }

        // Default message rendering
        if (typeof message.content === 'string') {
          return (
            <div className="text-sm text-slate-600 bg-slate-50 rounded-lg p-3">
              {message.content}
            </div>
          )
        }
        return (
          <pre className="text-xs text-slate-600 bg-slate-50 rounded-lg p-3 overflow-x-auto">
            {JSON.stringify(message.content, null, 2)}
          </pre>
        )
    }
  }

  // 根据消息类型决定是否显示Agent头部
  const showAgentHeader = !['status', 'progress'].includes(message.type)

  // 根据消息类型决定边框颜色
  const getBorderClass = () => {
    if (message.type === 'error') return 'border-red-200'
    if (message.type === 'prompt') return 'border-surface-300'
    if (message.type === 'response') return 'border-green-200'
    if (message.type === 'tool_call') return 'border-orange-200'
    if (message.type === 'code') return 'border-emerald-200'
    if (message.type === 'completed') return 'border-green-300'
    return 'border-surface-200'
  }

  return (
    <div className={`rounded-lg border bg-white shadow-card overflow-hidden animate-fade-in-up ${getBorderClass()}`}>
      {/* Agent Header */}
      {showAgentHeader && (
        <div className={`px-3 py-2 border-b flex items-center justify-between ${
          message.type === 'prompt' ? 'bg-surface-50 border-surface-200' :
          message.type === 'response' ? 'bg-green-50 border-green-100' :
          message.type === 'tool_call' ? 'bg-orange-50 border-orange-100' :
          message.type === 'code' ? 'bg-emerald-50 border-emerald-100' :
          message.type === 'completed' ? 'bg-green-50 border-green-100' :
          agent ? 'bg-surface-50 border-surface-100' : 'bg-surface-50 border-surface-100'
        }`}>
          <div className="flex items-center gap-2.5">
            {agent && agentGradients[agent] ? (
              <div className={`p-1.5 rounded-lg bg-gradient-to-br ${agentGradients[agent]}`}>
                <Icon className="w-3.5 h-3.5 text-white" />
              </div>
            ) : (
              <div className={`p-1 rounded ${
                message.type === 'prompt' ? 'bg-surface-200' :
                message.type === 'response' ? 'bg-green-100' :
                message.type === 'tool_call' ? 'bg-orange-100' :
                message.type === 'code' ? 'bg-emerald-100' :
                'bg-surface-200'
              }`}>
                <Icon className={`w-3.5 h-3.5 ${
                  message.type === 'prompt' ? 'text-surface-600' :
                  message.type === 'response' ? 'text-green-600' :
                  message.type === 'tool_call' ? 'text-orange-600' :
                  message.type === 'code' ? 'text-emerald-600' :
                  'text-surface-600'
                }`} />
              </div>
            )}
            <span className="text-xs font-semibold tracking-wide uppercase text-surface-500">
              {name}
            </span>
            {message.type === 'prompt' && (
              <span className="text-xs bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded flex items-center gap-1">
                <Send className="w-2.5 h-2.5" /> LLM
              </span>
            )}
            {message.type === 'response' && (
              <span className="text-xs bg-green-100 text-green-700 px-1.5 py-0.5 rounded flex items-center gap-1">
                <Reply className="w-2.5 h-2.5" /> LLM
              </span>
            )}
            {message.type === 'tool_call' && (
              <span className="text-xs bg-orange-100 text-orange-700 px-1.5 py-0.5 rounded flex items-center gap-1">
                <Wrench className="w-2.5 h-2.5" /> Tool
              </span>
            )}
          </div>
          <span className="text-[10px] text-surface-400">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
      )}
      
      {/* Content */}
      <div className={`p-3 ${message.type === 'status' ? 'py-2' : ''}`}>
        {renderContent()}
      </div>
    </div>
  )
}, (prevProps, nextProps) => {
  // Re-render if message id, content, timestamp, or language changed
  return prevProps.message.id === nextProps.message.id &&
         prevProps.message.content === nextProps.message.content &&
         prevProps.message.timestamp === nextProps.message.timestamp &&
         prevProps.language === nextProps.language
})

const COLLAPSED_MESSAGE_THRESHOLD = 20  // Only expand recent 20 messages

export function MessageStream() {
  const { messages, language } = useAppStore()
  const containerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [showCollapsed, setShowCollapsed] = useState(false)
  const [agentFilter, setAgentFilter] = useState<AgentType | 'all'>('all')

  // Filter messages by agent
  const filteredMessages = useMemo(() => {
    if (agentFilter === 'all') return messages
    return messages.filter(msg => msg.agent === agentFilter || !msg.agent)
  }, [messages, agentFilter])

  // Split messages into collapsed (older) and visible (recent 20)
  const { collapsedMessages, visibleMessages } = useMemo(() => {
    if (filteredMessages.length <= COLLAPSED_MESSAGE_THRESHOLD) {
      return { collapsedMessages: [], visibleMessages: filteredMessages }
    }
    const splitIndex = filteredMessages.length - COLLAPSED_MESSAGE_THRESHOLD
    return {
      collapsedMessages: filteredMessages.slice(0, splitIndex),
      visibleMessages: filteredMessages.slice(splitIndex)
    }
  }, [filteredMessages])

  // Get unique agents from messages for filter dropdown
  const availableAgents = useMemo(() => {
    const agents = new Set<AgentType>()
    messages.forEach(msg => {
      if (msg.agent) agents.add(msg.agent)
    })
    return Array.from(agents)
  }, [messages])

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (containerRef.current && autoScroll) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [visibleMessages, autoScroll])

  // Detect manual scroll
  const handleScroll = () => {
    if (containerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = containerRef.current
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 100
      setAutoScroll(isAtBottom)
    }
  }

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-4 text-center">
        <div className="text-surface-400">
          <MessageSquare className="w-10 h-10 mx-auto mb-3 opacity-30" />
          <p className="text-sm font-medium">{language === 'zh' ? '等待分析开始' : 'Waiting for Analysis'}</p>
          <p className="text-xs mt-1 text-surface-400">
            {language === 'zh' ? '消息将在这里实时显示' : 'Messages will appear here in real-time'}
          </p>
        </div>
      </div>
    )
  }

  const agentNames: Record<AgentType, string> = {
    pi: 'PI Agent',
    pathology: language === 'zh' ? '病理 Agent' : 'Pathology',
    bioinformatics: language === 'zh' ? '生信 Agent' : 'Bioinformatics',
    biochemistry: language === 'zh' ? '生化 Agent' : 'Biochemistry',
    literature: language === 'zh' ? '文献 Agent' : 'Literature',
    painter: 'Painter',
    supervisor: language === 'zh' ? '监督 Agent' : 'Supervisor',
    technique: language === 'zh' ? '技术 Agent' : 'Technique',
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-surface-200/60 bg-white/90 backdrop-blur-sm flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-surface-400" />
          <span className="text-sm font-medium text-surface-700">
            {language === 'zh' ? '实时消息' : 'Real-time Messages'}
          </span>
          <span className="text-xs bg-accent-50 text-accent-600 px-1.5 py-0.5 rounded-full font-medium">
            {filteredMessages.length}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {/* Agent Filter */}
          <select
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value as AgentType | 'all')}
            className="text-xs border border-surface-200 rounded-lg px-2 py-1 bg-white text-surface-600 focus:outline-none focus:ring-2 focus:ring-accent-500/20 focus:border-accent-400"
          >
            <option value="all">{language === 'zh' ? '全部 Agent' : 'All Agents'}</option>
            {availableAgents.map(agent => (
              <option key={agent} value={agent}>{agentNames[agent]}</option>
            ))}
          </select>
          {!autoScroll && (
            <button
              onClick={() => {
                setAutoScroll(true)
                if (containerRef.current) {
                  containerRef.current.scrollTop = containerRef.current.scrollHeight
                }
              }}
              className="text-xs text-blue-600 hover:text-blue-700"
            >
              {language === 'zh' ? '↓ 滚动到底部' : '↓ Scroll to bottom'}
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-3 space-y-2 bg-surface-50"
      >
        {/* Collapsed older messages */}
        {collapsedMessages.length > 0 && (
          <div className="mb-2">
            <button
              onClick={() => setShowCollapsed(!showCollapsed)}
              className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-surface-200 hover:bg-surface-300 rounded-lg text-sm text-surface-600 transition-colors"
            >
              {showCollapsed ? (
                <>
                  <ChevronDown className="w-4 h-4" />
                  {language === 'zh'
                    ? `收起 ${collapsedMessages.length} 条旧消息`
                    : `Hide ${collapsedMessages.length} older messages`}
                </>
              ) : (
                <>
                  <ChevronRight className="w-4 h-4" />
                  {language === 'zh'
                    ? `展开 ${collapsedMessages.length} 条旧消息`
                    : `Show ${collapsedMessages.length} older messages`}
                </>
              )}
            </button>
            {showCollapsed && (
              <div className="mt-2 space-y-2">
                {collapsedMessages.map((msg) => (
                  <MessageItem key={msg.id} message={msg} language={language} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Recent visible messages */}
        {visibleMessages.map((msg) => (
          <MessageItem key={msg.id} message={msg} language={language} />
        ))}
      </div>
    </div>
  )
}
