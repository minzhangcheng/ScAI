import { useEffect, useRef, useState, useMemo, memo } from 'react'
import { useAppStore, Message, AgentType } from '../stores/appStore'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  Users,
  Microscope,
  Code2,
  FlaskConical,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Image as ImageIcon,
  AlertCircle,
  CheckCircle2,
  MessageSquare,
  Bot,
  Sparkles,
  Terminal,
  Send,
  Reply,
  Wrench,
  HelpCircle,
  Palette,
  FileText,
  Clock
} from 'lucide-react'

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

interface GraphicAbstractContent {
  prompt?: string
  image_base64?: string
  imageBase64?: string
}

const agentIcons: Record<AgentType, typeof Users> = {
  pi: Users,
  pathology: Microscope,
  bioinformatics: Code2,
  biochemistry: FlaskConical,
  literature: BookOpen,
  painter: Palette,
}

const agentBgColors: Record<AgentType, string> = {
  pi: 'bg-violet-50 border-violet-200',
  pathology: 'bg-rose-50 border-rose-200',
  bioinformatics: 'bg-emerald-50 border-emerald-200',
  biochemistry: 'bg-amber-50 border-amber-200',
  literature: 'bg-blue-50 border-blue-200',
  painter: 'bg-pink-50 border-pink-200',
}

const MessageItem = memo(function MessageItem({ message, language }: { message: Message; language: 'zh' | 'en' }) {
  const [expanded, setExpanded] = useState(true)

  const agentNames: Record<AgentType, string> = {
    pi: 'PI Agent',
    pathology: language === 'zh' ? '病理 Agent' : 'Pathology Agent',
    bioinformatics: language === 'zh' ? '生信 Agent' : 'Bioinformatics Agent',
    biochemistry: language === 'zh' ? '生化 Agent' : 'Biochemistry Agent',
    literature: language === 'zh' ? '文献 Agent' : 'Literature Agent',
    painter: language === 'zh' ? 'Painter Agent' : 'Painter Agent',
  }

  const agent = message.agent
  const Icon = agent ? agentIcons[agent] : Bot
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
              {message.content}
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
                {typeof message.content === 'string' 
                  ? message.content.slice(0, 1000) + (message.content.length > 1000 ? '...' : '')
                  : JSON.stringify(message.content, null, 2).slice(0, 1000)}
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
                {typeof message.content === 'string' 
                  ? message.content.slice(0, 1500) + (message.content.length > 1500 ? '...' : '')
                  : JSON.stringify(message.content, null, 2).slice(0, 1500)}
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
              {message.content}
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
                  {message.content}
                </SyntaxHighlighter>
              </div>
            )}
          </div>
        )
      
      case 'figure':
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <ImageIcon className="w-3 h-3" />
              <span>{language === 'zh' ? '生成图表' : 'Generated Figure'}</span>
            </div>
            <img
              src={`data:image/png;base64,${message.content}`}
              alt="Analysis figure"
              className="max-w-full rounded-lg border border-slate-200 shadow-sm"
            />
          </div>
        )
      
      case 'result':
        const content: ResultContent = (message.content !== null && typeof message.content === 'object' && !Array.isArray(message.content))
          ? message.content as ResultContent
          : { text: message.content }
        return (
          <div className="space-y-3">
            {content.findings && content.findings.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-medium text-slate-600 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  {language === 'zh' ? '分析发现' : 'Findings'}
                </span>
                <ul className="space-y-1.5">
                  {content.findings.slice(0, 8).map((f: string, i: number) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-700 bg-slate-50 p-2 rounded">
                      <span className="text-green-500 font-medium">•</span>
                      {f}
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
              {typeof message.content === 'string' 
                ? message.content.slice(0, 1000) + (message.content.length > 1000 ? '...' : '')
                : JSON.stringify(message.content, null, 2)}
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
      
      case 'progress':
        // 进度更新
        return (
          <div className="flex items-center gap-2 text-sm text-blue-600">
            <CheckCircle2 className="w-4 h-4" />
            <span>{typeof message.content === 'string' ? message.content : `${message.content}%`}</span>
          </div>
        )
      
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
                  src={`data:image/png;base64,${figureContent.png_base64}`}
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
              <span>{language === 'zh' ? '整合报告生成' : 'Integrated Report Generated'}</span>
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
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-pink-600">
              <Sparkles className="w-3 h-3" />
              <span>{language === 'zh' ? '图形摘要生成' : 'Graphic Abstract Generated'}</span>
            </div>
            <div className="bg-pink-50 border border-pink-200 rounded-lg p-3">
              {gaContent.prompt && (
                <p className="text-xs text-pink-700 mb-2">{gaContent.prompt}</p>
              )}
              {imageBase64 && (
                <img
                  src={`data:image/png;base64,${imageBase64}`}
                  alt="Graphic Abstract"
                  className="max-w-full rounded border border-pink-200"
                />
              )}
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
                  {typeof message.content === 'string' 
                    ? message.content.slice(0, 1500) + (message.content.length > 1500 ? '...' : '')
                    : JSON.stringify(message.content, null, 2)}
                </pre>
              </div>
            </div>
          )
        }
        
        if (message.type === 'pi_question') {
          let questionData = message.content
          try {
            if (typeof message.content === 'string') {
              questionData = JSON.parse(message.content)
            }
          } catch { questionData = { question: message.content } }
          
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-violet-600">
                <HelpCircle className="w-3 h-3" />
                <span>{language === 'zh' ? '咨询 PI Agent' : 'Consulting PI Agent'}</span>
              </div>
              <div className="bg-violet-50 border border-violet-200 rounded-lg p-3">
                <p className="text-sm text-violet-800 font-medium">{questionData.question || message.content}</p>
                {questionData.context && (
                  <pre className="mt-2 text-xs text-violet-600 whitespace-pre-wrap">{questionData.context}</pre>
                )}
                {questionData.options && questionData.options.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {questionData.options.map((opt: string, i: number) => (
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
          let findingsData = message.content
          try {
            if (typeof message.content === 'string') {
              findingsData = JSON.parse(message.content)
            }
          } catch { findingsData = { findings: [message.content] } }
          
          return (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-green-600">
                <CheckCircle2 className="w-3 h-3" />
                <span>{language === 'zh' ? '分析发现' : 'Analysis Findings'}</span>
              </div>
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                {findingsData.summary && (
                  <p className="text-sm text-green-800 font-medium mb-2">{findingsData.summary}</p>
                )}
                {findingsData.findings && findingsData.findings.length > 0 && (
                  <ul className="space-y-1">
                    {findingsData.findings.map((f: string, i: number) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-green-700">
                        <span className="text-green-500">•</span>
                        {f}
                      </li>
                    ))}
                  </ul>
                )}
                {findingsData.suggestions && findingsData.suggestions.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-green-200">
                    <span className="text-xs text-green-600 font-medium">{language === 'zh' ? '建议:' : 'Suggestions:'}</span>
                    <ul className="mt-1 space-y-1">
                      {findingsData.suggestions.map((s: string, i: number) => (
                        <li key={i} className="text-xs text-green-600">→ {s}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
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
    if (message.type === 'prompt') return 'border-slate-300'
    if (message.type === 'response') return 'border-green-200'
    if (message.type === 'tool_call') return 'border-orange-200'
    if (message.type === 'code') return 'border-emerald-200'
    if (message.type === 'completed') return 'border-green-300'
    return 'border-slate-200'
  }

  return (
    <div className={`rounded-lg border bg-white shadow-sm overflow-hidden ${getBorderClass()}`}>
      {/* Agent Header */}
      {showAgentHeader && (
        <div className={`px-3 py-2 border-b flex items-center justify-between ${
          message.type === 'prompt' ? 'bg-slate-100 border-slate-200' :
          message.type === 'response' ? 'bg-green-50 border-green-100' :
          message.type === 'tool_call' ? 'bg-orange-50 border-orange-100' :
          message.type === 'code' ? 'bg-emerald-50 border-emerald-100' :
          message.type === 'completed' ? 'bg-green-50 border-green-100' :
          agent ? 'bg-slate-50 border-slate-100' : 'bg-slate-50 border-slate-100'
        }`}>
          <div className="flex items-center gap-2">
            {Icon && (
              <div className={`p-1 rounded ${
                message.type === 'prompt' ? 'bg-slate-200' :
                message.type === 'response' ? 'bg-green-100' :
                message.type === 'tool_call' ? 'bg-orange-100' :
                message.type === 'code' ? 'bg-emerald-100' :
                'bg-slate-200'
              }`}>
                <Icon className={`w-3.5 h-3.5 ${
                  message.type === 'prompt' ? 'text-slate-600' :
                  message.type === 'response' ? 'text-green-600' :
                  message.type === 'tool_call' ? 'text-orange-600' :
                  message.type === 'code' ? 'text-emerald-600' :
                  'text-slate-600'
                }`} />
              </div>
            )}
            <span className="text-sm font-medium text-slate-700">
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
          <span className="text-xs text-slate-400">
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
  // Only re-render if message id or language changed
  return prevProps.message.id === nextProps.message.id &&
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
        <div className="text-slate-400">
          <MessageSquare className="w-10 h-10 mx-auto mb-3 opacity-30" />
          <p className="text-sm font-medium">{language === 'zh' ? '等待分析开始' : 'Waiting for Analysis'}</p>
          <p className="text-xs mt-1 text-slate-400">
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
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 border-b border-slate-200 bg-white flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-slate-500" />
          <span className="text-sm font-medium text-slate-700">
            {language === 'zh' ? '实时消息' : 'Real-time Messages'}
          </span>
          <span className="text-xs bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">
            {filteredMessages.length}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {/* Agent Filter */}
          <select
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value as AgentType | 'all')}
            className="text-xs border border-slate-200 rounded px-2 py-1 bg-white text-slate-600 focus:outline-none focus:ring-1 focus:ring-blue-400"
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
        className="flex-1 overflow-y-auto p-3 space-y-2 bg-slate-100"
      >
        {/* Collapsed older messages */}
        {collapsedMessages.length > 0 && (
          <div className="mb-2">
            <button
              onClick={() => setShowCollapsed(!showCollapsed)}
              className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-slate-200 hover:bg-slate-300 rounded-lg text-sm text-slate-600 transition-colors"
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
