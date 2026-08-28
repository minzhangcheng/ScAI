import { memo, useState } from 'react'
import { Reply, Send, Wrench } from 'lucide-react'

import { agentBgColors, agentGradients, agentIcons, FALLBACK_AGENT_ICON } from '../../constants/agents'
import { useTranslation } from '../../i18n/translations'
import type { AgentType, Message } from '../../stores/appStore'
import { useSystemTheme } from '../SystemThemeProvider'
import { renderMessageContent } from './renderers'

export const MessageItem = memo(function MessageItem({ message, language }: { message: Message; language: 'zh' | 'en' }) {
  const [expanded, setExpanded] = useState(true)
  const { t } = useTranslation(language)
  const { isDark } = useSystemTheme()

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
  const bgColor = agent ? agentBgColors[agent] : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700'
  const name = agent ? agentNames[agent] : (language === 'zh' ? '系统' : 'System')

  // 根据消息类型决定是否显示Agent头部
  const showAgentHeader = !['status', 'progress'].includes(message.type)

  // 根据消息类型决定边框颜色
  const getBorderClass = () => {
    if (message.type === 'error') return 'border-red-200 dark:border-red-800'
    if (message.type === 'prompt') return 'border-surface-300 dark:border-surface-600'
    if (message.type === 'response') return 'border-green-200 dark:border-green-800'
    if (message.type === 'tool_call') return 'border-orange-200 dark:border-orange-800'
    if (message.type === 'code') return 'border-emerald-200 dark:border-emerald-800'
    if (message.type === 'completed') return 'border-green-300 dark:border-green-700'
    return 'border-surface-200 dark:border-surface-700'
  }

  return (
    <div className={`rounded-lg border bg-white dark:bg-surface-900 shadow-card overflow-hidden animate-fade-in-up ${getBorderClass()}`}>
      {/* Agent Header */}
      {showAgentHeader && (
        <div className={`px-3 py-2 border-b flex items-center justify-between ${
          message.type === 'prompt' ? 'bg-surface-50 dark:bg-surface-950 border-surface-200 dark:border-surface-700' :
          message.type === 'response' ? 'bg-green-50 dark:bg-green-950/40 border-green-100 dark:border-green-900' :
          message.type === 'tool_call' ? 'bg-orange-50 dark:bg-orange-950/40 border-orange-100 dark:border-orange-900' :
          message.type === 'code' ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-100 dark:border-emerald-900' :
          message.type === 'completed' ? 'bg-green-50 dark:bg-green-950/40 border-green-100 dark:border-green-900' :
          agent ? 'bg-surface-50 dark:bg-surface-950 border-surface-100 dark:border-surface-800' : 'bg-surface-50 dark:bg-surface-950 border-surface-100 dark:border-surface-800'
        }`}>
          <div className="flex items-center gap-2.5">
            {agent && agentGradients[agent] ? (
              <div className={`p-1.5 rounded-lg bg-gradient-to-br ${agentGradients[agent]}`}>
                <Icon className="w-3.5 h-3.5 text-white" />
              </div>
            ) : (
              <div className={`p-1 rounded ${
                message.type === 'prompt' ? 'bg-surface-200 dark:bg-surface-700' :
                message.type === 'response' ? 'bg-green-100 dark:bg-green-900/60' :
                message.type === 'tool_call' ? 'bg-orange-100 dark:bg-orange-900/60' :
                message.type === 'code' ? 'bg-emerald-100 dark:bg-emerald-900/60' :
                'bg-surface-200 dark:bg-surface-700'
              }`}>
                <Icon className={`w-3.5 h-3.5 ${
                  message.type === 'prompt' ? 'text-surface-600 dark:text-surface-300' :
                  message.type === 'response' ? 'text-green-600 dark:text-green-400' :
                  message.type === 'tool_call' ? 'text-orange-600 dark:text-orange-400' :
                  message.type === 'code' ? 'text-emerald-600 dark:text-emerald-400' :
                  'text-surface-600 dark:text-surface-300'
                }`} />
              </div>
            )}
            <span className="text-xs font-semibold tracking-wide uppercase text-surface-500 dark:text-surface-400">
              {name}
            </span>
            {message.type === 'prompt' && (
              <span className="text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 px-1.5 py-0.5 rounded flex items-center gap-1">
                <Send className="w-2.5 h-2.5" /> LLM
              </span>
            )}
            {message.type === 'response' && (
              <span className="text-xs bg-green-100 dark:bg-green-900/60 text-green-700 dark:text-green-300 px-1.5 py-0.5 rounded flex items-center gap-1">
                <Reply className="w-2.5 h-2.5" /> LLM
              </span>
            )}
            {message.type === 'tool_call' && (
              <span className="text-xs bg-orange-100 dark:bg-orange-900/60 text-orange-700 dark:text-orange-300 px-1.5 py-0.5 rounded flex items-center gap-1">
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
        {renderMessageContent({ message, language, expanded, setExpanded, isDark, bgColor, t })}
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
