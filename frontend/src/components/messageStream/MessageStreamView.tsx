import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, MessageSquare } from 'lucide-react'

import { useAppStore, type AgentType } from '../../stores/appStore'
import { MessageItem } from './MessageItem'

const COLLAPSED_MESSAGE_THRESHOLD = 20  // Only expand recent 20 messages

export function MessageStreamView() {
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
      <div className="px-4 py-2.5 border-b border-surface-200/60 dark:border-surface-700/60 bg-white/90 dark:bg-surface-900/90 backdrop-blur-sm flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-surface-400" />
          <span className="text-sm font-medium text-surface-700 dark:text-surface-200">
            {language === 'zh' ? '实时消息' : 'Real-time Messages'}
          </span>
          <span className="text-xs bg-accent-50 dark:bg-accent-900/40 text-accent-600 dark:text-accent-400 px-1.5 py-0.5 rounded-full font-medium">
            {filteredMessages.length}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {/* Agent Filter */}
          <select
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value as AgentType | 'all')}
            className="text-xs border border-surface-200 dark:border-surface-700 rounded-lg px-2 py-1 bg-white dark:bg-surface-900 text-surface-600 dark:text-surface-300 focus:outline-none focus:ring-2 focus:ring-accent-500/20 focus:border-accent-400"
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
              className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
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
        className="flex-1 overflow-y-auto p-3 space-y-2 bg-surface-50 dark:bg-surface-950"
      >
        {/* Collapsed older messages */}
        {collapsedMessages.length > 0 && (
          <div className="mb-2">
            <button
              onClick={() => setShowCollapsed(!showCollapsed)}
              className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-surface-200 dark:bg-surface-700 hover:bg-surface-300 dark:hover:bg-surface-600 rounded-lg text-sm text-surface-600 dark:text-surface-300 transition-colors"
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
