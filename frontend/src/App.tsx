import { useState, useEffect, useRef, useCallback } from 'react'
import { useAppStore } from './stores/appStore'
import { useConfig, useWebSocket } from './hooks/useApi'
import { devLog } from './utils/logger'
import { useTranslation } from './i18n/translations'
import { ConfigPanel } from './components/ConfigPanel'
import { DisclaimerDialog, hasAcceptedDisclaimer } from './components/DisclaimerDialog'
import { Sidebar } from './components/Sidebar'
import { AnalysisPanel } from './components/AnalysisPanel'
import { DataPanel } from './components/DataPanel'
import { PromptsPanel } from './components/PromptsPanel'
import { ResultsPanel } from './components/ResultsPanel'
import { ReportPanel } from './components/ReportPanel'
import { ChatPanel } from './components/ChatPanel'
import { MessageStream } from './components/MessageStream'
import { LanguageSelector } from './components/LanguageSelector'
import { AboutDialog } from './components/AboutDialog'
import {
  Beaker,
  FileText,
  Settings,
  BarChart3,
  FileBarChart,
  MessageCircle,
  Menu,
  X,
  Wifi,
  WifiOff,
  GripVertical
} from 'lucide-react'

function App() {
  const { activeTab, setActiveTab, sidebarOpen, setSidebarOpen, currentSession, language, wsConnected, messagePanelWidth, setMessagePanelWidth, setShowConfigPanelCallback, setWsCallbacks, chatDelegationActive } = useAppStore()
  const { checkStatus, saveConfig } = useConfig()
  const { connect, disconnect, connected } = useWebSocket()
  const [configured, setConfigured] = useState<boolean | null>(null)
  const [showConfig, setShowConfig] = useState(false)
  const [configSkipped, setConfigSkipped] = useState(false)
  const [showDisclaimer, setShowDisclaimer] = useState(!hasAcceptedDisclaimer())
  const { t } = useTranslation(language)

  // Resizable panel state
  const [isResizing, setIsResizing] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Handle mouse move for resizing
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing || !containerRef.current) return

    const containerRect = containerRef.current.getBoundingClientRect()
    const newWidth = ((containerRect.right - e.clientX) / containerRect.width) * 100

    // Clamp between 20% and 80%
    setMessagePanelWidth(Math.max(20, Math.min(80, newWidth)))
  }, [isResizing, setMessagePanelWidth])

  // Handle mouse up to stop resizing
  const handleMouseUp = useCallback(() => {
    setIsResizing(false)
  }, [])

  // Add/remove event listeners for resizing
  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    } else {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing, handleMouseMove, handleMouseUp])

  // Check API config status; auto-sync from localStorage if backend lost config
  useEffect(() => {
    checkStatus().then(async (status) => {
      if (!status.configured) {
        const { apiConfig } = useAppStore.getState()
        if (apiConfig.apiKey) {
          // Frontend has saved config but backend doesn't — auto-sync
          try {
            await saveConfig()
            setConfigured(true)
            return
          } catch (e) {
            console.warn('[ScAI] Auto-sync config on startup failed:', e)
          }
        }
        setShowConfig(true)
      }
      setConfigured(status.configured)
    })
  }, [checkStatus, saveConfig])

  // Register the showConfigPanel callback for other components (e.g., AnalysisPanel)
  useEffect(() => {
    setShowConfigPanelCallback(() => setShowConfig(true))
    return () => setShowConfigPanelCallback(null)
  }, [setShowConfigPanelCallback])

  // Register WebSocket connect/disconnect callbacks for child components (e.g., AnalysisPanel)
  useEffect(() => {
    setWsCallbacks(connect, disconnect)
    return () => setWsCallbacks(null, null)
  }, [connect, disconnect, setWsCallbacks])

  // Auto-reconnect WebSocket if session is running (with exponential backoff)
  const reconnectAttemptRef = useRef(0)
  const maxReconnectDelay = 30000 // Max 30 seconds between attempts

  useEffect(() => {
    if (currentSession && (currentSession.status === 'running' || chatDelegationActive) && !connected && !wsConnected) {
      // Calculate delay with exponential backoff: 1s, 2s, 4s, 8s... up to maxReconnectDelay
      const baseDelay = 1000
      const delay = Math.min(baseDelay * Math.pow(2, reconnectAttemptRef.current), maxReconnectDelay)

      const timer = setTimeout(() => {
        devLog(`App: Session is running but not connected, attempting reconnect (attempt ${reconnectAttemptRef.current + 1}, delay ${delay}ms)...`)
        reconnectAttemptRef.current += 1
        connect()
      }, delay)
      return () => clearTimeout(timer)
    } else if (connected || wsConnected) {
      // Reset reconnect attempts on successful connection
      reconnectAttemptRef.current = 0
    }
  }, [currentSession?.id, currentSession?.status, connected, wsConnected, connect, chatDelegationActive])

  if (showDisclaimer) {
    return <DisclaimerDialog onAccept={() => setShowDisclaimer(false)} />
  }

  if (configured === null) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-violet-500 border-t-transparent" />
      </div>
    )
  }

  const tabs = [
    { id: 'data', label: t('tabData'), icon: FileText },
    { id: 'prompts', label: t('tabPrompts'), icon: Settings },
    { id: 'analysis', label: t('tabAnalysis'), icon: Beaker },
    { id: 'results', label: t('tabResults'), icon: BarChart3 },
    { id: 'report', label: t('tabReport'), icon: FileBarChart },
    { id: 'chat', label: t('tabChat'), icon: MessageCircle },
  ] as const

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {/* Config Modal */}
      {showConfig && (
        <ConfigPanel
          onClose={() => {
            if (configured || configSkipped) setShowConfig(false)
          }}
          onSave={() => {
            setConfigured(true)
            setConfigSkipped(false)
            setShowConfig(false)
            useAppStore.getState().setReadMode(false)
          }}
          onReadMode={!configured ? () => {
            setConfigSkipped(true)
            setShowConfig(false)
            useAppStore.getState().setReadMode(true)
            setTimeout(() => {
              useAppStore.getState().openSessionBrowser?.()
            }, 100)
          } : undefined}
        />
      )}

      <div className="flex h-screen overflow-hidden">
        {/* Sidebar */}
        <Sidebar 
          isOpen={sidebarOpen} 
          onOpenConfig={() => setShowConfig(true)}
        />

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <header className="h-16 border-b border-slate-200 bg-white/80 backdrop-blur-sm flex items-center justify-between px-4 shrink-0">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors lg:hidden"
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              
              <div>
                <h1 className="font-semibold text-slate-900">
                  {currentSession?.name || t('appFullName')}
                </h1>
                {currentSession && (
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span>Round {currentSession.currentRound}</span>
                    <span>·</span>
                    <span className={`status-${currentSession.status}`}>
                      {currentSession.status}
                    </span>
                    {wsConnected ? (
                      <span className="flex items-center gap-1 text-green-600">
                        <Wifi className="w-3 h-3" />
                        <span>{language === 'zh' ? '已连接' : 'Connected'}</span>
                      </span>
                    ) : currentSession.status === 'running' ? (
                      <span className="flex items-center gap-1 text-amber-600">
                        <WifiOff className="w-3 h-3" />
                        <span>{language === 'zh' ? '断开' : 'Disconnected'}</span>
                      </span>
                    ) : null}
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Tabs */}
              <nav className="hidden md:flex items-center gap-1 bg-slate-100 rounded-lg p-1">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
                      activeTab === tab.id
                        ? 'bg-white text-violet-700 shadow-sm'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                ))}
              </nav>
              
              {/* Language Selector */}
              <LanguageSelector />

              {/* About */}
              <AboutDialog />
            </div>
          </header>

          {/* Content Area */}
          <div ref={containerRef} className="flex-1 flex overflow-hidden">
            {/* Left Panel - Tab Content */}
            <div
              className="flex flex-col"
              style={{ width: currentSession ? `${100 - messagePanelWidth}%` : '100%' }}
            >
              <div className="flex-1 overflow-auto p-6" style={{ display: activeTab === 'analysis' ? 'block' : 'none' }}>
                <AnalysisPanel />
              </div>
              <div className="flex-1 overflow-auto p-6" style={{ display: activeTab === 'data' ? 'block' : 'none' }}>
                <DataPanel />
              </div>
              <div className="flex-1 overflow-auto p-6" style={{ display: activeTab === 'prompts' ? 'block' : 'none' }}>
                <PromptsPanel />
              </div>
              <div className="flex-1 overflow-auto p-6" style={{ display: activeTab === 'results' ? 'block' : 'none' }}>
                <ResultsPanel />
              </div>
              <div className="flex-1 flex flex-col min-h-0" style={{ display: activeTab === 'report' ? 'flex' : 'none' }}>
                <ReportPanel />
              </div>
              <div className="flex-1 flex flex-col min-h-0" style={{ display: activeTab === 'chat' ? 'flex' : 'none' }}>
                <ChatPanel />
              </div>
            </div>

            {/* Resizable Divider */}
            {currentSession && (
              <div
                className="hidden lg:flex items-center justify-center w-1 bg-slate-200 hover:bg-violet-400 cursor-col-resize transition-colors group"
                onMouseDown={() => setIsResizing(true)}
              >
                <div className="absolute p-1 bg-slate-300 group-hover:bg-violet-500 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                  <GripVertical className="w-3 h-3 text-white" />
                </div>
              </div>
            )}

            {/* Right Panel - Message Stream */}
            {currentSession && (
              <div
                className="hidden lg:flex border-l border-slate-200 bg-white overflow-hidden flex-col"
                style={{ width: `${messagePanelWidth}%` }}
              >
                <MessageStream />
              </div>
            )}
          </div>

          {/* Mobile Tabs */}
          <nav className="md:hidden border-t border-slate-200 bg-white flex">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex flex-col items-center gap-1 py-3 text-xs font-medium transition-all ${
                  activeTab === tab.id
                    ? 'text-violet-700'
                    : 'text-slate-500'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Footer */}
          <footer className="shrink-0 border-t border-slate-100 bg-white/80 px-4 py-1.5 text-center">
            <p className="text-[11px] text-slate-400 select-none">
              {language === 'zh'
                ? 'ScAI · 单细胞测序AI分析 · © 2025–2026 程敏章 版权所有'
                : 'ScAI · Single Cell Sequencing Analysis Driven by AI · © 2025–2026 Minzhang Cheng. All rights reserved.'}
            </p>
          </footer>
        </div>
      </div>
    </div>
  )
}

export default App
