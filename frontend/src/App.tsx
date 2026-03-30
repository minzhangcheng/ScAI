import { useState, useEffect, useRef, useCallback } from 'react'
import { useAppStore } from './stores/appStore'
import { useConfig, useWebSocket } from './hooks/useApi'
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
  GripVertical,
  Dna
} from 'lucide-react'

function App() {
  const { activeTab, setActiveTab, sidebarOpen, setSidebarOpen, currentSession, language, wsConnected, messagePanelWidth, setMessagePanelWidth, setShowConfigPanelCallback, setWsCallbacks } = useAppStore()
  const { checkStatus, saveConfig } = useConfig()
  const { connect, disconnect } = useWebSocket()
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

  // Note: WebSocket auto-reconnect is handled entirely by useWebSocket hook.
  // No duplicate reconnection logic here to avoid competing reconnect attempts.

  if (showDisclaimer) {
    return <DisclaimerDialog onAccept={() => setShowDisclaimer(false)} />
  }

  if (configured === null) {
    return (
      <div className="min-h-screen bg-surface-900 flex flex-col items-center justify-center gap-4">
        <div className="p-4 bg-gradient-to-br from-accent-500 to-accent-700 rounded-2xl shadow-glow-accent">
          <Dna className="w-8 h-8 text-white animate-spin" />
        </div>
        <p className="text-surface-400 text-sm font-medium tracking-wide">Loading ScAI...</p>
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
    <div className="min-h-screen bg-surface-50">
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
          <header className="h-14 border-b border-surface-200/60 bg-white/90 backdrop-blur-md flex items-center justify-between px-4 shrink-0" style={{ boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.04)' }}>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 hover:bg-surface-100 rounded-lg transition-colors lg:hidden"
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>

              <div>
                <h1 className="font-display font-semibold text-surface-800 tracking-tight">
                  {currentSession?.name || t('appFullName')}
                </h1>
                {currentSession && (
                  <div className="flex items-center gap-2 text-xs text-surface-500">
                    <span>Round {currentSession.currentRound}</span>
                    <span className="text-surface-300">·</span>
                    <span className={`status-${currentSession.status}`}>
                      {currentSession.status}
                    </span>
                    {wsConnected ? (
                      <span className="flex items-center gap-1 text-emerald-600">
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
              {/* Tabs - underline style */}
              <nav className="hidden md:flex items-center gap-0.5">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`relative flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-all duration-200 ${
                      activeTab === tab.id
                        ? 'text-accent-600'
                        : 'text-surface-500 hover:text-surface-700'
                    }`}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                    {activeTab === tab.id && (
                      <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-accent-500 rounded-full animate-tab-indicator" />
                    )}
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
          <div ref={containerRef} className="flex-1 flex overflow-hidden bg-science-grid">
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
                className="hidden lg:flex items-center justify-center w-[3px] bg-surface-200 hover:bg-accent-400 cursor-col-resize transition-colors duration-200 group"
                onMouseDown={() => setIsResizing(true)}
              >
                <div className="absolute p-1 bg-surface-300 group-hover:bg-accent-500 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                  <GripVertical className="w-3 h-3 text-white" />
                </div>
              </div>
            )}

            {/* Right Panel - Message Stream */}
            {currentSession && (
              <div
                className="hidden lg:flex border-l border-surface-200/60 bg-white overflow-hidden flex-col"
                style={{ width: `${messagePanelWidth}%` }}
              >
                <MessageStream />
              </div>
            )}
          </div>

          {/* Mobile Tabs */}
          <nav className="md:hidden border-t border-surface-200 bg-white flex">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex flex-col items-center gap-1 py-3 text-xs font-medium transition-all ${
                  activeTab === tab.id
                    ? 'text-accent-600'
                    : 'text-surface-500'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Footer */}
          <footer className="shrink-0 border-t border-surface-100 bg-white/60 backdrop-blur-sm px-4 py-1.5 text-center">
            <p className="text-[10px] text-surface-400 select-none tracking-wider uppercase">
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
