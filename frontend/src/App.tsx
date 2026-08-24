import { lazy, Suspense, useState, useEffect, useRef, useCallback } from 'react'
import { useAppStore } from './stores/appStore'
import { useConfig, useWebSocket, useUserConfig } from './hooks/useApi'
import { useTranslation } from './i18n/translations'
import { devWarn } from './utils/logger'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Sidebar } from './components/Sidebar'
import { SidebarResizeHandle } from './components/SidebarResizeHandle'
import { LanguageSelector } from './components/LanguageSelector'
import { ScAIIcon } from './components/ScAIIcon'
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
  PanelLeftClose,
  PanelLeftOpen,
  AlertTriangle
} from 'lucide-react'

const ConfigPanel = lazy(() => import('./components/ConfigPanel').then(module => ({ default: module.ConfigPanel })))
const AnalysisPanel = lazy(() => import('./components/AnalysisPanel').then(module => ({ default: module.AnalysisPanel })))
const DataPanel = lazy(() => import('./components/DataPanel').then(module => ({ default: module.DataPanel })))
const PromptsPanel = lazy(() => import('./components/PromptsPanel').then(module => ({ default: module.PromptsPanel })))
const ResultsPanel = lazy(() => import('./components/ResultsPanel').then(module => ({ default: module.ResultsPanel })))
const ReportPanel = lazy(() => import('./components/ReportPanel').then(module => ({ default: module.ReportPanel })))
const ChatPanel = lazy(() => import('./components/ChatPanel').then(module => ({ default: module.ChatPanel })))
const MessageStream = lazy(() => import('./components/MessageStream').then(module => ({ default: module.MessageStream })))

function PanelFallback() {
  return <div className="flex h-full items-center justify-center text-sm text-surface-400">Loading…</div>
}

function App() {
  const { activeTab, setActiveTab, sidebarOpen, setSidebarOpen, sidebarWidth, setSidebarWidth, currentSession, language, wsConnected, messagePanelWidth, setMessagePanelWidth, setShowConfigPanelCallback, setWsCallbacks, authUser, sessionLoading } = useAppStore()
  const { checkStatus } = useConfig()
  const { getUserConfigStatus, getUserConfig } = useUserConfig()
  const { connect, disconnect } = useWebSocket()
  const [configured, setConfigured] = useState<boolean | null>(null)
  const [showConfig, setShowConfig] = useState(false)
  const { t } = useTranslation(language)

  // Resizable panel state
  const [isResizing, setIsResizing] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  // Load a tab only when first visited, then keep it mounted so switching tabs
  // preserves its local UI state without forcing every large panel into the
  // initial production bundle.
  const mountedTabsRef = useRef(new Set([activeTab]))
  mountedTabsRef.current.add(activeTab)

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

  // Fetch app version from backend on startup
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/api', { method: 'GET' })
        if (res.ok) {
          const data = await res.json()
          if (data.version) useAppStore.getState().setAppVersion(data.version)
        }
      } catch (err) {
        devWarn('Failed to fetch app version', err)
      }
    })()
  }, [])

  // Check API config status based on user role; sync config to store
  useEffect(() => {
    const run = async () => {
      const { apiConfig, setApiConfig } = useAppStore.getState()
      if (authUser?.role === 'admin') {
        const s = await checkStatus()
        if (!s.configured) setShowConfig(true)
        setConfigured(s.configured)
        useAppStore.getState().setApiConfigured(s.configured)
        // Sync admin config from store to backend if store has values
        if (s.configured && !apiConfig.apiKey) {
          // Admin has config on server but store is empty (e.g. cleared cache)
          // User will be prompted to re-enter key when needed
        }
      } else if (authUser) {
        const s = await getUserConfigStatus()
        if (!s.configured) setShowConfig(true)
        setConfigured(s.configured)
        useAppStore.getState().setApiConfigured(s.configured)
        // Sync non-admin user's llmBaseUrl from backend if store has defaults
        if (s.configured && !apiConfig.apiKey) {
          try {
            const uc = await getUserConfig()
            if (uc.llm_base_url) {
              setApiConfig({ llmBaseUrl: uc.llm_base_url })
            }
          } catch { /* ignore — store will use defaults */ }
        }
      }
    }
    run()
  }, [authUser, checkStatus, getUserConfigStatus, getUserConfig])

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

  if (configured === null) {
    return (
      <div className="min-h-screen bg-surface-50 dark:bg-surface-950 flex flex-col items-center justify-center gap-4">
        <ScAIIcon className="w-16 h-16 animate-pulse" />
        <p className="text-surface-500 dark:text-surface-400 text-sm font-medium tracking-wide">Loading ScAI...</p>
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
    <ErrorBoundary>
    <div className="min-h-screen bg-surface-50 dark:bg-surface-950">
      {/* Session loading overlay — shown while loadSession fetches, auto-closes on completion */}
      {sessionLoading && (
        <div className="fixed inset-0 z-[60] flex flex-col items-center justify-center gap-4 bg-surface-900/70 backdrop-blur-sm">
          <ScAIIcon className="w-16 h-16 animate-pulse" />
          <p className="text-white text-sm font-medium tracking-wide">{t('loadingAnalysis')}</p>
        </div>
      )}

      {/* Config Modal */}
      {showConfig && (
        <Suspense fallback={<PanelFallback />}>
          <ConfigPanel
            onClose={() => {
              setShowConfig(false)
            }}
            onSave={() => {
              setConfigured(true)
              setShowConfig(false)
              useAppStore.getState().setReadMode(false)
            }}
            onReadMode={!configured ? () => {
              setShowConfig(false)
              useAppStore.getState().setReadMode(true)
              setTimeout(() => {
                useAppStore.getState().openSessionBrowser?.()
              }, 100)
            } : undefined}
            isAdmin={authUser?.role === 'admin'}
          />
        </Suspense>
      )}

      <div className="flex h-screen overflow-hidden">
        {/* Sidebar */}
        <Sidebar 
          isOpen={sidebarOpen} 
          onOpenConfig={() => setShowConfig(true)}
          width={sidebarWidth}
        />
        {sidebarOpen && (
          <SidebarResizeHandle
            width={sidebarWidth}
            onWidthChange={setSidebarWidth}
            label={t('resizeSidebar')}
          />
        )}

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <header className="h-14 border-b border-surface-200/60 dark:border-surface-700/60 bg-white/90 dark:bg-surface-900/90 backdrop-blur-md flex items-center justify-between px-4 shrink-0" style={{ boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.04)' }}>
            <div className="flex items-center gap-4">
              <button
                type="button"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg transition-colors"
                aria-label={sidebarOpen ? t('collapseSidebar') : t('expandSidebar')}
                aria-controls="app-sidebar"
                aria-expanded={sidebarOpen}
                title={sidebarOpen ? t('collapseSidebar') : t('expandSidebar')}
              >
                <span className="lg:hidden">
                  {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                </span>
                <span className="hidden lg:block">
                  {sidebarOpen ? <PanelLeftClose className="w-5 h-5" /> : <PanelLeftOpen className="w-5 h-5" />}
                </span>
              </button>

              <div>
                <h1 className="font-display font-semibold text-surface-800 dark:text-surface-100 tracking-tight">
                  {currentSession?.name || t('appFullName')}
                </h1>
                {currentSession && (
                  <div className="flex items-center gap-2 text-xs text-surface-500 dark:text-surface-400">
                    <span>Round {currentSession.currentRound}</span>
                    <span className="text-surface-300 dark:text-surface-600">·</span>
                    <span className={`status-${currentSession.status}`}>
                      {currentSession.status}
                    </span>
                    {wsConnected ? (
                      <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                        <Wifi className="w-3 h-3" />
                        <span>{language === 'zh' ? '已连接' : 'Connected'}</span>
                      </span>
                    ) : currentSession.status === 'running' ? (
                      <span className="flex items-center gap-1 text-amber-600 dark:text-amber-400">
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
                        ? 'text-accent-600 dark:text-accent-400'
                        : 'text-surface-500 dark:text-surface-400 hover:text-surface-700 dark:hover:text-surface-200'
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
            </div>
          </header>

          {/* Read-only banner */}
          {authUser?.status === 'readonly' && (
            <div className="bg-amber-50 dark:bg-amber-950/40 border-b border-amber-200 dark:border-amber-800 px-4 py-2 text-sm text-amber-800 dark:text-amber-200 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Read-only mode — you can view analysis results but cannot create or modify sessions.
            </div>
          )}

          {/* Content Area */}
          <div ref={containerRef} className="flex-1 flex overflow-hidden bg-science-grid">
            {/* Left Panel - Tab Content */}
            <div
              className="flex flex-col"
              style={{ width: currentSession ? `${100 - messagePanelWidth}%` : '100%' }}
            >
              <div className="flex-1 overflow-auto p-6" style={{ display: activeTab === 'analysis' ? 'block' : 'none' }}>
                {mountedTabsRef.current.has('analysis') && <Suspense fallback={<PanelFallback />}><AnalysisPanel /></Suspense>}
              </div>
              <div className="flex-1 overflow-auto p-6" style={{ display: activeTab === 'data' ? 'block' : 'none' }}>
                {mountedTabsRef.current.has('data') && <Suspense fallback={<PanelFallback />}><DataPanel /></Suspense>}
              </div>
              <div className="flex-1 overflow-auto p-6" style={{ display: activeTab === 'prompts' ? 'block' : 'none' }}>
                {mountedTabsRef.current.has('prompts') && <Suspense fallback={<PanelFallback />}><PromptsPanel /></Suspense>}
              </div>
              <div className="flex-1 overflow-auto p-6" style={{ display: activeTab === 'results' ? 'block' : 'none' }}>
                {mountedTabsRef.current.has('results') && <Suspense fallback={<PanelFallback />}><ResultsPanel /></Suspense>}
              </div>
              <div className="flex-1 flex flex-col min-h-0" style={{ display: activeTab === 'report' ? 'flex' : 'none' }}>
                {mountedTabsRef.current.has('report') && <Suspense fallback={<PanelFallback />}><ReportPanel /></Suspense>}
              </div>
              <div className="flex-1 flex flex-col min-h-0" style={{ display: activeTab === 'chat' ? 'flex' : 'none' }}>
                {mountedTabsRef.current.has('chat') && <Suspense fallback={<PanelFallback />}><ChatPanel /></Suspense>}
              </div>
            </div>

            {/* Resizable Divider */}
            {currentSession && (
              <div
                className="hidden lg:flex items-center justify-center w-[3px] bg-surface-200 dark:bg-surface-700 hover:bg-accent-400 cursor-col-resize transition-colors duration-200 group"
                onMouseDown={() => setIsResizing(true)}
              >
                <div className="absolute p-1 bg-surface-300 dark:bg-surface-600 group-hover:bg-accent-500 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                  <GripVertical className="w-3 h-3 text-white" />
                </div>
              </div>
            )}

            {/* Right Panel - Message Stream */}
            {currentSession && (
              <div
                className="hidden lg:flex border-l border-surface-200/60 dark:border-surface-700/60 bg-white dark:bg-surface-900 overflow-hidden flex-col"
                style={{ width: `${messagePanelWidth}%` }}
              >
                <Suspense fallback={<PanelFallback />}><MessageStream /></Suspense>
              </div>
            )}
          </div>

          {/* Mobile Tabs */}
          <nav className="md:hidden border-t border-surface-200 dark:border-surface-700 bg-white dark:bg-surface-900 flex">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex flex-col items-center gap-1 py-3 text-xs font-medium transition-all ${
                  activeTab === tab.id
                    ? 'text-accent-600 dark:text-accent-400'
                    : 'text-surface-500 dark:text-surface-400'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Footer */}
          <footer className="shrink-0 border-t border-surface-100 dark:border-surface-800 bg-white/60 dark:bg-surface-900/60 backdrop-blur-sm px-4 py-1.5 text-center">
            <p className="text-[10px] text-surface-400 select-none tracking-wider uppercase">
              {language === 'zh'
                ? 'ScAI · 单细胞测序AI分析 · © 2025–2026 程敏章 版权所有'
                : 'ScAI · Single Cell Sequencing Analysis Driven by AI · © 2025–2026 Minzhang Cheng. All rights reserved.'}
            </p>
          </footer>
        </div>
      </div>
    </div>
    </ErrorBoundary>
  )
}

export default App
