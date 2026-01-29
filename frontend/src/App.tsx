import { useState, useEffect } from 'react'
import { useAppStore } from './stores/appStore'
import { useConfig, useWebSocket } from './hooks/useApi'
import { useTranslation } from './i18n/translations'
import { ConfigPanel } from './components/ConfigPanel'
import { Sidebar } from './components/Sidebar'
import { AnalysisPanel } from './components/AnalysisPanel'
import { DataPanel } from './components/DataPanel'
import { PromptsPanel } from './components/PromptsPanel'
import { ResultsPanel } from './components/ResultsPanel'
import { ReportPanel } from './components/ReportPanel'
import { MessageStream } from './components/MessageStream'
import { LanguageSelector } from './components/LanguageSelector'
import { 
  Beaker, 
  FileText, 
  Settings, 
  BarChart3,
  Menu,
  X,
  Wifi,
  WifiOff,
  FileSpreadsheet
} from 'lucide-react'

function App() {
  const { activeTab, setActiveTab, sidebarOpen, setSidebarOpen, currentSession, language, wsConnected } = useAppStore()
  const { checkStatus } = useConfig()
  const { connect, connected } = useWebSocket()
  const [configured, setConfigured] = useState<boolean | null>(null)
  const [showConfig, setShowConfig] = useState(false)
  const { t } = useTranslation(language)

  // Check API config status
  useEffect(() => {
    checkStatus().then((status) => {
      setConfigured(status.configured)
      if (!status.configured) {
        setShowConfig(true)
      }
    })
  }, [checkStatus])

  // Auto-reconnect WebSocket if session is running (with debounce to prevent race conditions)
  useEffect(() => {
    if (currentSession && currentSession.status === 'running' && !connected && !wsConnected) {
      // Debounce to let initial connection complete
      const timer = setTimeout(() => {
        console.log('App: Session is running but not connected, attempting reconnect...')
        connect()
      }, 1000)
      return () => clearTimeout(timer)
    }
  }, [currentSession?.id, currentSession?.status, connected, wsConnected, connect])

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
    { id: 'report', label: language === 'zh' ? '报告' : 'Report', icon: FileSpreadsheet },
  ] as const

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {/* Config Modal */}
      {showConfig && (
        <ConfigPanel 
          onClose={() => {
            if (configured) setShowConfig(false)
          }}
          onSave={() => {
            setConfigured(true)
            setShowConfig(false)
          }}
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
            </div>
          </header>

          {/* Content Area */}
          <div className="flex-1 flex overflow-hidden">
            {/* Left Panel - Tab Content */}
            <div className={`${currentSession ? 'flex-1 lg:w-1/2' : 'flex-1'} overflow-auto p-6`}>
              {activeTab === 'analysis' && <AnalysisPanel />}
              {activeTab === 'data' && <DataPanel />}
              {activeTab === 'prompts' && <PromptsPanel />}
              {activeTab === 'results' && <ResultsPanel />}
              {activeTab === 'report' && <ReportPanel />}
            </div>

            {/* Right Panel - Message Stream */}
            {currentSession && (
              <div className="hidden lg:flex lg:w-1/2 border-l border-slate-200 bg-white overflow-hidden flex-col">
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
        </div>
      </div>
    </div>
  )
}

export default App
