import { useCallback, useEffect, useRef, useState } from 'react'

import { AppLayout } from './components/app/AppLayout'
import { ScAIIcon } from './components/ScAIIcon'
import { useConfig, useUserConfig, useWebSocket } from './hooks/useApi'
import { useTranslation } from './i18n/translations'
import { useAppStore } from './stores/appStore'
import { devWarn } from './utils/logger'

function App() {
  const {
    activeTab, authUser, currentSession, language, messagePanelWidth,
    sessionLoading, setActiveTab, setMessagePanelWidth, setShowConfigPanelCallback,
    setSidebarOpen, setSidebarWidth, setWsCallbacks, sidebarOpen, sidebarWidth,
    wsConnected,
  } = useAppStore()
  const { checkStatus } = useConfig()
  const { getUserConfigStatus, getUserConfig } = useUserConfig()
  const { connect, disconnect } = useWebSocket()
  const [configured, setConfigured] = useState<boolean | null>(null)
  const [showConfig, setShowConfig] = useState(false)
  const { t } = useTranslation(language)
  const [isResizing, setIsResizing] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const mountedTabsRef = useRef(new Set([activeTab]))
  mountedTabsRef.current.add(activeTab)

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing || !containerRef.current) return
    const containerRect = containerRef.current.getBoundingClientRect()
    const newWidth = ((containerRect.right - e.clientX) / containerRect.width) * 100
    setMessagePanelWidth(Math.max(20, Math.min(80, newWidth)))
  }, [isResizing, setMessagePanelWidth])

  const handleMouseUp = useCallback(() => setIsResizing(false), [])

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

  useEffect(() => {
    const run = async () => {
      const { apiConfig, setApiConfig } = useAppStore.getState()
      if (authUser?.role === 'admin') {
        const s = await checkStatus()
        if (!s.configured) setShowConfig(true)
        setConfigured(s.configured)
        useAppStore.getState().setApiConfigured(s.configured)
      } else if (authUser) {
        const s = await getUserConfigStatus()
        if (!s.configured) setShowConfig(true)
        setConfigured(s.configured)
        useAppStore.getState().setApiConfigured(s.configured)
        if (s.configured && !apiConfig.apiKey) {
          try {
            const uc = await getUserConfig()
            if (uc.llm_base_url) setApiConfig({ llmBaseUrl: uc.llm_base_url })
          } catch { /* ignore — store will use defaults */ }
        }
      }
    }
    run()
  }, [authUser, checkStatus, getUserConfigStatus, getUserConfig])

  useEffect(() => {
    setShowConfigPanelCallback(() => setShowConfig(true))
    return () => setShowConfigPanelCallback(null)
  }, [setShowConfigPanelCallback])

  useEffect(() => {
    setWsCallbacks(connect, disconnect)
    return () => setWsCallbacks(null, null)
  }, [connect, disconnect, setWsCallbacks])

  if (configured === null) {
    return (
      <div className="min-h-screen bg-surface-50 dark:bg-surface-950 flex flex-col items-center justify-center gap-4">
        <ScAIIcon className="w-16 h-16 animate-pulse" />
        <p className="text-surface-500 dark:text-surface-400 text-sm font-medium tracking-wide">Loading ScAI...</p>
      </div>
    )
  }

  return (
    <AppLayout
      activeTab={activeTab}
      authUser={authUser}
      configured={configured}
      containerRef={containerRef}
      currentSession={currentSession}
      language={language}
      messagePanelWidth={messagePanelWidth}
      mountedTabsRef={mountedTabsRef}
      sessionLoading={sessionLoading}
      setActiveTab={setActiveTab}
      setConfigured={setConfigured}
      setIsResizing={setIsResizing}
      setShowConfig={setShowConfig}
      setSidebarOpen={setSidebarOpen}
      setSidebarWidth={setSidebarWidth}
      showConfig={showConfig}
      sidebarOpen={sidebarOpen}
      sidebarWidth={sidebarWidth}
      t={t}
      wsConnected={wsConnected}
    />
  )
}

export default App
