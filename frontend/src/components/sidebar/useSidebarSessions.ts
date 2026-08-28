import { useCallback, useEffect, useRef, useState } from 'react'

import { apiFetch, useSession } from '../../hooks/useApi'
import { useTranslation } from '../../i18n/translations'
import { useAppStore } from '../../stores/appStore'
import { DEFAULT_SESSION_INSTRUCTION, DEFAULT_SESSION_NAME } from '../../constants/sessionDefaults'
import { getTargetTab, hasCompletedSession, selectPrioritySession } from '../../utils/sessionPriority'
import { useSidebarImport } from './useSidebarImport'
import type { SidebarProgressInfo } from './types'

export type { SidebarProgressInfo } from './types'

function syncTabToSessionStatus(session: { status: string }) {
  useAppStore.getState().setActiveTab(getTargetTab(session.status))
}

export function useSidebarSessions() {
  const { currentSession, setCurrentSession, clearMessages, language, roundWaitSeconds, setRoundWaitSeconds, reportGenerated, setOpenSessionBrowserCallback, authUser, _hydrated } = useAppStore()
  const { createSession, loadSession, listSessions, deleteSession, updateSessionInfo, updateRoundWait } = useSession()
  const isReadOnly = authUser?.status === 'readonly'
  const { t } = useTranslation(language)
  const [showNewSession, setShowNewSession] = useState(false)
  const [showSessionBrowser, setShowSessionBrowser] = useState(false)
  const [savedSessions, setSavedSessions] = useState<any[]>([])
  const [loadingSessions, setLoadingSessions] = useState(false)
  const [formKey, setFormKey] = useState(0)

  const [instruction, setInstruction] = useState(DEFAULT_SESSION_INSTRUCTION)
  const [sessionName, setSessionName] = useState(DEFAULT_SESSION_NAME)
  const [editingSession, setEditingSession] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [importing, setImporting] = useState(false)
  const [progressInfo, setProgressInfo] = useState<SidebarProgressInfo | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Startup session resolution is server-authoritative. A persisted completed
  // session can be very large, so it must never be reloaded merely because it
  // was the last session viewed in this browser.
  const hasResolvedInitialSession = useRef(false)
  const hasPrefetchedBrowserSessions = useRef(false)
  const loadSessionRef = useRef(loadSession)
  loadSessionRef.current = loadSession
  const setCurrentSessionRef = useRef(setCurrentSession)
  setCurrentSessionRef.current = setCurrentSession
  const clearMessagesRef = useRef(clearMessages)
  clearMessagesRef.current = clearMessages
  const listSessionsRef = useRef(listSessions)
  listSessionsRef.current = listSessions

  const openSessionBrowser = useCallback(() => {
    useAppStore.getState().setSidebarOpen(true)
    setShowSessionBrowser(true)
  }, [])

  useEffect(() => {
    if (!_hydrated || hasResolvedInitialSession.current) return
    hasResolvedInitialSession.current = true

    // Discard the lightweight persisted snapshot before making the startup
    // choice, so stale panels and sockets do not treat it as an opened session.
    setCurrentSessionRef.current(null)
    clearMessagesRef.current()

    ;(async () => {
      try {
        const sessions = await listSessionsRef.current()
        const runningSession = selectPrioritySession(sessions)
        if (runningSession) {
          const session = await loadSessionRef.current(runningSession.id)
          if (session) {
            useAppStore.getState().setActiveTab('analysis')
          }
          return
        }

        if (hasCompletedSession(sessions)) {
          setSavedSessions(sessions)
          setSessionListError(null)
          hasPrefetchedBrowserSessions.current = true
          openSessionBrowser()
        }
      } catch (e) {
        console.warn('Sidebar: Initial session resolution failed:', e)
      }
    })()
  }, [_hydrated, openSessionBrowser])

  // Register session browser callback for external components (e.g., ConfigPanel's Read Mode)
  useEffect(() => {
    setOpenSessionBrowserCallback(openSessionBrowser)
    return () => setOpenSessionBrowserCallback(null)
  }, [openSessionBrowser, setOpenSessionBrowserCallback])

  // Fetch sessions when browser is opened
  useEffect(() => {
    if (showSessionBrowser) {
      if (hasPrefetchedBrowserSessions.current) {
        hasPrefetchedBrowserSessions.current = false
        return
      }
      fetchSessions()
    }
  }, [showSessionBrowser])

  const { handleImport } = useSidebarImport({
    clearMessages,
    fileInputRef,
    importing,
    language,
    loadSession,
    setImporting,
    setProgressInfo,
    t,
  })
  const [sessionListError, setSessionListError] = useState<string | null>(null)

  const fetchSessions = async () => {
    setLoadingSessions(true)
    setSessionListError(null)
    try {
      const sessions = await listSessions()
      setSavedSessions(sessions)
    } catch (e: any) {
      console.error('Failed to fetch sessions:', e)
      setSessionListError(e?.message || 'Failed to load sessions')
    } finally {
      setLoadingSessions(false)
    }
  }

  const handleLoadSession = async (sessionId: string) => {
    try {
      clearMessages()
      const session = await loadSession(sessionId)
      if (session) {
        syncTabToSessionStatus(session)
      }
      setShowSessionBrowser(false)
    } catch (e) {
      console.error('Failed to load session:', e)
      alert(language === 'zh' ? '加载会话失败' : 'Failed to load session')
    }
  }

  const handleDeleteSavedSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm(language === 'zh' ? '确定要删除这个会话吗？' : 'Delete this session?')) {
      try {
        await deleteSession(sessionId)
        fetchSessions()
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        alert(msg)
      }
    }
  }

  const handleCreateSession = async () => {
    if (!instruction.trim()) return

    try {
      clearMessages()
      const session = await createSession(instruction, sessionName || undefined, roundWaitSeconds)
      if (session) {
        syncTabToSessionStatus(session)
      }
      setShowNewSession(false)
      // Reset to defaults for next time
      setInstruction(DEFAULT_SESSION_INSTRUCTION)
      setSessionName(DEFAULT_SESSION_NAME)
      setFormKey(prev => prev + 1)
    } catch (e) {
      console.error('Failed to create session:', e)
      const msg = e instanceof Error ? e.message : String(e)
      alert(msg)
    }
  }

  const handleClearSession = () => {
    if (confirm(language === 'zh' ? '确定要清除当前项目吗？' : 'Clear current project?')) {
      setCurrentSession(null)
      clearMessages()
    }
  }

  const handleRefreshSession = async () => {
    if (!currentSession) return

    try {
      const session = await loadSession(currentSession.id)
      if (session) syncTabToSessionStatus(session)
    } catch (e) {
      console.error('Failed to refresh session:', e)
    }
  }

  const handleExport = async () => {
    if (!currentSession || exporting) return

    // Opt-in for runtime kernel state. Off by default — the pickle is only
    // restorable on the same trusted instance and can add multi-GB per snapshot.
    const includeRuntime = window.confirm(
      language === 'zh'
        ? '是否包含内核运行时状态 (session.pkl)？\n\n选择「确定」会让归档包含已加载的 adata 等内存变量和工作流断点，导入后（包括导入到其他实例）可以接着分析下去，但归档体积会变大（可能多几 GB）。\n\n选择「取消」打包不带运行时状态：体积小，数据、图表、报告和历史都在，但导入后无法继续分析，只能从数据重跑。'
        : 'Include kernel runtime state (session.pkl)?\n\nClicking OK bundles the live in-memory variables (adata etc.) and the workflow checkpoint, so the analysis can carry on after import — including on a different instance. The archive can be GBs larger.\n\nClick Cancel for a smaller archive: data, figures, reports and history all travel, but the imported session cannot resume and must be re-run from the data.'
    )

    setExporting(true)

    // Phase 1: 服务端打包，显示 spinner
    setProgressInfo({
      action: t('exportPackaging'),
      detail: t('exportPackagingDetail'),
      closable: false
    })

    try {
      // 等待服务端完成打包（不下载文件，只等元数据）
      const meta = await apiFetch<{
        status: string
        size_bytes: number
        filename: string
        export_token: string
        download_url: string
      }>(
        `/session/${currentSession.id}/export/prepare${includeRuntime ? '?include_runtime=true' : ''}`,
        { method: 'POST' },
      )

      const sizeMB = (meta.size_bytes / 1024 / 1024).toFixed(1)

      // Phase 2: 打包完成，显示确认按钮，用户点击后开始流式下载
      setProgressInfo({
        action: language === 'zh' ? '打包完成' : 'Packaging Complete',
        detail: language === 'zh'
          ? `文件大小: ${sizeMB} MB，点击确认开始下载`
          : `File size: ${sizeMB} MB. Click confirm to start download.`,
        closable: true,
        success: true,
        onConfirm: () => {
          // Phase 3: 触发浏览器流式下载（边下边存，不占内存）
          // The <a download> navigation carries no Authorization header, so the
          // one-shot token from /export/prepare is what authorizes it. Without a
          // token the endpoint now refuses -- the archive holds every upload,
          // figure, report and message of the session.
          const link = document.createElement('a')
          link.href = meta.download_url
            || `/api/session/${currentSession.id}/export/download?token=${encodeURIComponent(meta.export_token)}`
          link.download = meta.filename
          document.body.appendChild(link)
          link.click()
          document.body.removeChild(link)

          setProgressInfo({
            action: t('exportDownloadStarted'),
            detail: t('exportDownloadStartedDetail'),
            closable: true,
            success: true
          })
        }
      })
    } catch (err) {
      setProgressInfo({
        action: language === 'zh' ? '导出失败' : 'Export Failed',
        detail: String(err),
        closable: true
      })
    } finally {
      setExporting(false)
    }
  }


  return {
    currentSession, setCurrentSession, language, roundWaitSeconds, setRoundWaitSeconds, reportGenerated,
    authUser, isReadOnly, t, openSessionBrowser, showNewSession, setShowNewSession, showSessionBrowser,
    setShowSessionBrowser, savedSessions, loadingSessions, formKey, instruction,
    setInstruction, sessionName, setSessionName, editingSession, setEditingSession,
    exporting, importing, progressInfo, setProgressInfo, fileInputRef, sessionListError,
    fetchSessions, handleLoadSession, handleDeleteSavedSession, handleCreateSession,
    handleClearSession, handleRefreshSession, handleExport, handleImport, loadSession, updateSessionInfo,
    updateRoundWait,
  }
}

export type SidebarController = ReturnType<typeof useSidebarSessions>
