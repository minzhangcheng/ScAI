import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSession, apiFetch, generateId, authHeaders } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { getTargetTab, selectPrioritySession, selectCompletedSession, selectPendingSession } from '../utils/sessionPriority'
import type { SessionListItem } from '../types'
import { useTranslation } from '../i18n/translations'
import { ROUND_WAIT_OPTIONS } from '../constants/roundWait'
import { DataConfigMenu } from './DataConfigMenu'
import { UserMenu } from './UserMenu'
import { AboutDialog } from './AboutDialog'
import {
  Plus,
  Settings,
  Shield,
  Dna,
  Trash2,
  RefreshCw,
  FolderOpen,
  Clock,
  X,
  Loader2,
  CheckCircle,
  AlertCircle,
  Pencil
} from 'lucide-react'

const PRIORITY_TIERS: { select: (s: SessionListItem[]) => SessionListItem | null; tab: 'data' | 'analysis' | 'report' }[] = [
  { select: selectPrioritySession, tab: 'analysis' },
  { select: selectCompletedSession, tab: 'report' },
  { select: selectPendingSession, tab: 'data' },
]

function syncTabToSessionStatus(session: { status: string }) {
  useAppStore.getState().setActiveTab(getTargetTab(session.status))
}

interface SidebarProps {
  isOpen: boolean
  onOpenConfig: () => void
}

export function Sidebar({ isOpen, onOpenConfig }: SidebarProps) {
  const navigate = useNavigate()
  const { currentSession, setCurrentSession, clearMessages, language, roundWaitSeconds, setRoundWaitSeconds, reportGenerated, setOpenSessionBrowserCallback, authUser, _hydrated } = useAppStore()
  const { createSession, loadSession, listSessions, deleteSession, updateSessionInfo, updateRoundWait } = useSession()
  const isReadOnly = authUser?.status === 'readonly'
  const { t } = useTranslation(language)
  const [showNewSession, setShowNewSession] = useState(false)
  const [showSessionBrowser, setShowSessionBrowser] = useState(false)
  const [savedSessions, setSavedSessions] = useState<any[]>([])
  const [loadingSessions, setLoadingSessions] = useState(false)
  
  const [instruction, setInstruction] = useState(t('defaultProjectDescription'))
  const [sessionName, setSessionName] = useState(t('defaultProjectName'))
  const [editingSession, setEditingSession] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [importing, setImporting] = useState(false)
  const [progressInfo, setProgressInfo] = useState<{ action: string; detail: string; percent?: number; closable?: boolean; success?: boolean; onConfirm?: () => void; onCancel?: () => void; cancelLabel?: string } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Auto-reload session from API on page refresh to restore full data (images etc.)
  // Uses currentSession?.id as dep so it fires after Zustand hydration restores the session.
  // Other deps (loadSession, setCurrentSession, clearMessages) are intentionally omitted:
  // the hasAutoReloaded ref guard ensures this runs at most once per mount, and including
  // the hook-returned functions would cause unnecessary re-fires since their identity
  // changes on every render.
  const hasAutoReloaded = useRef(false)
  const hasPriorityLoaded = useRef(false)
  const loadSessionRef = useRef(loadSession)
  loadSessionRef.current = loadSession
  const setCurrentSessionRef = useRef(setCurrentSession)
  setCurrentSessionRef.current = setCurrentSession
  const clearMessagesRef = useRef(clearMessages)
  clearMessagesRef.current = clearMessages
  const listSessionsRef = useRef(listSessions)
  listSessionsRef.current = listSessions

  const doPriorityLoad = useCallback(async () => {
    if (hasPriorityLoaded.current) return
    hasPriorityLoaded.current = true
    try {
      const sessions = await listSessionsRef.current()
      for (const { select } of PRIORITY_TIERS) {
        const candidate = select(sessions)
        if (!candidate) continue
        const session = await loadSessionRef.current(candidate.id)
        if (session) {
          hasAutoReloaded.current = true
          syncTabToSessionStatus(session)
        }
        return
      }
    } catch (e) {
      console.warn('Sidebar: Priority session loading failed:', e)
    }
  }, [])

  useEffect(() => {
    if (currentSession?.id && !hasAutoReloaded.current) {
      hasAutoReloaded.current = true
      loadSessionRef.current(currentSession.id).then(session => {
        if (session) {
          syncTabToSessionStatus(session)
        }
      }).catch(e => {
        console.warn('Sidebar: Auto-reload session failed, clearing stale session:', e)
        setCurrentSessionRef.current(null)
        clearMessagesRef.current()
        // Session no longer exists on server — fall through to priority loading
        hasAutoReloaded.current = false
        hasPriorityLoaded.current = false
        doPriorityLoad()
      })
    }
  }, [currentSession?.id, doPriorityLoad])

  // Priority-based session loading: when no persisted session exists, auto-load
  // the best candidate from the server after Zustand hydration completes.
  useEffect(() => {
    if (!_hydrated || hasPriorityLoaded.current) return
    if (hasAutoReloaded.current || currentSession?.id) {
      hasPriorityLoaded.current = true
      return
    }
    doPriorityLoad()
  }, [_hydrated, doPriorityLoad])

  // Register session browser callback for external components (e.g., ConfigPanel's Read Mode)
  useEffect(() => {
    setOpenSessionBrowserCallback(() => setShowSessionBrowser(true))
    return () => setOpenSessionBrowserCallback(null)
  }, [setOpenSessionBrowserCallback])

  // Fetch sessions when browser is opened
  useEffect(() => {
    if (showSessionBrowser) {
      fetchSessions()
    }
  }, [showSessionBrowser])

  // Auto-resume polling for an in-progress import after page refresh.
  // The import job lives in backend memory (sessions/_import_jobs) and survives
  // a frontend refresh; without this hook the user loses visibility into the
  // job after reloading. Mount-only effect; guarded by ref to avoid double-fire.
  const resumedImportRef = useRef(false)
  useEffect(() => {
    if (resumedImportRef.current) return
    resumedImportRef.current = true
    let raw: string | null = null
    try { raw = localStorage.getItem('scai.pendingImport') } catch { return }
    if (!raw) return
    let saved: { importId?: string } | null = null
    try { saved = JSON.parse(raw) } catch { saved = null }
    const pendingId = saved?.importId
    if (!pendingId) {
      try { localStorage.removeItem('scai.pendingImport') } catch {}
      return
    }
    // Probe once before showing the modal — the job may have finished or expired.
    ;(async () => {
      let info: any
      try {
        info = await apiFetch<any>(`/session/import/${pendingId}`)
      } catch {
        try { localStorage.removeItem('scai.pendingImport') } catch {}
        return
      }
      const terminal = info?.status === 'completed' || info?.status === 'failed' || info?.status === 'cancelled'
      if (terminal) {
        try { localStorage.removeItem('scai.pendingImport') } catch {}
        return
      }
      // Active job — re-enter the polling loop and surface the modal.
      setImporting(true)
      let cancelled = false
      try {
        const sessionId = await pollImportUntilDone(pendingId, () => cancelled)
        if (sessionId) {
          clearMessages()
          await loadSession(sessionId)
        }
        try { localStorage.removeItem('scai.pendingImport') } catch {}
        setProgressInfo({
          action: t('importSuccess'),
          detail: language === 'zh' ? '会话已成功导入' : 'Session imported successfully',
          closable: true,
          success: true,
        })
      } catch (err: any) {
        try { localStorage.removeItem('scai.pendingImport') } catch {}
        setProgressInfo({
          action: t('importFailed'),
          detail: err?.message || '',
          closable: true,
        })
      } finally {
        setImporting(false)
        // Expose cancel hook to the resumed flow (declared but unused outside; satisfies linter)
        void cancelled
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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
      setInstruction(t('defaultProjectDescription'))
      setSessionName(t('defaultProjectName'))
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
        ? '是否包含内核运行时状态 (session.pkl)？\n\n选择「确定」会让归档包含已加载的 adata 等内存变量，恢复时可继续 pause/resume，但归档体积会变大（每个 snapshot 可能多几 GB）。仅在你打算导入回同一个实例时勾选。\n\n选择「取消」打包不带运行时状态（推荐，体积小、跨实例可移植）。'
        : 'Include kernel runtime state (session.pkl)?\n\nClicking OK bundles the live in-memory variables (adata etc.) so pause/resume can continue after import, but the archive can be GBs larger per snapshot. Only useful when importing back into the same instance.\n\nClick Cancel to skip runtime (recommended — smaller, portable across instances).'
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
      const meta = await apiFetch<{ status: string; size_bytes: number; filename: string }>(
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
          const link = document.createElement('a')
          link.href = `/api/session/${currentSession.id}/export/download`
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

  // Poll the backend import-status endpoint until the job finishes. No hard
  // timeout — large archives can legitimately take >1h to extract. The
  // shouldCancel() hook lets the caller bail out (set by the modal cancel button).
  // Returns the new session_id on success, or throws on failure/cancel.
  const pollImportUntilDone = async (pollId: string, shouldCancel: () => boolean): Promise<string> => {
    const POLL_INTERVAL = 3000
    const MAX_CONSECUTIVE_ERRORS = 10
    let consecutiveErrors = 0
    const formatBytes = (n: number) => {
      if (n <= 0) return '0 MB'
      if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
      if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
      return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
    }
    while (true) {
      if (shouldCancel()) {
        throw new Error(language === 'zh' ? '用户取消导入' : 'Import cancelled by user')
      }
      await new Promise(r => setTimeout(r, POLL_INTERVAL))
      let status: any = null
      try {
        status = await apiFetch<any>(`/session/import/${pollId}`)
        consecutiveErrors = 0
      } catch (pollErr: any) {
        consecutiveErrors += 1
        const is404 = pollErr.message?.includes('not found')
        // Bail out only after sustained errors; transient blips during long
        // extractions (proxy hiccups, server restart in dev) shouldn't kill it.
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS && !is404) {
          throw pollErr
        }
        if (is404 && consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          throw new Error(language === 'zh' ? '导入任务不存在或已过期' : 'Import job not found or expired')
        }
        continue
      }
      if (!status) continue

      if (status.status === 'completed') {
        return status.session_id
      }
      if (status.status === 'failed') {
        throw new Error(status.error || (language === 'zh' ? '解压失败' : 'Extraction failed'))
      }
      if (status.status === 'cancelled') {
        throw new Error(status.error || (language === 'zh' ? '导入已取消' : 'Import cancelled'))
      }

      // Render live progress from the server's status snapshot.
      const phase = status.phase || status.status || 'pending'
      const bytesProcessed: number = status.bytes_processed || 0
      const bytesTotal: number = status.bytes_total || 0
      const filesProcessed: number = status.files_processed || 0
      const currentFile: string = status.current_file || ''
      let percent: number | undefined
      let action = language === 'zh' ? '导入' : 'Import'
      let detail = ''
      if (phase === 'uploading') {
        if (bytesTotal > 0) percent = Math.min(100, Math.round((bytesProcessed / bytesTotal) * 100))
        detail = language === 'zh'
          ? `服务端接收中: ${formatBytes(bytesProcessed)} / ${formatBytes(bytesTotal)}`
          : `Server receiving: ${formatBytes(bytesProcessed)} / ${formatBytes(bytesTotal)}`
      } else if (phase === 'extracting' || phase === 'pending') {
        action = language === 'zh' ? '正在解压' : 'Extracting'
        if (bytesTotal > 0) percent = Math.min(99, Math.round((bytesProcessed / bytesTotal) * 100))
        const filesPart = filesProcessed > 0
          ? (language === 'zh' ? `已处理 ${filesProcessed} 个文件` : `${filesProcessed} files processed`)
          : (language === 'zh' ? '等待开始解压...' : 'Waiting to start extraction...')
        const filePart = currentFile
          ? (language === 'zh' ? ` · 当前: ${currentFile}` : ` · current: ${currentFile}`)
          : ''
        detail = filesPart + filePart
      } else {
        detail = language === 'zh' ? '处理中...' : 'Processing...'
      }
      setProgressInfo({
        action,
        detail,
        percent,
        onCancel: async () => {
          try {
            await apiFetch<any>(`/session/import/${pollId}/cancel`, { method: 'POST' })
          } catch (err) {
            console.debug('Cancel failed:', err)
          }
        },
        cancelLabel: language === 'zh' ? '取消导入' : 'Cancel Import',
      })
    }
  }

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || importing) return

    // Opt-in to load sandbox session pickles from the archive. Off by default
    // because dill.load is arbitrary code execution — the user must vouch for
    // the archive's origin. No backend restriction; entirely user judgment.
    const trustArchive = window.confirm(
      language === 'zh'
        ? '是否信任此归档并恢复内核运行时状态 (session.pkl)？\n\n⚠️ 风险提示：归档中的 session.pkl 在加载时会执行其中保存的 Python 对象代码 (dill.load = 任意代码执行)。仅在归档来自你完全信任的来源（例如你自己在同一实例的备份）时勾选「确定」。\n\n选择「取消」会丢弃 pkl，只恢复数据/历史/图表（推荐，安全默认）。'
        : 'Trust this archive and restore kernel runtime state (session.pkl)?\n\n⚠️ Warning: loading session.pkl runs arbitrary Python code (dill.load = arbitrary code execution). Only click OK for archives you fully trust (e.g. your own backup from the same instance).\n\nClick Cancel to drop pkl and restore only data/history/figures (recommended safe default).'
    )

    setImporting(true)
    const sizeMB = (file.size / 1024 / 1024).toFixed(1)
    setProgressInfo({ action: language === 'zh' ? '导入' : 'Import', detail: language === 'zh' ? `正在上传 (${sizeMB} MB)...` : `Uploading (${sizeMB} MB)...`, percent: 0 })

    // Cancellation control. cancelRequested gates the polling loop; if the user
    // clicks cancel mid-upload we abort the XHR. The pollId lives in
    // localStorage so a page refresh can resume polling instead of orphaning
    // the backend job.
    let cancelRequested = false
    let activeXhr: XMLHttpRequest | null = null
    let activePollId: string | null = null

    const cancelHandler = async () => {
      cancelRequested = true
      try {
        if (activeXhr) activeXhr.abort()
      } catch {}
      if (activePollId) {
        try {
          await apiFetch<any>(`/session/import/${activePollId}/cancel`, { method: 'POST' })
        } catch (err) {
          console.debug('Cancel request failed:', err)
        }
      }
    }

    try {
      // Pre-generate import_id so we can poll even if proxy times out before response arrives
      const preImportId = generateId()
      activePollId = preImportId
      // Persist so a page refresh can resume polling the same job instead of orphaning it.
      try {
        localStorage.setItem('scai.pendingImport', JSON.stringify({ importId: preImportId, fileName: file.name, startedAt: Date.now() }))
      } catch {}
      let uploadFinished = false

      // Phase 1: Upload with progress tracking — raw octet-stream upload
      // (no multipart). Server reads request.stream() directly into upload_temp,
      // skipping Starlette's SpooledTemporaryFile which used to add ~50 min
      // of redundant disk copying for multi-GB archives.
      let data: any = null
      try {
        data = await new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest()
          activeXhr = xhr
          xhr.open('POST', '/api/session/import')
          // Inject Authorization header — XHR doesn't go through apiFetch's auto-injection.
          // Without this, /api/session/ middleware treats request as unauth and the route returns 401.
          for (const [k, v] of Object.entries(authHeaders())) {
            xhr.setRequestHeader(k, v)
          }
          xhr.setRequestHeader('Content-Type', 'application/octet-stream')
          xhr.setRequestHeader('X-Filename', file.name)
          xhr.setRequestHeader('X-Import-Id', preImportId)
          if (trustArchive) xhr.setRequestHeader('X-Trust', 'true')

          xhr.upload.onprogress = (ev) => {
            if (ev.lengthComputable) {
              const pct = Math.round((ev.loaded / ev.total) * 100)
              if (pct >= 100) uploadFinished = true
              setProgressInfo({
                action: language === 'zh' ? '导入' : 'Import',
                detail: language === 'zh' ? `正在上传... ${pct}%` : `Uploading... ${pct}%`,
                percent: pct,
                onCancel: cancelHandler,
                cancelLabel: language === 'zh' ? '取消导入' : 'Cancel Import',
              })
            }
          }

          xhr.onload = () => {
            uploadFinished = true
            if (xhr.status >= 200 && xhr.status < 300) {
              try { resolve(JSON.parse(xhr.responseText)) } catch { console.debug('Import response parse failed'); resolve({}) }
            } else {
              try {
                const err = JSON.parse(xhr.responseText)
                reject(new Error(err.detail || 'Import failed'))
              } catch {
                reject(new Error('Import failed'))
              }
            }
          }

          xhr.onerror = () => reject(new Error('Network error'))
          xhr.onabort = () => reject(new Error(language === 'zh' ? '用户取消导入' : 'Import cancelled by user'))

          xhr.send(file)
        })
      } catch (uploadErr) {
        // If upload data was fully sent but response was lost (proxy timeout),
        // continue to polling — the backend is still processing
        if (!uploadFinished) throw uploadErr
        // uploadFinished=true means all bytes were sent; fall through to poll
      }

      let sessionId: string | undefined = data?.session_id

      // Phase 2: Poll extraction status (tar.gz async path)
      // Use import_id from response if available, otherwise the pre-generated one
      const pollId: string = data?.data?.import_id || preImportId
      activePollId = pollId
      const isTarGz = file.name.endsWith('.tar.gz') || file.name.endsWith('.tgz')

      if (isTarGz || !data) {
        // Unlimited polling — large archives can take >1h. Cancellation is via
        // the modal cancel button; transient network errors are tolerated.
        sessionId = await pollImportUntilDone(pollId, () => cancelRequested)
      }

      // Phase 3: Load session
      if (sessionId) {
        setProgressInfo({ action: language === 'zh' ? '导入' : 'Import', detail: language === 'zh' ? '正在加载会话...' : 'Loading session...' })
        clearMessages()
        await loadSession(sessionId)
      }

      // Phase 4: Show success
      setImporting(false)
      try { localStorage.removeItem('scai.pendingImport') } catch {}
      if (fileInputRef.current) fileInputRef.current.value = ''
      setProgressInfo({
        action: t('importSuccess'),
        detail: language === 'zh' ? '会话已成功导入' : 'Session imported successfully',
        closable: true,
        success: true
      })
    } catch (e: any) {
      console.error('Import failed:', e)
      setImporting(false)
      try { localStorage.removeItem('scai.pendingImport') } catch {}
      if (fileInputRef.current) fileInputRef.current.value = ''
      setProgressInfo({
        action: t('importFailed'),
        detail: e.message || '',
        closable: true
      })
    }
  }

  if (!isOpen) return null

  return (
    <div className="w-72 border-r border-surface-200 bg-accent-50 flex flex-col h-full shrink-0">
      {/* Export/Import Progress Modal */}
      {progressInfo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-modal p-6 w-80 text-center space-y-4 animate-scale-in">
            {/* 加载动画 - 进行中时显示 */}
            {!progressInfo.closable && (
              <Loader2 className="w-8 h-8 animate-spin text-accent-600 mx-auto" />
            )}

            {/* 成功图标 */}
            {progressInfo.closable && progressInfo.success && (
              <CheckCircle className="w-8 h-8 text-green-600 mx-auto" />
            )}

            {/* 失败/完成图标（导出完成或导入失败） */}
            {progressInfo.closable && !progressInfo.success && (
              <AlertCircle className="w-8 h-8 text-red-500 mx-auto" />
            )}

            {/* 文本信息 */}
            <div>
              <p className="text-sm font-semibold text-surface-800">{progressInfo.action}</p>
              <p className="text-xs text-surface-500 mt-1">{progressInfo.detail}</p>
            </div>

            {/* 进度条（如需要） */}
            {progressInfo.percent != null && (
              <div className="w-full bg-surface-200 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-accent-600 h-full rounded-full transition-all duration-300"
                  style={{ width: `${progressInfo.percent}%` }}
                />
              </div>
            )}

            {/* 确定/关闭按钮 - 仅在可关闭时显示 */}
            {progressInfo.closable && (
              <button
                onClick={() => {
                  if (progressInfo.onConfirm) {
                    progressInfo.onConfirm()
                  } else {
                    setProgressInfo(null)
                  }
                }}
                className="w-full px-4 py-2 bg-accent-600 text-white text-sm rounded-lg hover:bg-accent-700 transition-colors font-medium"
              >
                {t('confirm')}
              </button>
            )}

            {/* 取消按钮 - 进行中且支持取消时显示 */}
            {!progressInfo.closable && progressInfo.onCancel && (
              <button
                onClick={() => { progressInfo.onCancel && progressInfo.onCancel() }}
                className="w-full px-4 py-2 bg-surface-100 text-surface-700 text-sm rounded-lg hover:bg-surface-200 transition-colors font-medium"
              >
                {progressInfo.cancelLabel || (language === 'zh' ? '取消' : 'Cancel')}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Logo */}
      <div className="p-4 border-b border-surface-200">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-accent-500 to-accent-600 rounded-xl text-white shadow-glow-accent">
            <Dna className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-display font-bold text-surface-800">{t('appName')}</h1>
            <p className="text-xs text-surface-500">{t('appSubtitle')}</p>
          </div>
        </div>
      </div>

      {/* New Session Button */}
      <div className="p-3 space-y-2">
        {!isReadOnly && (
          <button
            onClick={() => setShowNewSession(true)}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-accent-500 to-accent-600 text-white rounded-xl font-medium hover:from-accent-600 hover:to-accent-700 transition-all duration-200 shadow-lg shadow-accent-500/25 hover:shadow-accent-500/40 hover:-translate-y-0.5 active:translate-y-0"
          >
            <Plus className="w-5 h-5" />
            {t('newAnalysis')}
          </button>
        )}

        <button
          onClick={() => setShowSessionBrowser(true)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-surface-100 text-surface-800 rounded-xl font-medium hover:bg-surface-200 transition-all"
        >
          <FolderOpen className="w-4 h-4" />
          {language === 'zh' ? '打开已有会话' : 'Open Saved Session'}
        </button>
      </div>

      {/* Session Browser Modal */}
      {showSessionBrowser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-modal w-full max-w-lg mx-4 max-h-[80vh] flex flex-col animate-scale-in">
            <div className="p-4 border-b border-surface-200 flex items-center justify-between">
              <h2 className="text-lg font-display font-semibold text-surface-800">
                {language === 'zh' ? '选择会话' : 'Select Session'}
              </h2>
              <button
                onClick={() => setShowSessionBrowser(false)}
                className="p-1 text-surface-400 hover:text-surface-600 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-4 flex-1 overflow-auto">
              {loadingSessions ? (
                <div className="text-center py-8 text-surface-500">
                  <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                  {language === 'zh' ? '加载中...' : 'Loading...'}
                </div>
              ) : sessionListError ? (
                <div className="text-center py-8 text-red-500">
                  <AlertCircle className="w-10 h-10 mx-auto mb-2 opacity-70" />
                  <p className="text-sm">{language === 'zh' ? '加载会话列表失败' : 'Failed to load sessions'}</p>
                  <button onClick={fetchSessions} className="mt-2 text-xs text-blue-500 hover:underline">
                    {language === 'zh' ? '重试' : 'Retry'}
                  </button>
                </div>
              ) : savedSessions.length === 0 ? (
                <div className="text-center py-8 text-surface-500">
                  <FolderOpen className="w-10 h-10 mx-auto mb-2 opacity-50" />
                  <p>{language === 'zh' ? '没有已保存的会话' : 'No saved sessions'}</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {savedSessions.map((session) => (
                    <div
                      key={session.id}
                      onClick={() => handleLoadSession(session.id)}
                      className={`p-3 rounded-xl border cursor-pointer transition-all hover:border-accent-300 hover:bg-accent-50 ${
                        currentSession?.id === session.id
                          ? 'border-accent-500 bg-accent-50'
                          : 'border-surface-200'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <h3 className="font-medium text-surface-800 truncate">
                            {session.name}
                          </h3>
                          <div className="flex items-center gap-3 mt-1 text-xs text-surface-500">
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {session.updated_at 
                                ? new Date(session.updated_at).toLocaleDateString()
                                : '-'}
                            </span>
                            <span className={`px-1.5 py-0.5 rounded text-xs ${
                              session.status === 'completed' 
                                ? 'bg-green-100 text-green-700'
                                : session.status === 'running'
                                  ? 'bg-blue-100 text-blue-700'
                                  : 'bg-surface-100 text-surface-600'
                            }`}>
                              {session.status}
                            </span>
                            <span>R{session.current_round || 0}</span>
                          </div>
                          <div className="flex gap-2 mt-1.5">
                            {session.has_pathology && (
                              <span className="text-xs px-1.5 py-0.5 bg-pink-100 text-pink-700 rounded">
                                {language === 'zh' ? '病理' : 'Path'}
                              </span>
                            )}
                            {session.has_sequencing && (
                              <span className="text-xs px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded">
                                {language === 'zh' ? '测序' : 'Seq'}
                              </span>
                            )}
                            {session.has_biochemistry && (
                              <span className="text-xs px-1.5 py-0.5 bg-amber-100 text-amber-700 rounded">
                                {language === 'zh' ? '生化' : 'Bio'}
                              </span>
                            )}
                          </div>
                        </div>
                        {!isReadOnly && (
                          <button
                            onClick={(e) => handleDeleteSavedSession(session.id, e)}
                            className="p-1.5 text-surface-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                            title={language === 'zh' ? '删除' : 'Delete'}
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="p-4 border-t border-surface-200 flex justify-between">
              <button
                onClick={fetchSessions}
                className="px-3 py-2 text-sm text-surface-800 hover:bg-surface-100 rounded-lg flex items-center gap-1"
              >
                <RefreshCw className="w-4 h-4" />
                {language === 'zh' ? '刷新' : 'Refresh'}
              </button>
              <button
                onClick={() => setShowSessionBrowser(false)}
                className="px-4 py-2 text-sm bg-surface-100 text-surface-700 rounded-lg hover:bg-surface-200"
              >
                {language === 'zh' ? '关闭' : 'Close'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* New Session Form */}
      {showNewSession && (
        <div className="px-3 pb-3">
          <div className="p-4 bg-surface-50 rounded-xl space-y-3 border border-surface-200">
            <input
              type="text"
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              placeholder={t('sessionNamePlaceholder')}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
            />
            <textarea
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              placeholder={t('researchGoalPlaceholder')}
              rows={12}
              className="w-full px-3 py-2 text-sm border border-surface-200 rounded-lg bg-white text-surface-800 placeholder:text-surface-400 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500 resize-y min-h-[200px]"
            />
            
            {/* Round Wait Time Setting */}
            <div className="p-3 bg-surface-50 rounded-lg border border-surface-200">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-accent-600" />
                <label className="text-sm font-medium text-surface-700">
                  {language === 'zh' ? '轮次等待时间' : 'Round Wait Time'}
                </label>
              </div>
              <select
                value={roundWaitSeconds}
                onChange={(e) => setRoundWaitSeconds(parseInt(e.target.value))}
                className="w-full text-sm font-mono text-surface-700 bg-white px-3 py-1.5 rounded border border-surface-200 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
              >
                {ROUND_WAIT_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>
                    {language === 'zh' ? opt.labelZh : opt.labelEn}
                  </option>
                ))}
              </select>
              <p className="text-xs text-surface-500 mt-2 leading-relaxed">
                {language === 'zh'
                  ? '💡 每轮分析后等待用户输入的时间，超时后自动继续'
                  : '💡 Wait time for user input between rounds, auto-continues on timeout'}
              </p>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setShowNewSession(false)}
                className="flex-1 px-3 py-2 text-sm text-surface-500 hover:bg-surface-100 rounded-lg"
              >
                {t('cancel')}
              </button>
              <button
                onClick={handleCreateSession}
                disabled={!instruction.trim()}
                className="flex-1 px-3 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50 font-medium"
              >
                {t('create')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Current Session + Spacer */}
      <div className="flex-1 min-h-0 flex flex-col">
        {currentSession && (
          <div className="px-3 pb-3 min-h-0 flex flex-col">
            <div className="p-3 bg-surface-50 border border-surface-200 rounded-xl min-h-0 flex flex-col">
              {editingSession ? (
                <div className="space-y-2">
                  <input
                    type="text"
                    defaultValue={currentSession.name}
                    onBlur={(e) => {
                      const val = e.target.value.trim()
                      if (val && val !== currentSession.name) updateSessionInfo({ name: val })
                    }}
                    className="w-full px-2 py-1 border border-surface-200 rounded text-sm font-medium bg-white text-surface-800"
                    autoFocus
                  />
                  <textarea
                    defaultValue={currentSession.instruction}
                    onBlur={(e) => {
                      const val = e.target.value.trim()
                      if (val !== currentSession.instruction) updateSessionInfo({ instruction: val })
                    }}
                    className="w-full px-2 py-1 border border-surface-200 rounded text-xs bg-white text-surface-800 resize-y"
                    rows={8}
                  />
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-accent-600 shrink-0" />
                    <select
                      value={roundWaitSeconds}
                      onChange={(e) => {
                        const v = parseInt(e.target.value)
                        setRoundWaitSeconds(v)
                        updateRoundWait(v).catch(err => {
                          console.error('Failed to update round wait', err)
                          loadSession(currentSession.id)
                        })
                      }}
                      className="flex-1 text-xs font-mono text-surface-700 bg-white px-2 py-1 rounded border border-surface-200 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
                    >
                      {ROUND_WAIT_OPTIONS.map(opt => (
                        <option key={opt.value} value={opt.value}>
                          {language === 'zh' ? opt.labelZh : opt.labelEn}
                        </option>
                      ))}
                    </select>
                  </div>
                  <button
                    onClick={() => setEditingSession(false)}
                    className="text-xs text-accent-600 hover:underline"
                  >
                    {language === 'zh' ? '完成' : 'Done'}
                  </button>
                </div>
              ) : (
                <>
                  <div className="flex items-center justify-between shrink-0">
                    <span className="text-sm font-medium text-surface-800">
                      {currentSession.name}
                    </span>
                    <div className="flex items-center gap-1">
                      {!isReadOnly && (
                        <button
                          onClick={() => setEditingSession(true)}
                          className="p-1 hover:bg-surface-100 rounded"
                          title={language === 'zh' ? '编辑' : 'Edit'}
                        >
                          <Pencil className="w-3.5 h-3.5 text-surface-500" />
                        </button>
                      )}
                      <button
                        onClick={handleRefreshSession}
                        className="p-1 hover:bg-surface-100 rounded"
                        title={language === 'zh' ? '刷新状态' : 'Refresh status'}
                      >
                        <RefreshCw className="w-3.5 h-3.5 text-surface-500" />
                      </button>
                      {!isReadOnly && (
                        <button
                          onClick={handleClearSession}
                          className="p-1 hover:bg-red-100 rounded"
                          title={language === 'zh' ? '清除项目' : 'Clear project'}
                        >
                          <Trash2 className="w-3.5 h-3.5 text-red-500" />
                        </button>
                      )}
                    </div>
                  </div>
                  <p className="text-sm text-surface-700 mt-1 overflow-y-auto min-h-0">
                    {currentSession.instruction}
                  </p>
                </>
              )}
              <div className="flex items-center gap-2 mt-2 shrink-0">
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  currentSession.status === 'completed' && !reportGenerated
                    ? 'bg-amber-100 text-amber-700'
                    : `status-${currentSession.status}`
                }`}>
                  {currentSession.status === 'completed'
                    ? (reportGenerated
                        ? (language === 'zh' ? '已完成' : 'completed')
                        : (language === 'zh' ? '分析完成' : 'analysis done'))
                    : currentSession.status}
                </span>
                <span className="text-xs text-accent-600">
                  Round {currentSession.currentRound}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Actions */}
      <div className="p-3 border-t border-surface-200 space-y-1">
        <DataConfigMenu
          sessionId={currentSession?.id}
          onImported={(sid) => {
            loadSession(sid).then(s => setCurrentSession(s))
          }}
          onExportSession={handleExport}
          exporting={exporting}
          onImportSession={() => fileInputRef.current?.click()}
          importing={importing}
          readOnly={isReadOnly}
        />
        {/* Hidden file input for session import */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".tar.gz,.tgz,.zip,.json"
          onChange={handleImport}
          className="hidden"
        />

        {!isReadOnly && (
          <button
            onClick={onOpenConfig}
            aria-label={t('apiSettings')}
            className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-surface-800 hover:bg-surface-100 rounded-lg"
          >
            <Settings className="w-4 h-4" aria-hidden="true" />
            {t('apiSettings')}
          </button>
        )}

        {/* UserMenu + About stacked */}
        <div className="flex flex-col gap-1 pt-1">
          {authUser?.role === 'admin' && (
            <button
              onClick={() => navigate('/admin')}
              className="flex items-center gap-2 px-3 py-2 text-sm text-surface-600 hover:bg-surface-100 rounded-lg transition-colors"
            >
              <Shield className="w-4 h-4" aria-hidden="true" />
              {language === 'zh' ? '用户管理' : 'User Admin'}
            </button>
          )}
          {authUser && <UserMenu />}
          <AboutDialog />
        </div>
      </div>
    </div>
  )
}
