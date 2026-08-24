import { useState, useEffect, useRef, useCallback, type CSSProperties } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSession, apiFetch, generateId, authHeaders } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { getTargetTab, hasCompletedSession, selectPrioritySession } from '../utils/sessionPriority'
import { useTranslation, type TranslationKey } from '../i18n/translations'
import { ROUND_WAIT_OPTIONS } from '../constants/roundWait'
import { ProgressModal } from './sidebar/ProgressModal'
import { SessionBrowserModal } from './sidebar/SessionBrowserModal'
import { NewSessionForm } from './sidebar/NewSessionForm'
import { DEFAULT_SESSION_NAME, DEFAULT_SESSION_INSTRUCTION } from '../constants/sessionDefaults'
import { DEFAULT_SIDEBAR_WIDTH } from '../constants/layout'
import { DataConfigMenu } from './DataConfigMenu'
import { UserMenu } from './UserMenu'
import { AboutDialog } from './AboutDialog'
import { ScAIIcon } from './ScAIIcon'
import {
  Plus,
  Settings,
  Shield,
  Trash2,
  RefreshCw,
  FolderOpen,
  Clock,
  Pencil
} from 'lucide-react'

function syncTabToSessionStatus(session: { status: string }) {
  useAppStore.getState().setActiveTab(getTargetTab(session.status))
}

// Human labels for the import worker's skipped-member counters. The kinds are
// the ones `_note_skip` reports (backend/routes/export_import_parts/import_impl.py);
// an unknown kind falls back to its raw key so a newly added counter still
// surfaces instead of being swallowed.
const IMPORT_SKIP_LABELS: Record<string, TranslationKey> = {
  snapshot_v2: 'importSkippedSnapshotV2',
  unsafe_path: 'importSkippedUnsafePath',
  off_layout: 'importSkippedOffLayout',
  untrusted_pickle: 'importSkippedUntrustedPickle',
  untrusted_runtime_h5ad: 'importSkippedUntrustedRuntimeH5ad',
  checkpoint_dropped: 'importSkippedCheckpointDropped',
}

interface SidebarProps {
  isOpen: boolean
  onOpenConfig: () => void
  width?: number
}

export function Sidebar({ isOpen, onOpenConfig, width = DEFAULT_SIDEBAR_WIDTH }: SidebarProps) {
  const navigate = useNavigate()
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
  const [progressInfo, setProgressInfo] = useState<{ action: string; detail: string; percent?: number; closable?: boolean; success?: boolean; onConfirm?: () => void; onCancel?: () => void; cancelLabel?: string } | null>(null)
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
        const { sessionId, skipped } = await pollImportUntilDone(pendingId, () => cancelled)
        await loadImportedSession(sessionId)
        showImportSuccess(skipped)
      } catch (err: any) {
        showImportFailed(err?.message || '')
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

  // Poll the backend import-status endpoint until the job finishes. No hard
  // timeout — large archives can legitimately take >1h to extract. The
  // shouldCancel() hook lets the caller bail out (set by the modal cancel button).
  // Returns the new session_id plus the worker's skipped-member counters on
  // success, or throws on failure/cancel.
  const pollImportUntilDone = async (
    pollId: string,
    shouldCancel: () => boolean,
  ): Promise<{ sessionId: string; skipped: Record<string, number> }> => {
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
        // The worker knows which members it dropped and now reports them
        // (import_impl.py: `_note_skip`); zero-valued kinds are filtered out
        // server-side, so an empty object is the common case.
        return {
          sessionId: status.session_id,
          skipped: status.skipped && typeof status.skipped === 'object' ? status.skipped : {},
        }
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

  // Success detail for a finished import, including one human line for what the
  // worker dropped. A skipped member is not a failed import, so this is purely
  // additive: with the common empty counter map nothing extra is rendered.
  const importSuccessDetail = (skipped: Record<string, number>): string => {
    const base = language === 'zh' ? '会话已成功导入' : 'Session imported successfully'
    const entries = Object.entries(skipped || {}).filter(([, n]) => Number(n) > 0)
    if (entries.length === 0) return base
    const total = entries.reduce((sum, [, n]) => sum + Number(n), 0)
    const items = entries
      .map(([kind, n]) => {
        const key = IMPORT_SKIP_LABELS[kind]
        return `${key ? t(key) : kind} (${n})`
      })
      .join(language === 'zh' ? '、' : ', ')
    const summary = t('importSkippedSummary', { count: total, items })
    return `${base}${language === 'zh' ? '。' : '. '}${summary}`
  }

  // The tail every finished import shares — the fresh-upload flow and the
  // resumed-poll flow both end here.
  const loadImportedSession = async (sessionId: string | undefined) => {
    if (!sessionId) return
    clearMessages()
    const session = await loadSession(sessionId)
    // An imported mid-analysis session lands paused or waiting, whose controls
    // live on the analysis tab; without this the user is left on the default
    // tab with the only way forward hidden.
    if (session) syncTabToSessionStatus(session)
  }

  const showImportSuccess = (skipped: Record<string, number>) => {
    try { localStorage.removeItem('scai.pendingImport') } catch {}
    setProgressInfo({
      action: t('importSuccess'),
      detail: importSuccessDetail(skipped),
      closable: true,
      success: true,
    })
  }

  const showImportFailed = (detail: string) => {
    try { localStorage.removeItem('scai.pendingImport') } catch {}
    setProgressInfo({
      action: t('importFailed'),
      detail,
      closable: true,
    })
  }

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || importing) return

    // Opt-in to load sandbox session pickles from the archive. Off by default
    // because dill.load is arbitrary code execution — the user must vouch for
    // the archive's origin. No backend restriction; entirely user judgment.
    const trustArchive = window.confirm(
      language === 'zh'
        ? '是否信任此归档并恢复内核运行时状态 (session.pkl)？\n\n⚠️ 风险提示：归档中的 session.pkl 在加载时会执行其中保存的 Python 对象代码 (dill.load = 任意代码执行)。仅在归档来自你完全信任的来源（例如你自己导出的备份）时勾选「确定」——归档来自哪台实例都可以，但必须是你信得过的。\n\n选择「取消」会丢弃 pkl，只恢复数据/历史/图表（安全默认）；该会话将无法继续分析，只能从数据重跑。'
        : 'Trust this archive and restore kernel runtime state (session.pkl)?\n\n⚠️ Warning: loading session.pkl runs arbitrary Python code (dill.load = arbitrary code execution). Only click OK for archives you fully trust (e.g. your own export). It may come from any instance — but it must be one you vouch for.\n\nClick Cancel to drop the pkl and restore only data/history/figures (safe default). That session will not be able to resume and must be re-run from the data.'
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
          // Percent-encode: header values are ByteStrings, and setRequestHeader
          // throws outright on a code point above 255. Export filenames keep the
          // session name verbatim, so every session named in Chinese produced an
          // archive its own UI could not re-upload. The backend decodes it.
          xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name))
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
      let skipped: Record<string, number> = {}

      // Phase 2: Poll extraction status (tar.gz async path)
      // Use import_id from response if available, otherwise the pre-generated one
      const asyncImportId: string | undefined = data?.data?.import_id
      const pollId: string = asyncImportId || preImportId
      activePollId = pollId
      // Match the backend's dispatch, which lowercases the name before testing
      // the extension (import_routes.py: `lowered = filename.lower()`), so an
      // archive named EXPORT.TAR.GZ is extracted in the background there. A
      // case-sensitive test here skipped polling, and the tar.gz response
      // carries no session_id, so the flow reported success without ever
      // loading the imported session.
      const lowerName = file.name.toLowerCase()
      const isTarGz = lowerName.endsWith('.tar.gz') || lowerName.endsWith('.tgz')

      if (asyncImportId || isTarGz) {
        // Unlimited polling — large archives can take >1h. Cancellation is via
        // the modal cancel button; transient network errors are tolerated.
        const polled = await pollImportUntilDone(pollId, () => cancelRequested)
        sessionId = polled.sessionId
        skipped = polled.skipped
      } else if (!sessionId) {
        // Inline .zip/.json imports register no job — import_routes.py creates
        // an _import_jobs entry only on the tar.gz branch — so there is nothing
        // to poll when the response is lost. Polling preImportId just 404s for
        // ~30s (10 × 3s) and reports a failure for an import that may well have
        // succeeded; say what is actually known instead.
        throw new Error(
          language === 'zh'
            ? '上传已完成，但服务端响应丢失，无法确认导入结果。请在「打开已有会话」中确认会话是否已导入。'
            : 'Upload finished but the server response was lost, so the outcome is unknown. Check "Open Saved Session" to see whether the session arrived.'
        )
      }

      // Phase 3: Load session
      if (sessionId) {
        setProgressInfo({ action: language === 'zh' ? '导入' : 'Import', detail: language === 'zh' ? '正在加载会话...' : 'Loading session...' })
      }
      await loadImportedSession(sessionId)

      // Phase 4: Show success
      setImporting(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
      showImportSuccess(skipped)
    } catch (e: any) {
      console.error('Import failed:', e)
      setImporting(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
      showImportFailed(e.message || '')
    }
  }

  if (!isOpen) return null

  return (
    <div
      id="app-sidebar"
      className="w-72 max-w-[85vw] lg:w-[var(--sidebar-width)] lg:max-w-none border-r border-surface-200 dark:border-surface-700 bg-accent-50 dark:bg-surface-900 flex flex-col h-full shrink-0"
      style={{ '--sidebar-width': `${width}px` } as CSSProperties}
    >
      <ProgressModal
        progressInfo={progressInfo}
        setProgressInfo={setProgressInfo}
        t={t}
        language={language}
      />

      {/* Logo */}
      <div className="p-4 border-b border-surface-200 dark:border-surface-700">
        <div className="flex items-center gap-3">
          <ScAIIcon className="w-10 h-10 rounded-xl" />
          <div>
            <h1 className="font-display font-bold text-surface-800 dark:text-surface-100">{t('appName')}</h1>
            <p className="text-xs text-surface-500 dark:text-surface-400">{t('appSubtitle')}</p>
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
          onClick={openSessionBrowser}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-surface-100 dark:bg-surface-800 text-surface-800 dark:text-surface-100 rounded-xl font-medium hover:bg-surface-200 dark:hover:bg-surface-700 transition-all"
        >
          <FolderOpen className="w-4 h-4" />
          {language === 'zh' ? '打开已有会话' : 'Open Saved Session'}
        </button>
      </div>

      <SessionBrowserModal
        show={showSessionBrowser}
        onClose={() => setShowSessionBrowser(false)}
        language={language}
        loadingSessions={loadingSessions}
        sessionListError={sessionListError}
        fetchSessions={fetchSessions}
        savedSessions={savedSessions}
        onLoadSession={handleLoadSession}
        currentSessionId={currentSession?.id}
        isReadOnly={isReadOnly}
        onNewAnalysis={() => {
          setShowSessionBrowser(false)
          useAppStore.getState().setSidebarOpen(true)
          setShowNewSession(true)
        }}
        onDeleteSession={handleDeleteSavedSession}
      />

      <NewSessionForm
        key={formKey}
        show={showNewSession}
        sessionName={sessionName}
        setSessionName={setSessionName}
        instruction={instruction}
        setInstruction={setInstruction}
        roundWaitSeconds={roundWaitSeconds}
        setRoundWaitSeconds={setRoundWaitSeconds}
        onCancel={() => setShowNewSession(false)}
        onCreate={handleCreateSession}
        t={t}
        language={language}
      />

      {/* Current Session + Spacer */}
      <div className="flex-1 min-h-0 flex flex-col">
        {currentSession && (
          <div className="px-3 pb-3 min-h-0 flex flex-col">
            <div className="p-3 bg-surface-50 dark:bg-surface-800 border border-surface-200 dark:border-surface-700 rounded-xl min-h-0 flex flex-col">
              {editingSession ? (
                <div className="space-y-2">
                  <input
                    type="text"
                    defaultValue={currentSession.name}
                    onBlur={(e) => {
                      const val = e.target.value.trim()
                      if (val && val !== currentSession.name) updateSessionInfo({ name: val })
                    }}
                    className="w-full px-2 py-1 border border-surface-200 dark:border-surface-700 rounded text-sm font-medium bg-white dark:bg-surface-900 text-surface-800 dark:text-surface-100"
                    autoFocus
                  />
                  <textarea
                    defaultValue={currentSession.instruction}
                    onBlur={(e) => {
                      const val = e.target.value.trim()
                      if (val !== currentSession.instruction) updateSessionInfo({ instruction: val })
                    }}
                    className="w-full px-2 py-1 border border-surface-200 dark:border-surface-700 rounded text-xs bg-white dark:bg-surface-900 text-surface-800 dark:text-surface-100 resize-y"
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
                      className="flex-1 text-xs font-mono text-surface-700 dark:text-surface-200 bg-white dark:bg-surface-900 px-2 py-1 rounded border border-surface-200 dark:border-surface-700 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
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
                    <span className="text-sm font-medium text-surface-800 dark:text-surface-100">
                      {currentSession.name}
                    </span>
                    <div className="flex items-center gap-1">
                      {!isReadOnly && (
                        <button
                          onClick={() => setEditingSession(true)}
                          className="p-1 hover:bg-surface-100 dark:hover:bg-surface-700 rounded"
                          title={language === 'zh' ? '编辑' : 'Edit'}
                        >
                          <Pencil className="w-3.5 h-3.5 text-surface-500" />
                        </button>
                      )}
                      <button
                        onClick={handleRefreshSession}
                        className="p-1 hover:bg-surface-100 dark:hover:bg-surface-700 rounded"
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
                  <p className="text-sm text-surface-700 dark:text-surface-200 mt-1 overflow-y-auto min-h-0">
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
      <div className="p-3 border-t border-surface-200 dark:border-surface-700 space-y-1">
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
            className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-surface-800 dark:text-surface-100 hover:bg-surface-100 dark:hover:bg-surface-700 rounded-lg"
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
              className="flex items-center gap-2 px-3 py-2 text-sm text-surface-600 dark:text-surface-300 hover:bg-surface-100 dark:hover:bg-surface-700 rounded-lg transition-colors"
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
