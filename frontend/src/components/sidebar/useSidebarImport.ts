import { useEffect, useRef, type ChangeEvent, type MutableRefObject } from 'react'

import { apiFetch, authHeaders, generateId } from '../../hooks/useApi'
import type { TranslationKey } from '../../i18n/translations'
import { useAppStore } from '../../stores/appStore'
import { getTargetTab } from '../../utils/sessionPriority'
import type { SidebarProgressInfo } from './types'

// Human labels for the import worker's skipped-member counters. The kinds are
// the ones `_note_skip` reports (backend/routes/export_import_parts/import_impl.py);
// an unknown kind falls back to its raw key so a newly added counter still
// surfaces instead of being swallowed.
const IMPORT_SKIP_LABELS: Record<string, TranslationKey> = {
  retired_snapshot: 'importSkippedRetiredSnapshot',
  unsafe_path: 'importSkippedUnsafePath',
  off_layout: 'importSkippedOffLayout',
  untrusted_pickle: 'importSkippedUntrustedPickle',
  untrusted_runtime_h5ad: 'importSkippedUntrustedRuntimeH5ad',
  checkpoint_dropped: 'importSkippedCheckpointDropped',
}

interface ImportContext {
  language: string
  setProgressInfo: (info: SidebarProgressInfo) => void
}

interface ImportRunContext extends ImportContext {
  clearMessages: () => void
  fileInputRef: MutableRefObject<HTMLInputElement | null>
  importing: boolean
  loadImportedSession: (sessionId: string | undefined) => Promise<void>
  setImporting: (importing: boolean) => void
  showImportFailed: (detail: string) => void
  showImportSuccess: (skipped: Record<string, number>) => void
}

interface SidebarImportOptions extends ImportContext {
  clearMessages: () => void
  fileInputRef: MutableRefObject<HTMLInputElement | null>
  importing: boolean
  loadSession: (sessionId: string) => Promise<any>
  setImporting: (importing: boolean) => void
  t: (key: TranslationKey, params?: Record<string, string | number>) => string
}

function syncTabToSessionStatus(session: { status: string }) {
  useAppStore.getState().setActiveTab(getTargetTab(session.status))
}

  // Poll the backend import-status endpoint until the job finishes. No hard
  // timeout — large archives can legitimately take >1h to extract. The
  // shouldCancel() hook lets the caller bail out (set by the modal cancel button).
  // Returns the new session_id plus the worker's skipped-member counters on
  // success, or throws on failure/cancel.
async function pollImportUntilDone(
  context: ImportContext,
  pollId: string,
  shouldCancel: () => boolean,
): Promise<{ sessionId: string; skipped: Record<string, number> }> {
  const { language, setProgressInfo } = context
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

async function runSidebarImport(
  context: ImportRunContext,
  e: ChangeEvent<HTMLInputElement>,
) {
  const {
    fileInputRef, importing, language, loadImportedSession,
    setImporting, setProgressInfo, showImportFailed, showImportSuccess,
  } = context
    const file = e.target.files?.[0]
    if (!file || importing) return

    // Opt-in to load sandbox session pickles from the archive. Off by default
    // because dill.load is arbitrary code execution — the user must vouch for
    // the archive's origin. No backend restriction; entirely user judgment.
    const trustArchive = window.confirm(
      language === 'zh'
        ? '是否信任此归档并恢复内核运行时状态 (session.pkl)？\n\n⚠️ 风险提示：归档中的 session.pkl 在加载时会执行其中保存的 Python 对象代码 (dill.load = 任意代码执行)。仅在归档来自你完全信任的来源（例如你自己导出的备份）时勾选「确定」——归档来自哪台实例都可以，但必须是你信得过的。\n\n选择「取消」会丢弃 pkl，只恢复数据/历史/图表（安全默认）；该会话将无法继续分析，只能从数据重跑。\n\n另外：若该归档标记为可继续，而本实例上已存在同 ID 的会话，导入会被拒绝——工作流断点与会话 ID 绑定，换个 ID 导入会丢掉断点，得到一个看似可继续、实际不能继续的会话。请先删除或重命名已有会话。'
        : 'Trust this archive and restore kernel runtime state (session.pkl)?\n\n⚠️ Warning: loading session.pkl runs arbitrary Python code (dill.load = arbitrary code execution). Only click OK for archives you fully trust (e.g. your own export). It may come from any instance — but it must be one you vouch for.\n\nClick Cancel to drop the pkl and restore only data/history/figures (safe default). That session will not be able to resume and must be re-run from the data.\n\nAlso: if the archive is marked resumable and a session with the same ID already exists here, the import is refused. The workflow checkpoint is bound to that ID, so importing under a new one would drop it and produce a session that looks resumable and is not. Delete or rename the existing session first.'
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
        const polled = await pollImportUntilDone(context, pollId, () => cancelRequested)
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
  }export function useSidebarImport(options: SidebarImportOptions) {
  const {
    clearMessages, language, loadSession, setImporting,
    setProgressInfo, t,
  } = options
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
        const { sessionId, skipped } = await pollImportUntilDone(options, pendingId, () => cancelled)
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


  const handleImport = (event: ChangeEvent<HTMLInputElement>) => runSidebarImport({
    ...options, loadImportedSession, showImportFailed, showImportSuccess,
  }, event)

  return { handleImport }
}
