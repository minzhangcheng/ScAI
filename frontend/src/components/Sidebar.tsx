import { useState, useEffect, useRef } from 'react'
import { useSession, apiFetch } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import {
  Plus,
  Settings,
  Dna,
  Trash2,
  RefreshCw,
  FolderOpen,
  Clock,
  X,
  Repeat,
  Download,
  Upload,
  Loader2,
  CheckCircle,
  AlertCircle,
  Pencil
} from 'lucide-react'

interface SidebarProps {
  isOpen: boolean
  onOpenConfig: () => void
}

export function Sidebar({ isOpen, onOpenConfig }: SidebarProps) {
  const { currentSession, setCurrentSession, clearMessages, language, maxRounds, setMaxRounds, roundWaitSeconds, setRoundWaitSeconds, reportGenerated, setOpenSessionBrowserCallback } = useAppStore()
  const { createSession, loadSession, listSessions, deleteSession, updateSessionInfo } = useSession()
  const { t } = useTranslation(language)
  const [showNewSession, setShowNewSession] = useState(false)
  const [showSessionBrowser, setShowSessionBrowser] = useState(false)
  const [savedSessions, setSavedSessions] = useState<any[]>([])
  const [loadingSessions, setLoadingSessions] = useState(false)
  
  // Default project - pre-filled for convenience, users can modify
  const DEFAULT_PROJECT_NAME = 'Ctr1/ZnT1 DKO mouse liver'
  const DEFAULT_PROJECT_DESCRIPTION = `Our research group focuses on the metabolic crosstalk between copper and iron ions. We are specifically investigating the impact of these ions on cellular metabolism and cell death. Previously, we identified ZnT1 as a non-canonical copper transporter. In our functional assays, liver-specific ZnT1 knockout mice (via Alb-cre) exhibited no discernible phenotype, whereas ZnT1/Ctr1 double knockout (DKO) mice displayed severe hepatic abnormalities. This stark contrast suggests a functional redundancy where ZnT1 acts as a compensatory mechanism for Ctr1-mediated copper uptake.

To dissect this, we performed histological analysis (H&E) and single-nucleus RNA sequencing (snRNA-Seq) on the DKO livers. For the following analysis, please operate under the strict premise that ZnT1 is functioning as a copper transporter. Specifically, interpret the severe DKO phenotype and any downregulation of metabolic pathways in the snRNA-Seq data as a consequence of intracellular Copper deficiency (due to blocked uptake), rather than Zinc toxicity.`
  
  const [instruction, setInstruction] = useState(DEFAULT_PROJECT_DESCRIPTION)
  const [sessionName, setSessionName] = useState(DEFAULT_PROJECT_NAME)
  const [editingSession, setEditingSession] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [importing, setImporting] = useState(false)
  const [progressInfo, setProgressInfo] = useState<{ action: string; detail: string; percent?: number; closable?: boolean; success?: boolean; onConfirm?: () => void } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Auto-reload session from API on page refresh to restore full data (images etc.)
  // Uses currentSession?.id as dep so it fires after Zustand hydration restores the session
  const hasAutoReloaded = useRef(false)
  useEffect(() => {
    if (currentSession?.id && !hasAutoReloaded.current) {
      hasAutoReloaded.current = true
      loadSession(currentSession.id).catch(e => {
        // Session no longer exists on server (404) - clear stale session from state
        console.warn('Sidebar: Auto-reload session failed, clearing stale session:', e)
        setCurrentSession(null)
        clearMessages()
      })
    }
  }, [currentSession?.id]) // eslint-disable-line react-hooks/exhaustive-deps

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

  const fetchSessions = async () => {
    setLoadingSessions(true)
    try {
      const sessions = await listSessions()
      setSavedSessions(sessions)
    } catch (e) {
      console.error('Failed to fetch sessions:', e)
    } finally {
      setLoadingSessions(false)
    }
  }

  const handleLoadSession = async (sessionId: string) => {
    try {
      clearMessages()
      await loadSession(sessionId)
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
        console.error('Failed to delete session:', e)
      }
    }
  }

  const handleCreateSession = async () => {
    if (!instruction.trim()) return
    
    try {
      clearMessages()
      await createSession(instruction, sessionName || undefined)
      setShowNewSession(false)
      // Reset to defaults for next time
      setInstruction(DEFAULT_PROJECT_DESCRIPTION)
      setSessionName(DEFAULT_PROJECT_NAME)
    } catch (e) {
      console.error('Failed to create session:', e)
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
      await loadSession(currentSession.id)
    } catch (e) {
      console.error('Failed to refresh session:', e)
    }
  }

  const handleExport = async () => {
    if (!currentSession || exporting) return

    setExporting(true)

    // Phase 1: 服务端打包，显示 spinner
    setProgressInfo({
      action: t('exportPackaging'),
      detail: t('exportPackagingDetail'),
      closable: false
    })

    try {
      // 等待服务端完成打包（不下载文件，只等元数据）
      const res = await fetch(`/api/session/${currentSession.id}/export/prepare`, { method: 'POST' })
      if (!res.ok) throw new Error(`Export prepare failed: ${res.status}`)
      const meta = await res.json() as { status: string; size_bytes: number; filename: string }

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

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || importing) return
    setImporting(true)
    const sizeMB = (file.size / 1024 / 1024).toFixed(1)
    setProgressInfo({ action: language === 'zh' ? '导入' : 'Import', detail: language === 'zh' ? `正在上传 (${sizeMB} MB)...` : `Uploading (${sizeMB} MB)...`, percent: 0 })

    try {
      // Pre-generate import_id so we can poll even if proxy times out before response arrives
      // crypto.randomUUID() requires secure context (HTTPS or localhost); fallback for HTTP
      const preImportId = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`
      let uploadFinished = false

      // Phase 1: Upload with progress tracking
      let data: any = null
      try {
        data = await new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest()
          xhr.open('POST', '/api/session/import')

          xhr.upload.onprogress = (ev) => {
            if (ev.lengthComputable) {
              const pct = Math.round((ev.loaded / ev.total) * 100)
              if (pct >= 100) uploadFinished = true
              setProgressInfo({ action: language === 'zh' ? '导入' : 'Import', detail: language === 'zh' ? `正在上传... ${pct}%` : `Uploading... ${pct}%`, percent: pct })
            }
          }

          xhr.onload = () => {
            uploadFinished = true
            if (xhr.status >= 200 && xhr.status < 300) {
              try { resolve(JSON.parse(xhr.responseText)) } catch { resolve({}) }
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

          const formData = new FormData()
          formData.append('file', file)
          formData.append('import_id', preImportId)
          xhr.send(formData)
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
      const isTarGz = file.name.endsWith('.tar.gz') || file.name.endsWith('.tgz')

      if (isTarGz || !data) {
        // For tar.gz or if response was lost: poll for status
        setProgressInfo({ action: language === 'zh' ? '导入' : 'Import', detail: language === 'zh' ? '正在解压导入，请稍候...' : 'Extracting and importing, please wait...' })
        const MAX_POLL_MS = 30 * 60 * 1000  // 30 minutes timeout for very large files
        const POLL_INTERVAL = 3000
        const MAX_RETRIES = 5
        const startTime = Date.now()

        while (true) {
          if (Date.now() - startTime > MAX_POLL_MS) {
            throw new Error(language === 'zh' ? '导入超时，请刷新页面检查是否已成功' : 'Import timed out. Please refresh to check if it succeeded.')
          }
          await new Promise(r => setTimeout(r, POLL_INTERVAL))

          // Poll with retry: tolerate transient network errors and initial 404
          let status: any = null
          for (let retry = 0; retry < MAX_RETRIES; retry++) {
            try {
              status = await apiFetch<any>(`/session/import/${pollId}`)
              break
            } catch (pollErr: any) {
              // Tolerate 404 during early polls (job may not be registered yet)
              const is404 = pollErr.message?.includes('not found')
              if (retry < MAX_RETRIES - 1 && (is404 || Date.now() - startTime < 60000)) {
                await new Promise(r => setTimeout(r, 3000))
                continue
              }
              if (retry === MAX_RETRIES - 1) throw pollErr
              await new Promise(r => setTimeout(r, 3000))
            }
          }

          if (!status) continue  // all retries hit 404, loop around

          if (status.status === 'completed') {
            sessionId = status.session_id
            break
          }
          if (status.status === 'failed') {
            throw new Error(status.error || (language === 'zh' ? '解压失败' : 'Extraction failed'))
          }
          // "uploading" or "pending" → keep polling
        }
      }

      // Phase 3: Load session
      if (sessionId) {
        setProgressInfo({ action: language === 'zh' ? '导入' : 'Import', detail: language === 'zh' ? '正在加载会话...' : 'Loading session...' })
        clearMessages()
        await loadSession(sessionId)
      }

      // Phase 4: Show success
      setImporting(false)
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
    <div className="w-72 border-r border-slate-200 bg-white flex flex-col h-full shrink-0">
      {/* Export/Import Progress Modal */}
      {progressInfo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-xl p-6 w-80 text-center space-y-4">
            {/* 加载动画 - 进行中时显示 */}
            {!progressInfo.closable && (
              <Loader2 className="w-8 h-8 animate-spin text-violet-600 mx-auto" />
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
              <p className="text-sm font-semibold text-slate-800">{progressInfo.action}</p>
              <p className="text-xs text-slate-500 mt-1">{progressInfo.detail}</p>
            </div>

            {/* 进度条（如需要） */}
            {progressInfo.percent != null && (
              <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-violet-600 h-full rounded-full transition-all duration-300"
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
                className="w-full px-4 py-2 bg-violet-600 text-white text-sm rounded-lg hover:bg-violet-700 transition-colors font-medium"
              >
                {t('confirm')}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Logo */}
      <div className="p-4 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-violet-500 to-indigo-600 rounded-xl text-white">
            <Dna className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-slate-900">{t('appName')}</h1>
            <p className="text-xs text-slate-500">{t('appSubtitle')}</p>
          </div>
        </div>
      </div>

      {/* New Session Button */}
      <div className="p-3 space-y-2">
        <button
          onClick={() => setShowNewSession(true)}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-violet-600 to-indigo-600 text-white rounded-xl font-medium hover:from-violet-700 hover:to-indigo-700 transition-all shadow-lg shadow-violet-500/25"
        >
          <Plus className="w-5 h-5" />
          {t('newAnalysis')}
        </button>
        
        <button
          onClick={() => setShowSessionBrowser(true)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-slate-100 text-slate-700 rounded-xl font-medium hover:bg-slate-200 transition-all"
        >
          <FolderOpen className="w-4 h-4" />
          {language === 'zh' ? '打开已有会话' : 'Open Saved Session'}
        </button>
      </div>

      {/* Session Browser Modal */}
      {showSessionBrowser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 max-h-[80vh] flex flex-col">
            <div className="p-4 border-b border-slate-200 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-900">
                {language === 'zh' ? '选择会话' : 'Select Session'}
              </h2>
              <button
                onClick={() => setShowSessionBrowser(false)}
                className="p-1 text-slate-400 hover:text-slate-600 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-4 flex-1 overflow-auto">
              {loadingSessions ? (
                <div className="text-center py-8 text-slate-500">
                  <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                  {language === 'zh' ? '加载中...' : 'Loading...'}
                </div>
              ) : savedSessions.length === 0 ? (
                <div className="text-center py-8 text-slate-500">
                  <FolderOpen className="w-10 h-10 mx-auto mb-2 opacity-50" />
                  <p>{language === 'zh' ? '没有已保存的会话' : 'No saved sessions'}</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {savedSessions.map((session) => (
                    <div
                      key={session.id}
                      onClick={() => handleLoadSession(session.id)}
                      className={`p-3 rounded-xl border cursor-pointer transition-all hover:border-violet-300 hover:bg-violet-50 ${
                        currentSession?.id === session.id 
                          ? 'border-violet-500 bg-violet-50' 
                          : 'border-slate-200'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <h3 className="font-medium text-slate-900 truncate">
                            {session.name}
                          </h3>
                          <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
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
                                  : 'bg-slate-100 text-slate-600'
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
                        <button
                          onClick={(e) => handleDeleteSavedSession(session.id, e)}
                          className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                          title={language === 'zh' ? '删除' : 'Delete'}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="p-4 border-t border-slate-200 flex justify-between">
              <button
                onClick={fetchSessions}
                className="px-3 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg flex items-center gap-1"
              >
                <RefreshCw className="w-4 h-4" />
                {language === 'zh' ? '刷新' : 'Refresh'}
              </button>
              <button
                onClick={() => setShowSessionBrowser(false)}
                className="px-4 py-2 text-sm bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200"
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
          <div className="p-4 bg-slate-50 rounded-xl space-y-3">
            <input
              type="text"
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              placeholder={t('sessionNamePlaceholder')}
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-violet-500"
            />
            <textarea
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              placeholder={t('researchGoalPlaceholder')}
              rows={12}
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-violet-500 resize-y min-h-[200px]"
            />
            
            {/* Max Rounds Setting */}
            <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-100">
              <div className="flex items-center gap-2 mb-2">
                <Repeat className="w-4 h-4 text-indigo-500" />
                <label className="text-sm font-medium text-indigo-700">
                  {language === 'zh' ? '分析轮次' : 'Analysis Rounds'}
                </label>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={maxRounds}
                  onChange={(e) => {
                    const val = parseInt(e.target.value)
                    if (!isNaN(val) && val >= 1 && val <= 30) setMaxRounds(val)
                  }}
                  className="ml-auto w-16 text-sm font-mono text-indigo-600 bg-white px-2 py-0.5 rounded border border-indigo-200 text-center focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
              <input
                type="range"
                min="1"
                max="30"
                step="1"
                value={maxRounds}
                onChange={(e) => setMaxRounds(parseInt(e.target.value))}
                className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
              <div className="flex justify-between text-xs text-indigo-400 mt-1">
                <span>1</span>
                <span>{language === 'zh' ? '默认: 3' : 'Default: 3'}</span>
                <span>30</span>
              </div>
              <p className="text-xs text-indigo-500 mt-2 leading-relaxed">
                {language === 'zh'
                  ? '💡 建议先设较小值，完成后可通过对话框要求继续分析'
                  : '💡 Start small - you can request more rounds via chat later'}
              </p>
            </div>

            {/* Round Wait Time Setting */}
            <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-100">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-indigo-500" />
                <label className="text-sm font-medium text-indigo-700">
                  {language === 'zh' ? '轮次等待时间' : 'Round Wait Time'}
                </label>
              </div>
              <select
                value={roundWaitSeconds}
                onChange={(e) => setRoundWaitSeconds(parseInt(e.target.value))}
                className="w-full text-sm font-mono text-indigo-600 bg-white px-3 py-1.5 rounded border border-indigo-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value={0}>{language === 'zh' ? '不等待' : 'No wait'}</option>
                <option value={300}>{language === 'zh' ? '5 分钟' : '5 minutes'}</option>
                <option value={900}>{language === 'zh' ? '15 分钟' : '15 minutes'}</option>
                <option value={3600}>{language === 'zh' ? '60 分钟' : '60 minutes'}</option>
                <option value={-1}>{language === 'zh' ? '手动跳过' : 'Manual skip'}</option>
              </select>
              <p className="text-xs text-indigo-500 mt-2 leading-relaxed">
                {language === 'zh'
                  ? '💡 每轮分析后等待用户输入的时间，超时后自动继续'
                  : '💡 Wait time for user input between rounds, auto-continues on timeout'}
              </p>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setShowNewSession(false)}
                className="flex-1 px-3 py-2 text-sm text-slate-600 hover:bg-slate-200 rounded-lg"
              >
                {t('cancel')}
              </button>
              <button
                onClick={handleCreateSession}
                disabled={!instruction.trim()}
                className="flex-1 px-3 py-2 text-sm bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-50"
              >
                {t('create')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Current Session */}
      {currentSession && (
        <div className="px-3 pb-3">
          <div className="p-3 bg-violet-50 border border-violet-200 rounded-xl">
            {editingSession ? (
              <div className="space-y-2">
                <input
                  type="text"
                  defaultValue={currentSession.name}
                  onBlur={(e) => {
                    const val = e.target.value.trim()
                    if (val && val !== currentSession.name) updateSessionInfo({ name: val })
                  }}
                  className="w-full px-2 py-1 border border-violet-300 rounded text-sm font-medium bg-white"
                  autoFocus
                />
                <textarea
                  defaultValue={currentSession.instruction}
                  onBlur={(e) => {
                    const val = e.target.value.trim()
                    if (val !== currentSession.instruction) updateSessionInfo({ instruction: val })
                  }}
                  className="w-full px-2 py-1 border border-violet-300 rounded text-xs bg-white resize-y"
                  rows={8}
                />
                <button
                  onClick={() => setEditingSession(false)}
                  className="text-xs text-indigo-600 hover:underline"
                >
                  {language === 'zh' ? '完成' : 'Done'}
                </button>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-violet-900">
                    {currentSession.name}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setEditingSession(true)}
                      className="p-1 hover:bg-violet-100 rounded"
                      title={language === 'zh' ? '编辑' : 'Edit'}
                    >
                      <Pencil className="w-3.5 h-3.5 text-violet-500" />
                    </button>
                    <button
                      onClick={handleRefreshSession}
                      className="p-1 hover:bg-violet-100 rounded"
                      title={language === 'zh' ? '刷新状态' : 'Refresh status'}
                    >
                      <RefreshCw className="w-3.5 h-3.5 text-violet-500" />
                    </button>
                    <button
                      onClick={handleClearSession}
                      className="p-1 hover:bg-red-100 rounded"
                      title={language === 'zh' ? '清除项目' : 'Clear project'}
                    >
                      <Trash2 className="w-3.5 h-3.5 text-red-500" />
                    </button>
                  </div>
                </div>
                <p className="text-xs text-violet-600 mt-1 line-clamp-2">
                  {currentSession.instruction}
                </p>
              </>
            )}
            <div className="flex items-center gap-2 mt-2">
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
              <span className="text-xs text-violet-500">
                Round {currentSession.currentRound} / {maxRounds}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Bottom Actions */}
      <div className="p-3 border-t border-slate-200 space-y-1">
        {/* Export - only when a session is open */}
        {currentSession && (
          <button
            onClick={handleExport}
            disabled={exporting}
            className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-slate-600 hover:bg-slate-100 rounded-lg disabled:opacity-50"
          >
            {exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            {exporting ? t('exporting') : t('exportSession')}
          </button>
        )}

        {/* Import */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".tar.gz,.tgz,.zip,.json"
          onChange={handleImport}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={importing}
          className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-slate-600 hover:bg-slate-100 rounded-lg disabled:opacity-50"
        >
          {importing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
          {importing ? t('importing') : t('importSession')}
        </button>

        <button
          onClick={onOpenConfig}
          className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-slate-600 hover:bg-slate-100 rounded-lg"
        >
          <Settings className="w-4 h-4" />
          {t('apiSettings')}
        </button>
      </div>
    </div>
  )
}
