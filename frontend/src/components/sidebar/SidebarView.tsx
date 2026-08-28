import type { CSSProperties } from 'react'
import { useNavigate } from 'react-router-dom'
import { Clock, FolderOpen, Pencil, Plus, RefreshCw, Settings, Shield, Trash2 } from 'lucide-react'

import { DEFAULT_SIDEBAR_WIDTH } from '../../constants/layout'
import { ROUND_WAIT_OPTIONS } from '../../constants/roundWait'
import { useAppStore } from '../../stores/appStore'
import { AboutDialog } from '../AboutDialog'
import { DataConfigMenu } from '../DataConfigMenu'
import { ScAIIcon } from '../ScAIIcon'
import { UserMenu } from '../UserMenu'
import { NewSessionForm } from './NewSessionForm'
import { ProgressModal } from './ProgressModal'
import { SessionBrowserModal } from './SessionBrowserModal'
import type { SidebarController } from './useSidebarSessions'

interface SidebarViewProps {
  controller: SidebarController
  isOpen: boolean
  onOpenConfig: () => void
  width?: number
}

export function SidebarView({
  controller, isOpen, onOpenConfig, width = DEFAULT_SIDEBAR_WIDTH,
}: SidebarViewProps) {
  const navigate = useNavigate()
  const {
    authUser, currentSession, editingSession, exporting, fetchSessions, fileInputRef,
    formKey, handleClearSession, handleCreateSession, handleDeleteSavedSession, handleExport,
    handleImport, handleLoadSession, handleRefreshSession, importing, instruction, isReadOnly,
    language, loadSession, loadingSessions, openSessionBrowser, progressInfo, reportGenerated, roundWaitSeconds, savedSessions,
    sessionListError, sessionName, setCurrentSession, setEditingSession, setInstruction, setProgressInfo,
    setRoundWaitSeconds, setSessionName, setShowNewSession, setShowSessionBrowser,
    showNewSession, showSessionBrowser, t, updateRoundWait, updateSessionInfo,
  } = controller
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
