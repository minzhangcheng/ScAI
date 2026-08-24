// Saved-session picker modal. Split out of Sidebar.tsx; JSX copied verbatim.
import { X, RefreshCw, AlertCircle, FolderOpen, Clock, Trash2, Plus } from 'lucide-react'
import { useTranslation, type Language } from '../../i18n/translations'

interface SessionBrowserModalProps {
  show: boolean
  onClose: () => void
  language: Language
  loadingSessions: boolean
  sessionListError: string | null
  fetchSessions: () => void
  savedSessions: any[]
  onLoadSession: (sessionId: string) => void
  currentSessionId?: string
  isReadOnly: boolean
  onNewAnalysis: () => void
  onDeleteSession: (sessionId: string, e: React.MouseEvent) => void
}

export function SessionBrowserModal({
  show,
  onClose,
  language,
  loadingSessions,
  sessionListError,
  fetchSessions,
  savedSessions,
  onLoadSession,
  currentSessionId,
  isReadOnly,
  onNewAnalysis,
  onDeleteSession,
}: SessionBrowserModalProps) {
  const { t } = useTranslation(language)
  if (!show) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="session-browser-title"
        className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal w-full max-w-lg mx-4 max-h-[80vh] flex flex-col animate-scale-in"
      >
        <div className="p-4 border-b border-surface-200 dark:border-surface-700 flex items-center justify-between">
          <h2 id="session-browser-title" className="text-lg font-display font-semibold text-surface-800 dark:text-surface-100">
            {language === 'zh' ? '选择会话' : 'Select Session'}
          </h2>
          <button
            onClick={onClose}
            className="p-1 text-surface-400 hover:text-surface-600 dark:hover:text-surface-200 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {!isReadOnly && (
          <div className="p-4 border-b border-surface-200 dark:border-surface-700">
            <button
              type="button"
              onClick={onNewAnalysis}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent-600 text-white rounded-xl font-medium hover:bg-accent-700 transition-colors"
            >
              <Plus className="w-4 h-4" />
              {t('newAnalysis')}
            </button>
          </div>
        )}

        <div className="p-4 flex-1 overflow-auto">
          {loadingSessions ? (
            <div className="text-center py-8 text-surface-500 dark:text-surface-400">
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
            <div className="text-center py-8 text-surface-500 dark:text-surface-400">
              <FolderOpen className="w-10 h-10 mx-auto mb-2 opacity-50" />
              <p>{language === 'zh' ? '没有已保存的会话' : 'No saved sessions'}</p>
            </div>
          ) : (
            <div className="space-y-2">
              {savedSessions.map((session) => (
                <div
                  key={session.id}
                  onClick={() => onLoadSession(session.id)}
                  className={`p-3 rounded-xl border cursor-pointer transition-all hover:border-accent-300 hover:bg-accent-50 dark:hover:bg-accent-900/30 ${
                    currentSessionId === session.id
                      ? 'border-accent-500 bg-accent-50 dark:bg-accent-900/40'
                      : 'border-surface-200 dark:border-surface-700'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-surface-800 dark:text-surface-100 truncate">
                        {session.name}
                      </h3>
                      <p
                        className="mt-0.5 font-mono text-[10px] leading-tight text-surface-400 truncate"
                        title={session.id}
                      >
                        {session.id}
                      </p>
                      <div className="flex items-center gap-3 mt-1 text-xs text-surface-500 dark:text-surface-400">
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
                        onClick={(e) => onDeleteSession(session.id, e)}
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
            onClick={onClose}
            className="px-4 py-2 text-sm bg-surface-100 text-surface-700 rounded-lg hover:bg-surface-200"
          >
            {language === 'zh' ? '关闭' : 'Close'}
          </button>
        </div>
      </div>
    </div>
  )
}
