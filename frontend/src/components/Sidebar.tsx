import { useState, useEffect } from 'react'
import { useSession } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { useTranslation } from '../i18n/translations'
import { 
  Plus, 
  Settings, 
  ChevronRight,
  Dna,
  Trash2,
  RefreshCw,
  FolderOpen,
  Clock,
  X,
  Repeat
} from 'lucide-react'

interface SidebarProps {
  isOpen: boolean
  onOpenConfig: () => void
}

export function Sidebar({ isOpen, onOpenConfig }: SidebarProps) {
  const { currentSession, setCurrentSession, clearMessages, language, maxRounds, setMaxRounds } = useAppStore()
  const { createSession, loadSession, listSessions, deleteSession } = useSession()
  const { t } = useTranslation(language)
  const [showNewSession, setShowNewSession] = useState(false)
  const [showSessionBrowser, setShowSessionBrowser] = useState(false)
  const [savedSessions, setSavedSessions] = useState<any[]>([])
  const [loadingSessions, setLoadingSessions] = useState(false)
  
  // Default project - pre-filled for convenience, users can modify
  const DEFAULT_PROJECT_NAME = 'Ctr1/ZnT1 DKO mouse liver'
  const DEFAULT_PROJECT_DESCRIPTION = `We focus on the impact of copper ions on cellular metabolism and cell death. Previously, we identified both Ctr1 and ZnT1 as copper ion transporters. Therefore, using the Alb-cre driver, we knocked out the Ctr1 and ZnT1 genes in mouse liver cells. We performed histological sectioning and H&E staining on the mouse liver tissues and conducted single-nucleus RNA sequencing (snRNA-Seq). Please integrate the analysis of the pathological sections and snRNA-Seq data from the Ctr1/ZnT1 DKO mice to summarize the effects on the mouse liver, covering both phenotypic and mechanistic aspects.`
  
  const [instruction, setInstruction] = useState(DEFAULT_PROJECT_DESCRIPTION)
  const [sessionName, setSessionName] = useState(DEFAULT_PROJECT_NAME)

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

  if (!isOpen) return null

  return (
    <div className="w-72 border-r border-slate-200 bg-white flex flex-col h-full shrink-0">
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
              rows={4}
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-violet-500 resize-none"
            />
            
            {/* Max Rounds Setting */}
            <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-100">
              <div className="flex items-center gap-2 mb-2">
                <Repeat className="w-4 h-4 text-indigo-500" />
                <label className="text-sm font-medium text-indigo-700">
                  {language === 'zh' ? '分析轮次' : 'Analysis Rounds'}
                </label>
                <span className="ml-auto text-sm font-mono text-indigo-600 bg-white px-2 py-0.5 rounded border border-indigo-200">
                  {maxRounds}
                </span>
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
                <span>{language === 'zh' ? '默认: 10' : 'Default: 10'}</span>
                <span>30</span>
              </div>
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
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-violet-900">
                {currentSession.name}
              </span>
              <div className="flex items-center gap-1">
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
            <div className="flex items-center gap-2 mt-2">
              <span className={`text-xs px-2 py-0.5 rounded-full status-${currentSession.status}`}>
                {currentSession.status}
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
