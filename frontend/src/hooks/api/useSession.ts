import { useAppStore } from '../../stores/appStore'
import { useSessionAnalysis } from './session/useSessionAnalysis'
import { useSessionData } from './session/useSessionData'
import { useSessionLoading } from './session/useSessionLoading'

export function useSession() {
  const { currentSession, setCurrentSession, updateSession, setMessages, clearMessages } = useAppStore()
  const loading = useSessionLoading({ setCurrentSession, setMessages, clearMessages })
  const data = useSessionData({
    currentSession,
    setCurrentSession,
    clearMessages,
    setLoading: loading.setLoading,
    setError: loading.setError,
  })
  const analysis = useSessionAnalysis({ currentSession, updateSession })

  return {
    currentSession,
    loading: loading.loading,
    error: loading.error,
    createSession: data.createSession,
    loadSession: loading.loadSession,
    retryRealtimeHistory: loading.retryRealtimeHistory,
    listSessions: data.listSessions,
    deleteSession: data.deleteSession,
    saveSession: data.saveSession,
    updateSessionInfo: data.updateSessionInfo,
    updateRoundWait: data.updateRoundWait,
    addGroup: data.addGroup,
    updateGroup: data.updateGroup,
    deleteGroup: data.deleteGroup,
    uploadData: data.uploadData,
    deleteDataFile: data.deleteDataFile,
    addIndicator: data.addIndicator,
    deleteIndicator: data.deleteIndicator,
    startAnalysis: analysis.startAnalysis,
    pauseAnalysis: analysis.pauseAnalysis,
    resumeAnalysis: analysis.resumeAnalysis,
    continueRound: analysis.continueRound,
    stopAnalysis: analysis.stopAnalysis,
    exportState: analysis.exportState,
    getReport: analysis.getReport,
    updateSession,
  }
}
