import { useCallback } from 'react'

import { apiFetch, generateId } from '../../../hooks/useApi'
import { useAppStore, type Session } from '../../../stores/appStore'
import type { UploadedFileWithAgent } from '../../RoundFeedbackDialog'

interface StartActionOptions {
  readMode: boolean
  translatedReadModeRestriction: string
  translatedApiKeyInvalid: string
  authRole?: string
  showConfigPanel: (() => void) | null
  checkStatus: () => Promise<{ configured: boolean }>
  saveConfig: () => Promise<unknown>
  validateApiKey: () => Promise<{ valid: boolean; message?: string; uncertain?: boolean }>
  validateSavedConfig: () => Promise<{ valid: boolean; message?: string; uncertain?: boolean }>
  validateUserSavedConfig: () => Promise<{ valid: boolean; message?: string; uncertain?: boolean }>
  startAnalysis: () => Promise<string[]>
  notifySyncFailures: (failures: string[]) => void
  wsConnect: (() => void) | null
  addMessage: ReturnType<typeof useAppStore.getState>['addMessage']
  setValidating: (value: boolean) => void
  setShowValidationDialog: (value: boolean) => void
  setValidationMessage: (value: string) => void
  setValidationUncertain: (value: boolean) => void
}

export function useAnalysisPanelStartActions(options: StartActionOptions) {
  const handleStart = async () => {
    if (options.readMode) {
      alert(options.translatedReadModeRestriction)
      return
    }
    try {
      options.setValidating(true)
      let validation: { valid: boolean; message?: string; uncertain?: boolean }
      if (options.authRole === 'admin') {
        const { apiConfig, apiConfigured } = useAppStore.getState()
        if (!apiConfig.apiKey && !apiConfigured) {
          options.setValidating(false)
          options.showConfigPanel?.()
          return
        }
        if (apiConfig.apiKey) {
          const status = await options.checkStatus()
          if (!status.configured) await options.saveConfig()
          validation = await options.validateApiKey()
        } else {
          validation = await options.validateSavedConfig()
        }
      } else {
        validation = await options.validateUserSavedConfig()
      }
      options.setValidating(false)
      if (validation.valid) {
        const failures = await options.startAnalysis()
        if (failures) options.notifySyncFailures(failures)
        setTimeout(() => options.wsConnect?.(), 100)
      } else {
        options.setValidationMessage(validation.message || options.translatedApiKeyInvalid)
        options.setValidationUncertain(validation.uncertain || false)
        options.setShowValidationDialog(true)
      }
    } catch (error: any) {
      options.setValidating(false)
      console.error('Failed to start analysis:', error)
      options.addMessage({
        id: generateId(),
        type: 'error',
        content: `Failed to start analysis: ${error?.message || String(error)}`,
        timestamp: new Date().toISOString(),
      })
    }
  }

  const handleIgnoreValidationError = async () => {
    options.setShowValidationDialog(false)
    try {
      const failures = await options.startAnalysis()
      if (failures) options.notifySyncFailures(failures)
      setTimeout(() => options.wsConnect?.(), 100)
    } catch (error: any) {
      console.error('Failed to start analysis:', error)
      options.addMessage({
        id: generateId(),
        type: 'error',
        content: `Failed to start analysis: ${error?.message || String(error)}`,
        timestamp: new Date().toISOString(),
      })
    }
  }

  const handleReconfigureFromValidation = () => {
    options.setShowValidationDialog(false)
    options.showConfigPanel?.()
  }

  return { handleStart, handleIgnoreValidationError, handleReconfigureFromValidation }
}

interface LifecycleActionOptions {
  currentSession: Session | null
  language: string
  isForwardResumeBlocked: () => boolean
  addMessage: ReturnType<typeof useAppStore.getState>['addMessage']
  resumeAnalysis: () => Promise<void>
  continueRound: () => Promise<void>
  pauseAnalysis: () => Promise<void>
  stopAnalysis: (action: 'immediate' | 'summarize') => Promise<void>
  wsConnected: boolean
  wsConnect: (() => void) | null
  setShowFeedbackDialog: (show: boolean) => void
  setStopping: (stopping: boolean) => void
  setShowStopDialog: (show: boolean) => void
}

export function useAnalysisPanelLifecycleActions(options: LifecycleActionOptions) {
  const blockForwardResume = useCallback((): boolean => {
    if (!options.isForwardResumeBlocked()) return false
    options.addMessage({
      id: generateId(),
      type: 'error',
      content: options.language === 'zh'
        ? '当前会话禁止继续分析（运行时不可用或快照失败）。请使用重新运行或快照失败处置操作。'
        : 'Forward resume is blocked (runtime unavailable or snapshot failure). Use re-run or snapshot failure actions.',
      timestamp: new Date().toISOString(),
    })
    return true
  }, [options.isForwardResumeBlocked, options.addMessage, options.language])

  const handleDialogSend = useCallback(async (message: string, files: UploadedFileWithAgent[]) => {
    if (!options.currentSession || blockForwardResume()) return
    const uploadedData: Array<{ type: string; filename: string; id?: string }> = []
    for (const { file, targetAgent, groupId, sampleNumber, dataType } of files) {
      try {
        const formData = new FormData()
        formData.append('file', file)
        let endpoint: string
        const section = targetAgent === 'pathology' ? 'pathology'
          : targetAgent === 'bioinformatics' ? 'sequencing'
          : targetAgent === 'biochemistry' ? 'biochemistry'
          : null
        if (section) {
          formData.append('section', section)
          formData.append('data_type', dataType || (section === 'biochemistry'
            ? (file.type.startsWith('image/') ? 'image' : 'file')
            : 'user_upload'))
          if (groupId) formData.append('group_id', groupId)
          if (sampleNumber != null) formData.append('sample_number', String(sampleNumber))
          endpoint = `/session/${options.currentSession.id}/upload/data`
        } else {
          endpoint = `/session/${options.currentSession.id}/upload/file`
        }
        const response = await apiFetch<any>(endpoint, { method: 'POST', body: formData })
        if (response) {
          uploadedData.push({
            type: file.type.startsWith('image/') ? 'image' : 'file',
            filename: file.name,
            id: response.data?.file_id || response.data?.image_id,
          })
        }
      } catch (error) {
        console.error(`Failed to upload ${file.name}:`, error)
      }
    }
    try {
      await apiFetch(`/session/${options.currentSession.id}/human-input`, {
        method: 'POST',
        body: JSON.stringify({ message, uploaded_files: uploadedData, target_agent: 'pi' }),
      })
      await options.resumeAnalysis()
      if (!options.wsConnected) setTimeout(() => options.wsConnect?.(), 100)
    } catch (error) {
      console.error('Failed to send input:', error)
    }
    options.setShowFeedbackDialog(false)
  }, [
    options.currentSession,
    options.resumeAnalysis,
    options.wsConnected,
    options.wsConnect,
    blockForwardResume,
  ])

  const handleDialogSkip = useCallback(async () => {
    if (blockForwardResume()) {
      options.setShowFeedbackDialog(false)
      return
    }
    try {
      await options.continueRound()
      if (!options.wsConnected) setTimeout(() => options.wsConnect?.(), 100)
    } catch (error) {
      console.error('Failed to continue round:', error)
    }
    options.setShowFeedbackDialog(false)
  }, [options.continueRound, options.wsConnected, options.wsConnect, blockForwardResume])

  const handlePause = useCallback(async () => {
    try { await options.pauseAnalysis() } catch (error) { console.error('Failed to pause:', error) }
  }, [options.pauseAnalysis])

  const handleResume = useCallback(async () => {
    if (blockForwardResume()) return
    try {
      await options.resumeAnalysis()
      if (!options.wsConnected) setTimeout(() => options.wsConnect?.(), 100)
    } catch (error) {
      console.error('Failed to resume:', error)
    }
  }, [options.resumeAnalysis, options.wsConnected, options.wsConnect, blockForwardResume])

  const handleRerun = useCallback(async () => {
    if (!options.currentSession) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/run`, { method: 'POST' })
      useAppStore.getState().setRuntimeRestoreStatus('complete')
      useAppStore.getState().setRuntimeRestoreWarning(null)
      useAppStore.getState().updateSession({ status: 'running' })
      if (!options.wsConnected) setTimeout(() => options.wsConnect?.(), 100)
    } catch (error) {
      console.error('Failed to re-run analysis:', error)
      options.addMessage({
        id: generateId(), type: 'error',
        content: `Failed to re-run analysis: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date().toISOString(),
      })
    }
  }, [options.currentSession, options.wsConnected, options.wsConnect, options.addMessage])

  const handleStop = useCallback(async (action: 'immediate' | 'summarize') => {
    options.setStopping(true)
    try {
      await options.stopAnalysis(action)
      options.setShowStopDialog(false)
      options.setShowFeedbackDialog(false)
    } catch (error) {
      console.error('Failed to stop:', error)
    } finally {
      options.setStopping(false)
    }
  }, [options.stopAnalysis])

  return { blockForwardResume, handleDialogSend, handleDialogSkip, handlePause, handleResume, handleRerun, handleStop }
}
