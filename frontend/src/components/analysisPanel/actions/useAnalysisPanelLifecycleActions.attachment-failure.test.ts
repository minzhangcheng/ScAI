import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore, type Session } from '../../../stores/appStore'
import { useAnalysisPanelLifecycleActions } from './useAnalysisPanelLifecycleActions'

const currentSession: Session = {
  id: 'session-one',
  name: 'Session one',
  status: 'waiting_input',
  currentRound: 1,
  createdAt: '2026-08-30T00:00:00Z',
  instruction: '',
  treatmentGroups: [],
  dataFiles: [],
  biochemistryIndicators: [],
  roundResults: [],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function options(overrides: Record<string, unknown> = {}) {
  return {
    currentSession,
    language: 'en',
    isForwardResumeBlocked: () => false,
    addMessage: vi.fn(),
    resumeAnalysis: vi.fn().mockResolvedValue(undefined),
    continueRound: vi.fn().mockResolvedValue(undefined),
    pauseAnalysis: vi.fn().mockResolvedValue(undefined),
    stopAnalysis: vi.fn().mockResolvedValue(undefined),
    wsConnected: true,
    wsConnect: vi.fn(),
    setShowFeedbackDialog: vi.fn(),
    setStopping: vi.fn(),
    setShowStopDialog: vi.fn(),
    ...overrides,
  }
}

describe('manual-input attachment uploads', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    useAppStore.setState({ currentSession })
  })

  it('keeps the dialog open and stops submission when an attachment fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      jsonResponse({ detail: { message: 'file exceeds limit' } }, 413),
    ))
    const lifecycle = options()
    const { result } = renderHook(() => useAnalysisPanelLifecycleActions(lifecycle))
    const file = new File(['payload'], 'oversized.csv', { type: 'text/csv' })

    await act(async () => {
      await result.current.handleDialogSend('analyze this file', [{
        file,
        targetAgent: 'bioinformatics',
      }])
    })

    expect(fetch).toHaveBeenCalledTimes(1)
    expect(String(vi.mocked(fetch).mock.calls[0][0])).toContain('/upload/data')
    expect(lifecycle.resumeAnalysis).not.toHaveBeenCalled()
    expect(lifecycle.setShowFeedbackDialog).not.toHaveBeenCalled()
    expect(lifecycle.addMessage).toHaveBeenCalledWith(expect.objectContaining({
      type: 'error',
      content: expect.stringContaining('oversized.csv: file exceeds limit'),
    }))
  })

  it('preserves upload, input, resume, and close ordering when all uploads succeed', async () => {
    const order: string[] = []
    vi.stubGlobal('fetch', vi.fn((input: string | URL | Request) => {
      const url = String(input)
      if (url.includes('/upload/')) {
        order.push('upload')
        return Promise.resolve(jsonResponse({ data: { file_id: 'file-one' } }))
      }
      order.push('input')
      return Promise.resolve(jsonResponse({ ok: true }))
    }))
    const lifecycle = options({
      resumeAnalysis: vi.fn(async () => { order.push('resume') }),
      setShowFeedbackDialog: vi.fn((show: boolean) => {
        if (!show) order.push('close')
      }),
    })
    const { result } = renderHook(() => useAnalysisPanelLifecycleActions(lifecycle))
    const file = new File(['payload'], 'valid.csv', { type: 'text/csv' })

    await act(async () => {
      await result.current.handleDialogSend('analyze this file', [{
        file,
        targetAgent: 'bioinformatics',
      }])
    })

    expect(order).toEqual(['upload', 'input', 'resume', 'close'])
    expect(lifecycle.addMessage).not.toHaveBeenCalled()
  })

  it('keeps the stop dialog open and reports a stop failure', async () => {
    const lifecycle = options({
      language: 'zh',
      stopAnalysis: vi.fn().mockRejectedValue(new Error('server failure')),
    })
    const { result } = renderHook(() => useAnalysisPanelLifecycleActions(lifecycle))

    await act(async () => {
      await result.current.handleStop('summarize')
    })

    expect(lifecycle.setStopping).toHaveBeenNthCalledWith(1, true)
    expect(lifecycle.setStopping).toHaveBeenLastCalledWith(false)
    expect(lifecycle.setShowStopDialog).not.toHaveBeenCalled()
    expect(lifecycle.setShowFeedbackDialog).not.toHaveBeenCalled()
    expect(lifecycle.addMessage).toHaveBeenCalledWith(expect.objectContaining({
      type: 'error',
      content: '停止分析失败：server failure',
    }))
  })
})
