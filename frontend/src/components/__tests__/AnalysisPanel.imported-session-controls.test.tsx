import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisPanel } from '../AnalysisPanel'
import { useAppStore, type Session } from '../../stores/appStore'
import { apiFetch } from '../../hooks/useApi'

const resumeAnalysis = vi.fn()

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  generateId: () => 'test-id',
  useConfig: () => ({
    checkStatus: vi.fn(),
    validateApiKey: vi.fn(),
    validateSavedConfig: vi.fn(),
    saveConfig: vi.fn(),
  }),
  useSession: () => ({
    startAnalysis: vi.fn(),
    resumeAnalysis,
    continueRound: vi.fn(),
    pauseAnalysis: vi.fn(),
    stopAnalysis: vi.fn(),
  }),
  useUserConfig: () => ({
    getUserConfigStatus: vi.fn(),
    validateSavedConfig: vi.fn(),
  }),
}))

const mockApiFetch = apiFetch as unknown as ReturnType<typeof vi.fn>

function makeImportedSession(): Session {
  return {
    id: 'sess-imported',
    name: 'Imported mid-analysis',
    status: 'paused',
    currentRound: 3,
    analysisPhase: 'exploration',
    createdAt: '2026-07-29T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }
}

// An import lands paused, and which control it gets is
// decided entirely by runtime_restore_status. Both arrangements below are
// reachable from an ordinary export/import, and each leaves exactly one way
// forward — so a session that renders neither control is stuck with no error
// message to explain why.
describe('controls offered to an imported session', () => {
  beforeEach(() => {
    mockApiFetch.mockReset()
    mockApiFetch.mockResolvedValue({ success: true, message: '', data: {} })
    resumeAnalysis.mockReset()
    useAppStore.setState({
      currentSession: makeImportedSession(),
      messages: [],
      language: 'en',
      wsConnected: true,
      runtimeRestoreStatus: 'complete',
      runtimeRestoreWarning: null,
    })
  })

  it('offers Recover and refuses Resume when the archive carried no kernel', async () => {
    useAppStore.setState({
      runtimeRestoreStatus: 'unavailable',
      runtimeRestoreWarning: 'imported without kernel runtime state',
    })

    render(<AnalysisPanel />)
    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.getByRole('button', { name: /resume/i })).toBeDisabled()
    const recover = screen.getByRole('button', { name: /recover/i })
    expect(recover).toBeEnabled()

    // The label used to say "Re-run", which describes the outcome this button
    // takes only when no generation pair can be restored. Its ordinary outcome
    // is a rollback to the last saved runtime, and a user reading "re-run from
    // raw inputs" on a long analysis does not press it.
    expect(recover.getAttribute('title')).toMatch(/last saved runtime/i)
    expect(recover.textContent).not.toMatch(/re-run/i)

    fireEvent.click(recover)
    await waitFor(() => {
      expect(mockApiFetch).toHaveBeenCalledWith(
        '/session/sess-imported/run',
        expect.objectContaining({ method: 'POST' }),
      )
    })
  })

  // The mirror case: re-importing a runtime-carrying archive onto the instance
  // that wrote it keeps the kernel but drops the checkpoint, so the runtime
  // reads `complete` and Recover is not rendered at all. Resume has to work,
  // which is only true because the import arms a restore intent for it.
  it('offers Resume and no Recover when the kernel survived', async () => {
    render(<AnalysisPanel />)
    await act(async () => {
      await Promise.resolve()
    })

    const resume = screen.getByRole('button', { name: /resume/i })
    expect(resume).toBeEnabled()
    expect(screen.queryByRole('button', { name: /recover/i })).not.toBeInTheDocument()

    fireEvent.click(resume)
    await waitFor(() => {
      expect(resumeAnalysis).toHaveBeenCalled()
    })
  })

})
