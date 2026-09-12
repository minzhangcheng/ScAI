/**
 * The annotation gate must be reachable after a browser restart.
 *
 * The dialog used to be built only by scanning the realtime message stream for
 * an `annotation_confirmation` message. That stream is server-authoritative but
 * lossy (bounded /load replay, capped stream), and it is never persisted to
 * localStorage. Once the message was gone the gate was
 * unreachable: the dialog never reopened, and the "Waiting for Annotation"
 * button only flipped a boolean while the payload stayed null.
 *
 * These tests pin the reload path: an empty message stream plus the server's
 * authoritative annotation state must still reach the dialog.
 */
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisPanel } from '../AnalysisPanel'
import { useAppStore, type Session } from '../../stores/appStore'
import type { AnnotationData } from '../../utils/annotationGate'

const sessionMocks = vi.hoisted(() => ({
  startAnalysis: vi.fn(),
  resumeAnalysis: vi.fn(),
  continueRound: vi.fn().mockResolvedValue(undefined),
  pauseAnalysis: vi.fn(),
  stopAnalysis: vi.fn(),
}))

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  generateId: () => 'test-id',
  useConfig: () => ({
    checkStatus: vi.fn(),
    validateApiKey: vi.fn(),
    validateSavedConfig: vi.fn(),
    saveConfig: vi.fn(),
  }),
  useSession: () => sessionMocks,
  useUserConfig: () => ({
    getUserConfigStatus: vi.fn(),
    validateSavedConfig: vi.fn(),
  }),
}))

function makeSession(status: Session['status']): Session {
  return {
    id: 'session-1',
    name: 'Annotation gate reload',
    status,
    currentRound: 1,
    analysisPhase: 'exploration',
    createdAt: '2026-07-14T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }
}

const gateFromServer: AnnotationData = {
  annotations: [
    {
      cluster_id: '0',
      cell_type: 'Hepatocyte',
      subtypes: [],
      key_markers: ['Alb'],
      confidence: 'high',
      evidence: 'Canonical markers',
      alternative_cell_types: [],
      low_confidence_reason: '',
      user_confirmed: false,
      user_override: null,
    },
  ],
  tissue: 'liver',
  organism: 'mouse',
  tissue_summary: 'Liver summary',
  proportion_group_labels: [],
  proportion_dimension: 'group',
}

describe('AnalysisPanel — annotation gate survives a reload', () => {
  beforeEach(() => {
    useAppStore.setState({
      currentSession: makeSession('waiting_annotation'),
      // A restarted browser starts with an empty stream: realtime messages are
      // never persisted, and the backend may no longer replay the gate message.
      messages: [],
      readMode: false,
    })
    useAppStore.getState().setAnnotationGateData(gateFromServer)
  })

  it('opens the dialog from the server annotation state when no gate message is in the stream', async () => {
    render(<AnalysisPanel />)

    await waitFor(() => {
      expect(screen.getByText('Hepatocyte')).toBeInTheDocument()
    })
  })

  it('reopens the dialog via the waiting-for-annotation button after it is dismissed', async () => {
    render(<AnalysisPanel />)

    await waitFor(() => expect(screen.getByText('Hepatocyte')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /^close$/i }))
    await waitFor(() => expect(screen.queryByText('Hepatocyte')).not.toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /waiting for annotation/i }))
    await waitFor(() => expect(screen.getByText('Hepatocyte')).toBeInTheDocument())
  })

  it('does not open the dialog when no gate is pending', async () => {
    useAppStore.setState({
      currentSession: makeSession('running'),
      messages: [],
    })
    useAppStore.getState().setAnnotationGateData(null)

    render(<AnalysisPanel />)

    await waitFor(() => {
      expect(screen.queryByText('Hepatocyte')).not.toBeInTheDocument()
    })
  })
})
