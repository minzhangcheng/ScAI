import { render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Sidebar } from '../Sidebar'
import { useAppStore } from '../../stores/appStore'
import type { SessionListItem } from '../../types'

const createSession = vi.fn()
const loadSession = vi.fn()
const listSessions = vi.fn()

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  generateId: () => 'test-import-id',
  authHeaders: () => ({}),
  useSession: () => ({
    createSession,
    loadSession,
    listSessions,
    deleteSession: vi.fn(),
    updateSessionInfo: vi.fn(),
    updateRoundWait: vi.fn(),
  }),
}))

vi.mock('../UserMenu', () => ({ UserMenu: () => null }))
vi.mock('../AboutDialog', () => ({ AboutDialog: () => null }))

function session(id: string, status: string, createdAt = '2026-08-11T00:00:00Z'): SessionListItem {
  return {
    id,
    name: `Session ${id}`,
    status,
    current_round: 1,
    created_at: createdAt,
    updated_at: createdAt,
  }
}

function renderSidebar() {
  return render(
    <MemoryRouter>
      <Sidebar isOpen onOpenConfig={() => {}} />
    </MemoryRouter>,
  )
}

describe('Sidebar initial session', () => {
  beforeEach(() => {
    createSession.mockReset()
    loadSession.mockReset()
    listSessions.mockReset()
    localStorage.clear()
    useAppStore.setState({
      currentSession: {
        id: 'persisted-completed',
        name: 'Persisted completed session',
        status: 'completed',
      } as any,
      messages: [{ id: 'stale', type: 'status', content: 'stale', timestamp: '' }],
      language: 'en',
      activeTab: 'report',
      authUser: null,
      sidebarOpen: true,
      _hydrated: true,
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('loads active sessions without loading the persisted completed session', async () => {
    listSessions.mockResolvedValue([
      session('completed', 'completed'),
      session('running', 'running', '2026-08-11T01:00:00Z'),
    ])
    loadSession.mockResolvedValue({ id: 'running', status: 'running' })

    renderSidebar()

    await waitFor(() => expect(loadSession).toHaveBeenCalledWith('running'))
    expect(loadSession).not.toHaveBeenCalledWith('persisted-completed')
    expect(useAppStore.getState().activeTab).toBe('analysis')
    expect(screen.queryByRole('dialog', { name: 'Select Session' })).not.toBeInTheDocument()
  })

  it('hides New Analysis for read-only users', async () => {
    useAppStore.setState({ currentSession: null, messages: [], authUser: { status: 'readonly' } as any, _hydrated: true })
    listSessions.mockReset()
    listSessions.mockResolvedValue([session('completed-readonly', 'completed')])
    renderSidebar()
    const readonlyDialog = await screen.findByRole('dialog', { name: 'Select Session' })
    expect(within(readonlyDialog).queryByRole('button', { name: 'New Analysis' })).not.toBeInTheDocument()
  })
})
