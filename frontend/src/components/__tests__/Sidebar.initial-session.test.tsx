import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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

  it('selects completed sessions without loading their payload', async () => {
    listSessions.mockResolvedValue([session('completed', 'completed')])
    useAppStore.setState({ currentSession: null, messages: [], _hydrated: true })

    renderSidebar()

    const dialog = await screen.findByRole('dialog', { name: 'Select Session' })
    expect(loadSession).not.toHaveBeenCalled()
    expect(listSessions).toHaveBeenCalledTimes(1)
    expect(within(dialog).getByText('Session completed')).toBeInTheDocument()
    expect(useAppStore.getState().currentSession).toBeNull()
    expect(useAppStore.getState().messages).toEqual([])
  })

  it('leaves pending, failed, and stopped-only lists alone', async () => {
    useAppStore.setState({ currentSession: null, messages: [], _hydrated: true })
    listSessions.mockResolvedValue([
      session('pending', 'pending'),
      session('failed', 'failed'),
      session('stopped', 'stopped'),
    ])
    renderSidebar()
    await waitFor(() => expect(listSessions).toHaveBeenCalledTimes(1))
    expect(loadSession).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog', { name: 'Select Session' })).not.toBeInTheDocument()
  })

  it('loads waiting_species_input as active without loading the persisted completed session', async () => {
    useAppStore.setState({ currentSession: null, messages: [], _hydrated: true })
    listSessions.mockResolvedValue([
      session('completed', 'completed'),
      session('species-gate', 'waiting_species_input'),
    ])
    loadSession.mockResolvedValue({ id: 'species-gate', status: 'waiting_species_input' })

    renderSidebar()

    await waitFor(() => expect(loadSession).toHaveBeenCalledWith('species-gate'))
    expect(loadSession).not.toHaveBeenCalledWith('persisted-completed')
    expect(useAppStore.getState().activeTab).toBe('analysis')
    expect(screen.queryByRole('dialog', { name: 'Select Session' })).not.toBeInTheDocument()
  })

  it('New Analysis is writable-only and opens the existing flow', async () => {
    useAppStore.setState({ currentSession: null, messages: [], authUser: null, _hydrated: true })
    listSessions.mockResolvedValue([session('completed', 'completed')])

    renderSidebar()

    const dialog = await screen.findByRole('dialog', { name: 'Select Session' })
    fireEvent.click(within(dialog).getByRole('button', { name: 'New Analysis' }))
    expect(screen.queryByRole('dialog', { name: 'Select Session' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Create' })).toBeInTheDocument()
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
