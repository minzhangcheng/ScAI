import { act, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Sidebar } from '../Sidebar'
import { useAppStore } from '../../stores/appStore'
import { apiFetch } from '../../hooks/useApi'

const loadSession = vi.fn()
const listSessions = vi.fn(async () => [])

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  generateId: () => 'pre-import-id',
  authHeaders: () => ({}),
  useSession: () => ({
    createSession: vi.fn(),
    loadSession,
    listSessions,
    deleteSession: vi.fn(),
    updateSessionInfo: vi.fn(),
    updateRoundWait: vi.fn(),
  }),
}))

// Neither of these participates in the import flow, and both drag in dialogs of
// their own; stub them so the test exercises Sidebar alone.
vi.mock('../UserMenu', () => ({ UserMenu: () => null }))
vi.mock('../AboutDialog', () => ({ AboutDialog: () => null }))

const mockApiFetch = apiFetch as unknown as ReturnType<typeof vi.fn>
const realXhr = globalThis.XMLHttpRequest

// Minimal XMLHttpRequest stand-in: the upload succeeds and answers with the
// body the test supplies. `''` reproduces the response the client cannot parse
// (proxy truncation), which Sidebar turns into `{}`.
function stubXhr(responseText: string) {
  const sent: string[] = []
  class XhrStub {
    status = 200
    responseText = responseText
    upload: { onprogress: ((ev: any) => void) | null } = { onprogress: null }
    onload: (() => void) | null = null
    onerror: (() => void) | null = null
    onabort: (() => void) | null = null
    open() {}
    setRequestHeader(k: string, v: string) { sent.push(`${k}: ${v}`) }
    abort() {}
    send() {
      this.upload.onprogress?.({ lengthComputable: true, loaded: 1, total: 1 })
      this.onload?.()
    }
  }
  ;(globalThis as any).XMLHttpRequest = XhrStub as unknown as typeof XMLHttpRequest
  return sent
}

// The session-import input is not the only file input Sidebar renders — the
// data-file menu has its own — so select it by the accept list it declares.
function importFile(container: HTMLElement, name: string) {
  const input = container.querySelector(
    'input[type="file"][accept=".tar.gz,.tgz,.zip,.json"]',
  ) as HTMLInputElement
  expect(input).toBeTruthy()
  const file = new File(['x'], name, { type: 'application/octet-stream' })
  Object.defineProperty(input, 'files', { value: [file], configurable: true })
  fireEvent.change(input)
}

function renderSidebar() {
  return render(
    <MemoryRouter>
      <Sidebar isOpen onOpenConfig={() => {}} />
    </MemoryRouter>,
  )
}

// The poll loop sleeps POLL_INTERVAL (3s) before every request, so every
// assertion below has to let at least one interval elapse.
async function tickPolls(ms = 7000) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

describe('what the import flow reports when it finishes', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mockApiFetch.mockReset()
    loadSession.mockReset()
    loadSession.mockResolvedValue({ id: 's1', status: 'paused' })
    listSessions.mockClear()
    localStorage.clear()
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    useAppStore.setState({
      currentSession: null,
      messages: [],
      language: 'en',
      activeTab: 'data',
      authUser: null,
      // Leaving hydration false keeps the priority-session loader out of the
      // way; this test is only about the import handler.
      _hydrated: false,
    })
  })

  afterEach(() => {
    globalThis.XMLHttpRequest = realXhr
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  // The extraction worker knows exactly which members it refused to restore and
  // used to tell only the server log, so the user saw an unqualified success.
  it('names what the worker skipped, and still calls the import a success', async () => {
    localStorage.setItem('scai.pendingImport', JSON.stringify({ importId: 'imp-1' }))
    mockApiFetch.mockImplementation(async (path: string) => {
      expect(path).toBe('/session/import/imp-1')
      return mockApiFetch.mock.calls.length === 1
        ? { status: 'extracting', phase: 'extracting' }
        : {
            status: 'completed',
            session_id: 's1',
            skipped: { retired_snapshot: 3, checkpoint_dropped: 1 },
            skipped_examples: ['retired_snapshot: s1/v2/objects/aa'],
          }
    })

    renderSidebar()
    await tickPolls()

    expect(screen.getByText('Import successful')).toBeInTheDocument()
    const detail = screen.getByText(/Session imported successfully/)
    expect(detail.textContent).toContain('4 item(s) were not restored')
    expect(detail.textContent).toContain('snapshot data from an older build (3)')
    expect(detail.textContent).toContain('workflow checkpoints (1)')
  })

  // An inline .zip registers no pollable job — only the tar.gz branch does — so
  // polling the pre-generated id just 404s for ~30s and then blames the import.
  it('admits the outcome is unknown when an inline import loses its response', async () => {
    stubXhr('')

    const { container } = renderSidebar()
    importFile(container, 'session.zip')
    await tickPolls()

    expect(mockApiFetch).not.toHaveBeenCalled()
    expect(screen.getByText('Import failed')).toBeInTheDocument()
    expect(
      screen.getByText(/the server response was lost, so the outcome is unknown/),
    ).toBeInTheDocument()
  })
})
