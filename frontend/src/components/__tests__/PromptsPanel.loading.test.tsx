import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore } from '../../stores/appStore'
import type { AgentSettingsResponse } from '../../types'
import { defaultAgentSettings } from '../../utils/lastAgentSettings'
import { PromptsPanel } from '../PromptsPanel'

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }))

vi.mock('../../hooks/useApi', () => ({
  apiFetch: apiFetchMock,
  applyStoreUpdates: vi.fn(),
}))
vi.mock('../../utils/transformLoadResponse', () => ({
  transformAgentSettings: vi.fn(() => ({})),
}))
vi.mock('../prompts/usePromptsController', () => ({
  usePromptsController: vi.fn(() => ({})),
}))
vi.mock('../prompts/PromptEditor', () => ({
  PromptEditor: () => <div>Agent editor ready</div>,
}))

describe('PromptsPanel loading gate', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({
      _hydrated: true,
      authUser: {
        id: 'user-1', username: 'tester', role: 'user', status: 'active',
      },
      currentSession: { id: 'session-1' } as never,
      language: 'en',
    })
  })

  it('does not display an existing analysis editor before authoritative settings load', async () => {
    let resolveSettings!: (value: AgentSettingsResponse) => void
    apiFetchMock.mockReturnValue(new Promise(resolve => { resolveSettings = resolve }))

    render(<PromptsPanel />)

    expect(screen.queryByText('Agent editor ready')).not.toBeInTheDocument()
    expect(screen.getByText('Loading Agent settings…')).toBeInTheDocument()
    expect(apiFetchMock).toHaveBeenCalledWith('/session/session-1/agent-settings')

    resolveSettings(defaultAgentSettings('en'))
    await waitFor(() => expect(screen.getByText('Agent editor ready')).toBeInTheDocument())
  })
})
