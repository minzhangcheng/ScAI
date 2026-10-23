import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import { useAppStore } from './stores/appStore'

const { checkStatus, getUserConfigStatus, getUserConfig, connect, disconnect } = vi.hoisted(() => ({
  checkStatus: vi.fn(),
  getUserConfigStatus: vi.fn(),
  getUserConfig: vi.fn(),
  connect: vi.fn(),
  disconnect: vi.fn(),
}))

vi.mock('./hooks/useApi', () => ({
  useConfig: () => ({ checkStatus }),
  useUserConfig: () => ({ getUserConfigStatus, getUserConfig }),
  useWebSocket: () => ({ connect, disconnect }),
}))

vi.mock('./components/app/AppLayout', () => ({
  AppLayout: ({ showConfig, configured }: { showConfig: boolean; configured: boolean }) => (
    <main aria-label="Analysis workspace" data-configured={String(configured)}>
      {showConfig && <div role="dialog" aria-label="API Key settings" />}
    </main>
  ),
}))

beforeEach(() => {
  vi.clearAllMocks()
  checkStatus.mockResolvedValue({ configured: false })
  getUserConfigStatus.mockResolvedValue({ configured: false })
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ version: 'test' }))))
  useAppStore.setState({ language: 'en', apiConfigured: false })
})

afterEach(() => { vi.unstubAllGlobals() })

describe('configuration prompt after login', () => {
  it.each([
    ['user', 'readonly', false],
    ['admin', 'readonly', false],
    ['user', 'active', true],
    ['admin', 'active', true],
  ] as const)('handles an unconfigured %s account with %s status', async (role, status, shouldPrompt) => {
    useAppStore.setState({ authUser: { id: 'account-id', username: 'tester', role, status } })

    render(<App />)

    const workspace = await screen.findByRole('main', { name: 'Analysis workspace' })
    expect(workspace).toHaveAttribute('data-configured', 'false')
    expect(useAppStore.getState().apiConfigured).toBe(false)
    expect(role === 'admin' ? checkStatus : getUserConfigStatus).toHaveBeenCalledOnce()
    if (shouldPrompt) {
      expect(screen.getByRole('dialog', { name: 'API Key settings' })).toBeInTheDocument()
    } else {
      expect(screen.queryByRole('dialog', { name: 'API Key settings' })).not.toBeInTheDocument()
    }
  })
})
