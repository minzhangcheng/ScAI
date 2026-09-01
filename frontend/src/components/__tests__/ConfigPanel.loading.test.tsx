import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore } from '../../stores/appStore'
import { ConfigPanel } from '../ConfigPanel'

const { getConfig, getUserConfig, saveConfig, saveUserConfig } = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getUserConfig: vi.fn(),
  saveConfig: vi.fn(),
  saveUserConfig: vi.fn(),
}))

vi.mock('../../hooks/useApi', () => ({
  useConfig: () => ({
    getConfig,
    saveConfig,
    loading: false,
    error: null,
    clearError: vi.fn(),
  }),
  useUserConfig: () => ({
    getUserConfig,
    saveUserConfig,
    loading: false,
    error: null,
  }),
}))

describe('ConfigPanel initial load', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({ language: 'en', pubmedEmail: '' })
  })

  it('loads saved config before showing the form and preserves an entered HTTP URL', async () => {
    let resolveConfig!: (value: unknown) => void
    getConfig.mockReturnValue(new Promise(resolve => { resolveConfig = resolve }))
    saveConfig.mockResolvedValue(undefined)

    render(<ConfigPanel isAdmin onClose={() => {}} onSave={() => {}} />)

    expect(screen.getByText('Loading saved configuration...')).toBeInTheDocument()
    expect(screen.queryByPlaceholderText('https://openrouter.ai/api/v1')).not.toBeInTheDocument()

    resolveConfig({
      configured: true,
      llm_base_url: 'http://saved.example/v1',
      masked_api_key: 'sk-******789',
      has_proxy_url: false,
    })
    const serverInput = await screen.findByPlaceholderText('https://openrouter.ai/api/v1')
    expect(serverInput).toHaveValue('http://saved.example/v1')

    fireEvent.change(serverInput, { target: { value: 'http://localhost:8000/v1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save Config' }))

    await waitFor(() => expect(saveConfig).toHaveBeenCalledWith({
      apiKey: 'sk-******789',
      llmBaseUrl: 'http://localhost:8000/v1',
      proxyUrl: '',
    }))
  })
})
