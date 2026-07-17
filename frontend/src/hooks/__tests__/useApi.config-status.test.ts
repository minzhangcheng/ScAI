import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore } from '../../stores/appStore'
import { useConfig } from '../useApi'


const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

vi.mock('../../utils/authStorage', () => ({
  getToken: vi.fn(() => 'test-token'),
  clearToken: vi.fn(),
}))


describe('useConfig snapshot capability', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({ snapshotV2Enabled: false })
  })

  it('hydrates the Snapshot V2 capability from config status', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: new Headers(),
      json: () => Promise.resolve({
        configured: true,
        has_api_key: true,
        snapshot_v2_enabled: true,
      }),
    })

    const { result } = renderHook(() => useConfig())
    await act(async () => {
      await result.current.checkStatus()
    })

    expect(useAppStore.getState().snapshotV2Enabled).toBe(true)
  })

  it('fails closed when config status cannot be loaded', async () => {
    useAppStore.setState({ snapshotV2Enabled: true })
    mockFetch.mockRejectedValueOnce(new Error('offline'))

    const { result } = renderHook(() => useConfig())
    await act(async () => {
      await result.current.checkStatus()
    })

    expect(useAppStore.getState().snapshotV2Enabled).toBe(false)
  })
})
