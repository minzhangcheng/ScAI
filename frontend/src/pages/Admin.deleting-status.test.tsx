import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { apiFetch } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { Admin } from './Admin'

vi.mock('../hooks/useApi', () => ({
  apiFetch: vi.fn(),
}))

describe('Admin deleting account state', () => {
  beforeEach(() => {
    vi.mocked(apiFetch).mockReset()
    vi.mocked(apiFetch).mockResolvedValue([
      {
        id: 'user-deleting',
        username: 'cleanup-target',
        role: 'user',
        status: 'deleting',
        created_at: '2026-08-03T00:00:00Z',
      },
    ])
    useAppStore.setState({
      language: 'en',
      authUser: {
        id: 'admin-active',
        username: 'admin',
        role: 'admin',
        status: 'active',
      },
    })
  })

  it('renders deletion as read-only while leaving cleanup retry available', async () => {
    render(
      <MemoryRouter>
        <Admin />
      </MemoryRouter>,
    )

    await screen.findByText('cleanup-target')

    expect(screen.getByText('deleting')).toBeInTheDocument()
    expect(screen.getByText('Deletion in progress')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry cleanup' })).toBeInTheDocument()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Password' })).not.toBeInTheDocument()
  })

  it('refreshes the row into deleting state after retryable cleanup conflict', async () => {
    vi.mocked(apiFetch)
      .mockResolvedValueOnce([{
        id: 'user-deleting',
        username: 'cleanup-target',
        role: 'user',
        status: 'active',
        created_at: '2026-08-03T00:00:00Z',
      }])
      .mockRejectedValueOnce(new Error('Owned data is still being retired'))
      .mockResolvedValueOnce([{
        id: 'user-deleting',
        username: 'cleanup-target',
        role: 'user',
        status: 'deleting',
        created_at: '2026-08-03T00:00:00Z',
      }])

    render(
      <MemoryRouter>
        <Admin />
      </MemoryRouter>,
    )
    await screen.findByText('cleanup-target')
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' }).at(-1)!)

    await screen.findByText('Deletion in progress')
    expect(screen.getByText('Owned data is still being retired')).toBeInTheDocument()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
  })
})
