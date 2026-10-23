import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiFetch } from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { clearToken, getToken, setToken } from '../utils/authStorage'
import { Admin } from './Admin'

vi.mock('../hooks/useApi', () => ({ apiFetch: vi.fn() }))

beforeEach(() => {
  vi.mocked(apiFetch).mockReset()
  setToken('admin-token')
  useAppStore.setState({
    language: 'en',
    authUser: { id: 'admin-id', username: 'boss', role: 'admin', status: 'active' },
  })
})

afterEach(() => { clearToken() })

describe('admin username changes', () => {
  it.each([false, true])('renames a user and refreshes the table (self=%s)', async self => {
    const user = { id: self ? 'admin-id' : 'user-id', username: self ? 'boss' : 'alice', role: self ? 'admin' : 'user', status: 'active', created_at: '2026-10-23T00:00:00Z' }
    vi.mocked(apiFetch)
      .mockResolvedValueOnce([user])
      .mockResolvedValueOnce({ username: 'renamed', access_token: self ? 'new-admin-token' : null })
      .mockResolvedValueOnce([{ ...user, username: 'renamed' }])
    render(<MemoryRouter><Admin /></MemoryRouter>)
    fireEvent.click(await screen.findByRole('button', { name: 'Rename' }))
    fireEvent.change(screen.getByPlaceholderText('Enter new username'), { target: { value: 'renamed' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await screen.findByText('renamed')
    expect(apiFetch).toHaveBeenCalledWith(`/admin/users/${user.id}/username`, {
      method: 'PATCH', body: JSON.stringify({ new_username: 'renamed' }),
    })
    await waitFor(() => expect(screen.queryByPlaceholderText('Enter new username')).not.toBeInTheDocument())
    expect(useAppStore.getState().authUser?.username).toBe(self ? 'renamed' : 'boss')
    expect(getToken()).toBe(self ? 'new-admin-token' : 'admin-token')
  })

  it('keeps the rename dialog open when a username is taken', async () => {
    vi.mocked(apiFetch)
      .mockResolvedValueOnce([{ id: 'user-id', username: 'alice', role: 'user', status: 'pending', created_at: '2026-10-23T00:00:00Z' }])
      .mockRejectedValueOnce(new Error('Username already taken'))
    render(<MemoryRouter><Admin /></MemoryRouter>)
    fireEvent.click(await screen.findByRole('button', { name: 'Rename' }))
    fireEvent.change(screen.getByPlaceholderText('Enter new username'), { target: { value: 'boss' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await screen.findByText('Username already taken')
    expect(screen.getByPlaceholderText('Enter new username')).toHaveValue('boss')
    expect(screen.getByText('alice')).toBeInTheDocument()
    expect(useAppStore.getState().authUser?.username).toBe('boss')
  })
})
