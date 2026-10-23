import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore } from '../../stores/appStore'
import { clearToken, getToken, setToken } from '../../utils/authStorage'
import { SettingsDialog } from '../SettingsDialog'

const fetchMock = vi.fn()

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  setToken('original-token')
  useAppStore.setState({
    language: 'en',
    authUser: { id: 'alice-id', username: 'alice', role: 'user', status: 'active' },
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  clearToken()
})

function submitUsername() {
  fireEvent.change(screen.getByLabelText('New Username'), { target: { value: 'alice-renamed' } })
  fireEvent.change(screen.getByLabelText('Current Password'), { target: { value: 'password123' } })
  fireEvent.click(screen.getByRole('button', { name: 'Update Username' }))
}

describe('username settings', () => {
  it.each(['active', 'readonly'] as const)('renames a %s account and updates its displayed identity and token', async status => {
    useAppStore.setState(state => ({ authUser: { ...state.authUser!, status } }))
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ username: 'alice-renamed', access_token: 'new-token' })))
    render(<SettingsDialog open onClose={() => {}} />)

    submitUsername()

    await screen.findByText('Username updated. Use your new username next time you sign in.')
    expect(fetchMock).toHaveBeenCalledWith('/api/auth/profile/username', expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({ Authorization: 'Bearer original-token' }),
      body: JSON.stringify({ current_password: 'password123', new_username: 'alice-renamed' }),
    }))
    expect(useAppStore.getState().authUser).toEqual({ id: 'alice-id', username: 'alice-renamed', role: 'user', status })
    expect(getToken()).toBe('new-token')
    expect(screen.getByLabelText('Current Password')).toHaveValue('')
    if (status === 'readonly') expect(screen.queryByRole('button', { name: 'Email' })).not.toBeInTheDocument()
  })

  it('shows name conflicts without changing the current identity or token', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: 'Username already taken' }), { status: 409 }))
    render(<SettingsDialog open onClose={() => {}} />)
    submitUsername()
    await screen.findByText('Username already taken')
    expect(useAppStore.getState().authUser?.username).toBe('alice')
    expect(getToken()).toBe('original-token')
  })

  it('does not restore a login after the user signs out during a rename', async () => {
    let finish!: (response: Response) => void
    fetchMock.mockReturnValue(new Promise<Response>(resolve => { finish = resolve }))
    const request = useAppStore.getState().updateProfileUsername('password123', 'alice-renamed')
    useAppStore.getState().logout()
    finish(new Response(JSON.stringify({ username: 'alice-renamed', access_token: 'new-token' })))
    await request
    await waitFor(() => expect(useAppStore.getState().authUser).toBeNull())
    expect(getToken()).toBeNull()
  })
})
