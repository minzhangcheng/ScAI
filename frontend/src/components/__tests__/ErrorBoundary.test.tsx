import { render, screen, fireEvent } from '@testing-library/react'
import { it, expect, beforeAll } from 'vitest'
import { ErrorBoundary } from '../ErrorBoundary'
import { PERSIST_STORAGE_KEY } from '../../constants/storage'
import { JWT_STORAGE_KEY } from '../../constants/auth'

function Boom(): JSX.Element {
  throw new Error('x')
}

// jsdom's location.reload isn't configurable via spyOn; replace it with a stub
// so the reset handler's reload() is a no-op instead of console noise.
beforeAll(() => {
  Object.defineProperty(window, 'location', {
    configurable: true,
    value: { ...window.location, reload: () => {} },
  })
})

it('renders fallback on child crash', () => {
  render(<ErrorBoundary><Boom /></ErrorBoundary>)
  expect(screen.getByText(/something went wrong/i)).toBeInTheDocument()
})

it('"Reset local data" clears app-owned localStorage keys', () => {
  localStorage.setItem(PERSIST_STORAGE_KEY, '{"stale":true}')
  localStorage.setItem(JWT_STORAGE_KEY, 'token')

  render(<ErrorBoundary><Boom /></ErrorBoundary>)
  fireEvent.click(screen.getByRole('button', { name: /reset local data/i }))

  expect(localStorage.getItem(PERSIST_STORAGE_KEY)).toBeNull()
  expect(localStorage.getItem(JWT_STORAGE_KEY)).toBeNull()
})
