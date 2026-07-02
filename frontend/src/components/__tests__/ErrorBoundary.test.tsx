import { render, screen } from '@testing-library/react'
import { it, expect } from 'vitest'
import { ErrorBoundary } from '../ErrorBoundary'

function Boom(): JSX.Element {
  throw new Error('x')
}

it('renders fallback on child crash', () => {
  render(<ErrorBoundary><Boom /></ErrorBoundary>)
  expect(screen.getByText(/something went wrong/i)).toBeInTheDocument()
})
