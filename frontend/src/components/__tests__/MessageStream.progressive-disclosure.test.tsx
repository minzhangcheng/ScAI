import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { useAppStore, type Message } from '../../stores/appStore'
import { MessageStream } from '../MessageStream'

describe('MessageStream progressive disclosure', () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [], language: 'en' })
  })

  it('keeps the newest 20 messages visible and expands older messages in place', () => {
    const messages: Message[] = Array.from({ length: 21 }, (_, index) => ({
      id: `message-${index}`,
      type: 'status',
      content: `status-${index}`,
      timestamp: `2026-08-27T00:00:${String(index).padStart(2, '0')}Z`,
    }))
    useAppStore.setState({ messages })

    render(<MessageStream />)

    expect(screen.queryByText('status-0')).not.toBeInTheDocument()
    expect(screen.getByText('status-20')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Show 1 older messages' }))
    expect(screen.getByText('status-0')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Hide 1 older messages' })).toBeInTheDocument()
  })
})
