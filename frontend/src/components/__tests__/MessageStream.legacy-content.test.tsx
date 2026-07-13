import { render } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { MessageStream } from '../MessageStream'
import { useAppStore, type Message } from '../../stores/appStore'

// Regression: legacy/corrupted persisted messages whose array-shaped fields
// (findings / suggestions / directions / options) were stored as strings must
// not crash the whole app. A string has `.length`, so the old
// `x && x.length > 0` guards passed and then `.map()` threw
// "x.map is not a function", taking down App via the ErrorBoundary.

function baseMsg(partial: Partial<Message>): Message {
  return {
    id: Math.random().toString(36).slice(2),
    timestamp: '2026-07-13T00:00:00Z',
    ...partial,
  } as Message
}

const legacyMessages: Message[] = [
  // findings message with string fields (the exact crash from the bug report)
  baseMsg({
    type: 'findings',
    content: { summary: 'S', findings: 'a plain string', suggestions: 'a plain string' } as unknown as Message['content'],
  }),
  // result message with a string findings field
  baseMsg({
    type: 'result',
    content: { findings: 'a plain string', summary: 'ok' } as unknown as Message['content'],
  }),
  // direction_correlation with string directions
  baseMsg({
    type: 'direction_correlation',
    content: { correlation: 'c', directions: 'a plain string' } as unknown as Message['content'],
  }),
  // pi_question with string options
  baseMsg({
    type: 'pi_question',
    content: { question: 'q?', options: 'a plain string' } as unknown as Message['content'],
  }),
]

describe('MessageStream legacy content resilience', () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [], language: 'en' })
  })

  it('renders string-shaped legacy array fields without crashing', () => {
    useAppStore.setState({ messages: legacyMessages })
    // Would throw "x.map is not a function" before the fix.
    expect(() => render(<MessageStream />)).not.toThrow()
  })
})
