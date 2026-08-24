import { describe, expect, it } from 'vitest'
import type { ChatMessage } from '../types'

function assistant(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: 'm1',
    role: 'assistant',
    content: 'hello',
    timestamp: '2026-08-06T00:00:00Z',
    ...overrides,
  }
}

describe('chat history reconciliation', () => {
  it('preserves the array and element references when values are unchanged', async () => {
    const { reconcileChatHistory } = await import('./chatMessages')
    const local = [assistant()]
    const server = [assistant()]

    const result = reconcileChatHistory(server, local)

    expect(result).toBe(local)
    expect(result[0]).toBe(local[0])
  })

  it('replaces only the changed message and detects figure progress by ID', async () => {
    const { reconcileChatHistory } = await import('./chatMessages')
    const local = [
      assistant({ id: 'unchanged' }),
      assistant({ id: 'assistant-1', pending_figures: true }),
    ]
    const server = [
      assistant({ id: 'unchanged' }),
      assistant({
        id: 'assistant-1',
        pending_figures: false,
        figures: [{ name: 'result_plot', path: '/data/result_plot.png' }],
      }),
    ]

    const result = reconcileChatHistory(server, local)

    expect(result).not.toBe(local)
    expect(result[0]).toBe(local[0])
    expect(result[1]).not.toBe(local[1])
    expect(result[1].figures?.[0].name).toBe('result_plot')
  })

  it('keeps local optimistic messages while server values win by ID', async () => {
    const { reconcileChatHistory } = await import('./chatMessages')
    const optimistic = assistant({ id: 'optimistic', role: 'user', content: 'question' })
    const local = [assistant({ id: 'assistant-1' }), optimistic]
    const server = [assistant({ id: 'assistant-1', content: 'authoritative' })]

    const result = reconcileChatHistory(server, local)

    expect(result.map(message => message.id)).toEqual(['assistant-1', 'optimistic'])
    expect(result[0].content).toBe('authoritative')
    expect(result[1]).toBe(optimistic)
  })
})
