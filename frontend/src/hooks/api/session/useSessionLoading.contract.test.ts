import { describe, expect, it } from 'vitest'

import source from './useSessionLoading.ts?raw'

describe('useSessionLoading agent settings contract', () => {
  it('hydrates the Agents page from the selected session', () => {
    expect(source).toContain(`/session/\${sessionId}/agent-settings`)
    expect(source).toContain('transformAgentSettings(agentSettings)')
  })
})
