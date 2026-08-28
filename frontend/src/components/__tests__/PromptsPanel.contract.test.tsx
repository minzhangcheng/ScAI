import { describe, expect, it } from 'vitest'

import panelSource from '../PromptsPanel.tsx?raw'
import controllerSource from '../prompts/usePromptsController.ts?raw'

describe('PromptsPanel extraction contract', () => {
  it('keeps the public panel on one controller and editor', () => {
    expect(panelSource).toContain('usePromptsController()')
    expect(panelSource).toContain('<PromptEditor controller=')
  })

  it('keeps save payloads, reset defaults, and timer cleanup in the controller', () => {
    expect(controllerSource).toContain('/agent-server-config')
    expect(controllerSource).toContain('/vision-settings')
    expect(controllerSource).toContain('/literature-config')
    expect(controllerSource).toContain('clearTimeout(savedTimerRef.current)')
    expect(controllerSource).toContain("setLiteratureMode('pubmed_llm')")
  })
})
