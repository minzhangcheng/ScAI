import fs from 'node:fs'
import { it, expect } from 'vitest'

it('ChatPanel does not use array index as key in chatMessages', () => {
  const src = fs.readFileSync('src/components/ChatPanel.tsx', 'utf-8')
  expect(src).not.toMatch(/chatMessages\.map\(\(msg,\s*i\)/)
})
