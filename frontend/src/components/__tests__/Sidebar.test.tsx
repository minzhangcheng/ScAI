import fs from 'node:fs'
import { it, expect } from 'vitest'

it('has no hardcoded species/disease in Sidebar source', () => {
  const src = fs.readFileSync('src/components/Sidebar.tsx', 'utf-8')
  expect(src).not.toMatch(/Ctr1|ZnT1|DKO|mouse liver/i)
})
