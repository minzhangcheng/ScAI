import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

const root = process.cwd()
const source = (path: string) => readFileSync(join(root, path), 'utf8')

describe('ScAIIcon', () => {
  it('uses the single PNG asset on every brand surface', () => {
    expect(existsSync(join(root, 'public/scai-icon.png'))).toBe(true)
    expect(existsSync(join(root, 'public/cell-icon.svg'))).toBe(false)

    const iconSource = source('src/components/ScAIIcon.tsx')
    expect(iconSource).toContain('src="/scai-icon.png"')

    for (const path of [
      'src/App.tsx',
      'src/components/AuthLayout.tsx',
      'src/components/LoadingScreen.tsx',
      'src/components/Sidebar.tsx',
    ]) {
      const surface = source(path)
      expect(surface).toContain('ScAIIcon')
      expect(surface).toContain('<ScAIIcon')
    }
    expect(source('index.html')).toContain('href="/scai-icon.png"')
  })
})
