import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const SIDEBAR_ROOT = path.dirname(fileURLToPath(import.meta.url))

function source(name: string): string {
  return fs.readFileSync(path.join(SIDEBAR_ROOT, name), 'utf8')
}

describe('sidebar dependency direction', () => {
  it('keeps the import worker independent from the sessions controller', () => {
    expect(source('useSidebarImport.ts')).not.toMatch(
      /from\s+['"]\.\/useSidebarSessions['"]/,
    )
    expect(source('useSidebarSessions.ts')).toMatch(
      /export\s+type\s+\{\s*SidebarProgressInfo\s*\}\s+from\s+['"]\.\/types['"]/,
    )
  })
})
