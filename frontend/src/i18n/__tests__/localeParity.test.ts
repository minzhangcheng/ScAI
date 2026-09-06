import { describe, expect, it } from 'vitest'

import { en } from '../locales/en'
import { zh } from '../locales/zh'

/**
 * Key paths present in one locale and not the other.
 *
 * Empty, and measured rather than assumed: at d6c2dc5e both files declare the
 * same 416 keys. The plan for this batch recorded "37 vs 36" from a
 * case-insensitive `grep snapshot`, which counts *lines* — the Chinese values
 * differ from the English ones, so a line count was never a key count.
 *
 * This list may shrink. It must never grow: CLAUDE.md forbids editing one
 * locale without the other, and a key added to `en.ts` alone renders as its own
 * name to every Chinese user.
 */
const KNOWN_ASYMMETRY: readonly string[] = []

function keyPaths(value: unknown, prefix = ''): string[] {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return prefix ? [prefix] : []
  }
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) =>
    keyPaths(child, prefix ? `${prefix}.${key}` : key),
  )
}

describe('locale parity', () => {
  it('has no unknown key present in only one locale', () => {
    const englishKeys = new Set(keyPaths(en))
    const chineseKeys = new Set(keyPaths(zh))
    const onlyOne = [
      ...[...englishKeys].filter((key) => !chineseKeys.has(key)),
      ...[...chineseKeys].filter((key) => !englishKeys.has(key)),
    ]

    expect(onlyOne.filter((key) => !KNOWN_ASYMMETRY.includes(key)).sort()).toEqual([])
  })

  it('records no asymmetry that has since been repaired', () => {
    const englishKeys = new Set(keyPaths(en))
    const chineseKeys = new Set(keyPaths(zh))
    const stale = KNOWN_ASYMMETRY.filter(
      (key) => englishKeys.has(key) === chineseKeys.has(key),
    )

    expect(stale).toEqual([])
  })

  it('leaves no value empty in either locale', () => {
    const empty: string[] = []
    for (const [locale, table] of [['en', en], ['zh', zh]] as const) {
      for (const path of keyPaths(table)) {
        const value = path
          .split('.')
          .reduce<unknown>((node, part) => (node as Record<string, unknown>)?.[part], table)
        if (typeof value === 'string' && value.trim() === '') {
          empty.push(`${locale}:${path}`)
        }
      }
    }

    expect(empty.sort()).toEqual([])
  })
})
