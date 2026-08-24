import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SystemThemeProvider, useSystemTheme } from './SystemThemeProvider'

function ThemeProbe() {
  const { colorScheme } = useSystemTheme()
  return <output aria-label="system theme">{colorScheme}</output>
}

function createMediaQueryList(initialMatches: boolean) {
  let changeListener: ((event: MediaQueryListEvent) => void) | undefined
  const mediaQueryList = {
    matches: initialMatches,
    media: '(prefers-color-scheme: dark)',
    onchange: null,
    addEventListener: vi.fn((_type: string, listener: (event: MediaQueryListEvent) => void) => {
      changeListener = listener
    }),
    removeEventListener: vi.fn((_type: string, listener: (event: MediaQueryListEvent) => void) => {
      if (changeListener === listener) changeListener = undefined
    }),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(() => true),
  } as unknown as MediaQueryList

  return {
    mediaQueryList,
    change(matches: boolean) {
      changeListener?.({ matches } as MediaQueryListEvent)
    },
  }
}

afterEach(() => {
  vi.unstubAllGlobals()
  localStorage.removeItem('theme')
})

describe('SystemThemeProvider', () => {
  it('follows live modern or legacy media-query changes and cleans up', () => {
    const media = createMediaQueryList(true)
    vi.stubGlobal('matchMedia', vi.fn(() => media.mediaQueryList))

    const { unmount } = render(
      <SystemThemeProvider>
        <ThemeProbe />
      </SystemThemeProvider>,
    )

    expect(screen.getByLabelText('system theme')).toHaveTextContent('dark')
    expect(window.matchMedia).toHaveBeenCalledWith('(prefers-color-scheme: dark)')

    act(() => media.change(false))
    expect(screen.getByLabelText('system theme')).toHaveTextContent('light')

    unmount()
    expect(media.mediaQueryList.removeEventListener).toHaveBeenCalledOnce()
  })
})
