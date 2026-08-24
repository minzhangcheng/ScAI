import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import type { PropsWithChildren } from 'react'

const DARK_MODE_QUERY = '(prefers-color-scheme: dark)'

export interface SystemTheme {
  colorScheme: 'light' | 'dark'
  isDark: boolean
}

const LIGHT_THEME: SystemTheme = { colorScheme: 'light', isDark: false }
const SystemThemeContext = createContext<SystemTheme>(LIGHT_THEME)

function systemPrefersDark(): boolean {
  return typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia(DARK_MODE_QUERY).matches
}

/** Exposes the live browser preference to JavaScript-owned colors. */
export function SystemThemeProvider({ children }: PropsWithChildren) {
  const [isDark, setIsDark] = useState(systemPrefersDark)

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return

    const query = window.matchMedia(DARK_MODE_QUERY)
    const handleChange = (event: MediaQueryListEvent) => setIsDark(event.matches)
    setIsDark(query.matches)

    if (typeof query.addEventListener === 'function') {
      query.addEventListener('change', handleChange)
      return () => query.removeEventListener('change', handleChange)
    }

    query.addListener(handleChange)
    return () => query.removeListener(handleChange)
  }, [])

  const theme = useMemo<SystemTheme>(() => ({
    colorScheme: isDark ? 'dark' : 'light',
    isDark,
  }), [isDark])

  return (
    <SystemThemeContext.Provider value={theme}>
      {children}
    </SystemThemeContext.Provider>
  )
}

export function useSystemTheme(): SystemTheme {
  return useContext(SystemThemeContext)
}
