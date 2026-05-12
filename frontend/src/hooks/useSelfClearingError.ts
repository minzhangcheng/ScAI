import { useState, useEffect } from 'react'

export function useSelfClearingError(clearAfterMs = 10000) {
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!error) return
    const timer = setTimeout(() => setError(null), clearAfterMs)
    return () => clearTimeout(timer)
  }, [error, clearAfterMs])

  return [error, setError] as const
}