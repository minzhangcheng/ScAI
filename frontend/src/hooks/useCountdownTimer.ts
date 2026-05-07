import { useState, useEffect, useRef } from 'react'

/**
 * Wall-clock countdown timer hook.
 *
 * Uses a deadline (absolute timestamp) rather than interval counting so that
 * background tabs don't accumulate frozen time. Optional pause/resume shifts
 * the deadline when resuming.
 */
export function useCountdownTimer({
  countdownSeconds,
  onExpire,
  resetKey,
}: {
  countdownSeconds: number
  onExpire?: () => void
  resetKey?: string
}) {
  const hasCountdown = countdownSeconds > 0

  const [timeRemaining, setTimeRemaining] = useState(hasCountdown ? countdownSeconds : 0)
  const [isPaused, setIsPaused] = useState(false)

  const deadlineRef = useRef(hasCountdown ? Date.now() + countdownSeconds * 1000 : 0)
  const pausedAtRef = useRef<number | null>(null) // timestamp when paused, null if running
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const expiredRef = useRef(false) // prevent firing onExpire multiple times
  const onExpireRef = useRef(onExpire)
  onExpireRef.current = onExpire

  // Reset state when countdownSeconds changes (e.g., new round)
  useEffect(() => {
    const active = countdownSeconds > 0
    deadlineRef.current = active ? Date.now() + countdownSeconds * 1000 : 0
    pausedAtRef.current = null
    expiredRef.current = false
    setTimeRemaining(active ? countdownSeconds : 0)
    setIsPaused(false)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [countdownSeconds, resetKey])

  // Handle pause/resume: shift deadline when pausing/resuming
  useEffect(() => {
    if (!hasCountdown) return
    if (isPaused) {
      pausedAtRef.current = Date.now()
    } else if (pausedAtRef.current !== null) {
      // Resuming: shift deadline forward by the time spent paused
      const pausedDuration = Date.now() - pausedAtRef.current
      deadlineRef.current += pausedDuration
      pausedAtRef.current = null
    }
  }, [isPaused, hasCountdown])

  // Countdown timer — uses wall-clock diff so background tabs catch up correctly
  useEffect(() => {
    if (!hasCountdown) return

    intervalRef.current = setInterval(() => {
      if (pausedAtRef.current !== null) return // paused, don't update
      const remaining = Math.max(0, Math.ceil((deadlineRef.current - Date.now()) / 1000))
      setTimeRemaining(remaining)
      if (remaining <= 0) {
        if (intervalRef.current) clearInterval(intervalRef.current)
      }
    }, 500)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [hasCountdown])

  // Auto-proceed when timer reaches 0
  useEffect(() => {
    if (hasCountdown && timeRemaining === 0 && !expiredRef.current) {
      expiredRef.current = true
      onExpireRef.current?.()
    }
  }, [hasCountdown, timeRemaining])

  const pause = () => setIsPaused(true)
  const resume = () => setIsPaused(false)

  return { timeRemaining, isPaused, pause, resume, hasCountdown }
}