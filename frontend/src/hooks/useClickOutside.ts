import { useEffect, type RefObject } from 'react'

/**
 * Invokes `handler` when a pointer event fires outside *all* provided refs.
 *
 * - `enabled=false` skips listener attachment entirely (cheap idle state).
 * - `eventType` defaults to 'mousedown' (match modal/menu UX); pass 'click'
 *   for delayed-close patterns (e.g. dropdowns that should react after the
 *   triggering element finishes its own onClick).
 */
export function useClickOutside(
  refs: ReadonlyArray<RefObject<HTMLElement | null>>,
  handler: (e: MouseEvent) => void,
  options: { enabled?: boolean; eventType?: 'mousedown' | 'click' } = {},
): void {
  const { enabled = true, eventType = 'mousedown' } = options

  useEffect(() => {
    if (!enabled) return
    const onPointer = (e: MouseEvent) => {
      const target = e.target as Node | null
      const inside = refs.some(r => r.current && target && r.current.contains(target))
      if (!inside) handler(e)
    }
    document.addEventListener(eventType, onPointer)
    return () => document.removeEventListener(eventType, onPointer)
  }, [enabled, eventType, handler, refs])
}