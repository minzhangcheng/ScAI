import { useCallback, useEffect, useRef, useState } from 'react'
import {
  DEFAULT_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
  SIDEBAR_WIDTH_STEP,
  clampSidebarWidth,
} from '../constants/layout'

interface SidebarResizeHandleProps {
  width: number
  onWidthChange: (width: number) => void
  label: string
}

export function SidebarResizeHandle({ width, onWidthChange, label }: SidebarResizeHandleProps) {
  const [isResizing, setIsResizing] = useState(false)
  const dragStartRef = useRef({ x: 0, width: DEFAULT_SIDEBAR_WIDTH })

  const handleMouseMove = useCallback((event: MouseEvent) => {
    if (!isResizing) return
    const nextWidth = dragStartRef.current.width + event.clientX - dragStartRef.current.x
    onWidthChange(clampSidebarWidth(nextWidth))
  }, [isResizing, onWidthChange])

  const stopResizing = useCallback(() => {
    setIsResizing(false)
  }, [])

  useEffect(() => {
    if (!isResizing) return

    const previousCursor = document.body.style.cursor
    const previousUserSelect = document.body.style.userSelect
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', stopResizing)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', stopResizing)
      document.body.style.cursor = previousCursor
      document.body.style.userSelect = previousUserSelect
    }
  }, [handleMouseMove, isResizing, stopResizing])

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    let nextWidth: number | null = null
    if (event.key === 'ArrowLeft') nextWidth = width - SIDEBAR_WIDTH_STEP
    if (event.key === 'ArrowRight') nextWidth = width + SIDEBAR_WIDTH_STEP
    if (event.key === 'Home') nextWidth = MIN_SIDEBAR_WIDTH
    if (event.key === 'End') nextWidth = MAX_SIDEBAR_WIDTH
    if (nextWidth === null) return

    event.preventDefault()
    onWidthChange(clampSidebarWidth(nextWidth))
  }

  return (
    <div
      role="separator"
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemin={MIN_SIDEBAR_WIDTH}
      aria-valuemax={MAX_SIDEBAR_WIDTH}
      aria-valuenow={Math.round(width)}
      aria-valuetext={`${Math.round(width)} px`}
      tabIndex={0}
      title={label}
      onMouseDown={(event) => {
        if (event.button !== 0) return
        event.preventDefault()
        dragStartRef.current = { x: event.clientX, width }
        setIsResizing(true)
      }}
      onDoubleClick={() => onWidthChange(DEFAULT_SIDEBAR_WIDTH)}
      onKeyDown={handleKeyDown}
      className={`hidden lg:flex w-1.5 shrink-0 items-center justify-center cursor-col-resize outline-none transition-colors group ${
        isResizing
          ? 'bg-accent-100 dark:bg-accent-900/60'
          : 'bg-surface-50 dark:bg-surface-950 hover:bg-accent-100 dark:hover:bg-accent-900/60 focus-visible:bg-accent-100 dark:focus-visible:bg-accent-900/60 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-500'
      }`}
    >
      <span className="h-full w-px bg-surface-200 dark:bg-surface-700 group-hover:bg-accent-400 group-focus-visible:bg-accent-500" />
    </div>
  )
}
