import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SidebarResizeHandle } from '../SidebarResizeHandle'
import {
  DEFAULT_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
  SIDEBAR_WIDTH_STEP,
} from '../../constants/layout'

describe('SidebarResizeHandle', () => {
  it('supports keyboard, drag, clamp, mouseup, and reset', () => {
    const onWidthChange = vi.fn()
    render(
      <SidebarResizeHandle
        width={DEFAULT_SIDEBAR_WIDTH}
        onWidthChange={onWidthChange}
        label="Resize sidebar"
      />,
    )

    const handle = screen.getByRole('separator', { name: 'Resize sidebar' })
    expect(handle).toHaveAttribute('aria-valuemin', String(MIN_SIDEBAR_WIDTH))
    expect(handle).toHaveAttribute('aria-valuemax', String(MAX_SIDEBAR_WIDTH))
    expect(handle).toHaveAttribute('aria-valuenow', String(DEFAULT_SIDEBAR_WIDTH))

    fireEvent.keyDown(handle, { key: 'ArrowRight' })
    expect(onWidthChange).toHaveBeenLastCalledWith(DEFAULT_SIDEBAR_WIDTH + SIDEBAR_WIDTH_STEP)

    fireEvent.keyDown(handle, { key: 'ArrowLeft' })
    expect(onWidthChange).toHaveBeenLastCalledWith(DEFAULT_SIDEBAR_WIDTH - SIDEBAR_WIDTH_STEP)

    fireEvent.keyDown(handle, { key: 'Home' })
    expect(onWidthChange).toHaveBeenLastCalledWith(MIN_SIDEBAR_WIDTH)

    fireEvent.keyDown(handle, { key: 'End' })
    expect(onWidthChange).toHaveBeenLastCalledWith(MAX_SIDEBAR_WIDTH)

    fireEvent.mouseDown(handle, { button: 0, clientX: DEFAULT_SIDEBAR_WIDTH })
    fireEvent.mouseMove(document, { clientX: MAX_SIDEBAR_WIDTH + 100 })
    expect(onWidthChange).toHaveBeenLastCalledWith(MAX_SIDEBAR_WIDTH)

    fireEvent.mouseUp(document)
    onWidthChange.mockClear()
    fireEvent.mouseMove(document, { clientX: MIN_SIDEBAR_WIDTH })
    expect(onWidthChange).not.toHaveBeenCalled()

    fireEvent.doubleClick(handle)
    expect(onWidthChange).toHaveBeenCalledWith(DEFAULT_SIDEBAR_WIDTH)
  })
})
