import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useDataUploadQueue, type DataSectionProps } from './useDataUploadQueue'

function props(onUpload: DataSectionProps['onUpload']): DataSectionProps {
  return {
    section: 'pathology',
    title: 'Pathology',
    description: 'Images',
    icon: () => null,
    iconColor: 'text-blue-500',
    groups: [],
    dataFiles: [],
    typeOptions: ['image'],
    multipleFiles: true,
    onUpload,
    onDelete: vi.fn(),
    language: 'en',
    t: key => String(key),
  }
}

describe('DataSection upload queue', () => {
  beforeEach(() => vi.stubGlobal('alert', vi.fn()))

  it('uploads accepted files sequentially', async () => {
    const releases: Array<() => void> = []
    const onUpload = vi.fn((..._args: Parameters<DataSectionProps['onUpload']>) => (
      new Promise<void>(resolve => releases.push(resolve))
    ))
    const { result } = renderHook(() => useDataUploadQueue(props(onUpload)))

    act(() => result.current.onDrop([
      new File(['a'], 'first.png', { type: 'image/png' }),
      new File(['b'], 'second.jpg', { type: 'image/jpeg' }),
    ]))

    await waitFor(() => expect(onUpload).toHaveBeenCalledTimes(1))
    expect(onUpload.mock.calls[0][0].name).toBe('first.png')
    await act(async () => releases[0]())
    await waitFor(() => expect(onUpload).toHaveBeenCalledTimes(2))
    expect(onUpload.mock.calls[1][0].name).toBe('second.jpg')
    await act(async () => releases[1]())
  })

  it('keeps blocked and unsupported drops out of the queue', () => {
    const onUpload = vi.fn()
    const { result } = renderHook(() => useDataUploadQueue(props(onUpload)))

    act(() => result.current.onDrop([
      new File(['x'], 'payload.exe'),
      new File(['x'], 'notes.txt'),
    ]))

    expect(onUpload).not.toHaveBeenCalled()
    expect(result.current.visibleQueue).toEqual([])
    expect(alert).toHaveBeenCalledTimes(2)
  })
})
