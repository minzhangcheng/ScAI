import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchAndDownload } from '../../utils/download'
import { GraphicAbstract } from './GraphicAbstract'

vi.mock('../../utils/download', async () => {
  const actual = await vi.importActual<typeof import('../../utils/download')>(
    '../../utils/download',
  )
  return { ...actual, fetchAndDownload: vi.fn() }
})

describe('GraphicAbstract contract', () => {
  beforeEach(() => {
    vi.mocked(fetchAndDownload).mockReset().mockResolvedValue(undefined)
  })

  it('uses the report artifact when persisted base64 was stripped', () => {
    render(
      <GraphicAbstract
        sessionId="session-ga"
        graphicAbstract={{
          id: 'ga-1', prompt: 'Summary', imageBase64: '[image]',
          createdAt: '2026-08-28T00:00:00Z',
        }}
      />,
    )

    expect(screen.getByAltText('Graphic Abstract')).toHaveAttribute(
      'src',
      '/api/session/session-ga/report/figure/graphic_abstract.jpg',
    )
    fireEvent.click(screen.getByRole('button', { name: 'Download HD' }))
    expect(fetchAndDownload).toHaveBeenCalledWith(
      '/api/session/session-ga/report/figure/graphic_abstract.jpg',
      'graphic_abstract.jpg',
    )
  })

  it('scopes the fallback asset and download to a restored branch', () => {
    render(
      <GraphicAbstract
        sessionId="session-ga"
        branchId="restore/a b"
        graphicAbstract={{
          id: 'ga-1', prompt: 'Summary', imageBase64: '[image]',
          createdAt: '2026-08-28T00:00:00Z',
        }}
      />,
    )

    expect(screen.getByAltText('Graphic Abstract')).toHaveAttribute(
      'src',
      '/api/session/session-ga/report/figure/graphic_abstract.jpg?branch_id=restore%2Fa%20b',
    )
    fireEvent.click(screen.getByRole('button', { name: 'Download HD' }))
    expect(fetchAndDownload).toHaveBeenCalledWith(
      '/api/session/session-ga/report/figure/graphic_abstract.jpg?branch_id=restore%2Fa%20b',
      'graphic_abstract.jpg',
    )
  })
})
