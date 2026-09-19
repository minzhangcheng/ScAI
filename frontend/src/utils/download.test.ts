import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchAndDownload } from './download'

vi.mock('../hooks/useApi', () => ({ authHeaders: () => ({ Authorization: 'Bearer test-token' }) }))

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('download response filenames', () => {
  it.each([
    { optIn: true, disposition: 'attachment; filename="stage1_review_v2.html"', expected: 'stage1_review_v2.html' },
    { optIn: false, disposition: 'attachment; filename="stage1_review_v2.html"', expected: 'fallback.html' },
    { optIn: true, disposition: null, expected: 'fallback.html' },
  ])('uses $expected with opt-in $optIn', async ({ optIn, disposition, expected }) => {
    const headers = new Headers()
    if (disposition) headers.set('Content-Disposition', disposition)
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, headers, blob: async () => new Blob(['report']),
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:report'), revokeObjectURL: vi.fn() })
    let downloadedFilename = ''
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      downloadedFilename = this.download
    })

    await fetchAndDownload('/api/session/s/stage-review-report/v2/html', 'fallback.html',
      optIn ? { preferResponseFilename: true } : undefined)

    expect(downloadedFilename).toBe(expected)
    expect(fetchMock).toHaveBeenCalledWith('/api/session/s/stage-review-report/v2/html', {
      headers: { Authorization: 'Bearer test-token' },
    })
  })
})
