import { authHeaders } from '../hooks/useApi'

/** Error thrown by fetchAndDownload on a non-OK response, carrying the HTTP status. */
export class FetchError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message)
    this.name = 'FetchError'
  }
}

/**
 * Download a Blob as a file by creating a temporary anchor element.
 * Handles Object URL lifecycle (create + revoke).
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * Fetch a URL with auth, convert the response to a Blob, and trigger a download.
 * Uses the supplied filename unless response filenames are explicitly requested.
 * Throws on non-OK responses.
 */
export async function fetchAndDownload(
  url: string,
  filename: string,
  options?: RequestInit & { preferResponseFilename?: boolean }
): Promise<void> {
  const { preferResponseFilename = false, ...requestOptions } = options ?? {}
  const mergedHeaders = {
    ...authHeaders(),
    ...(requestOptions.headers as Record<string, string> || {}),
  }
  const response = await fetch(url, { ...requestOptions, headers: mergedHeaders })
  if (!response.ok) {
    const errorText = (await response.text().catch(() => 'Unknown error')).slice(0, 1000)
    throw new FetchError(response.status, `Download failed (${response.status}): ${errorText}`)
  }
  const blob = await response.blob()
  const disposition = preferResponseFilename ? response.headers.get('Content-Disposition') : null
  const match = disposition?.match(/(?:^|;)\s*filename\s*=\s*(?:"([^"]+)"|([^;]+))/i)
  const responseFilename = (match?.[1] ?? match?.[2])?.trim()
  downloadBlob(blob, responseFilename || filename)
}
