import { authHeaders } from '../hooks/useApi'

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
 * Throws on non-OK responses.
 */
export async function fetchAndDownload(
  url: string,
  filename: string,
  options?: RequestInit
): Promise<void> {
  const mergedHeaders = {
    ...authHeaders(),
    ...(options?.headers as Record<string, string> || {}),
  }
  const response = await fetch(url, { ...options, headers: mergedHeaders })
  if (!response.ok) {
    const errorText = (await response.text().catch(() => 'Unknown error')).slice(0, 1000)
    throw new Error(`Download failed (${response.status}): ${errorText}`)
  }
  const blob = await response.blob()
  downloadBlob(blob, filename)
}
