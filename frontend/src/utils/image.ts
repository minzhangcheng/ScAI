import { STRIPPED_FIGURE_PLACEHOLDER } from '../constants/storage'

/**
 * Return only image payloads that can be rendered as base64 data URIs.
 * Persisted sessions replace binary payloads with a sentinel and may retain a
 * blob-store handle; both must fall back to the authenticated image endpoint.
 */
export function normalizeImageBase64(value: string | null | undefined): string | undefined {
  const normalized = typeof value === 'string' ? value.trim() : ''
  if (
    !normalized
    || normalized === STRIPPED_FIGURE_PLACEHOLDER
    || normalized.startsWith('@scai-blob:')
  ) {
    return undefined
  }
  return normalized
}

/**
 * Detect MIME type from base64-encoded image data by inspecting magic byte prefixes.
 */
export function detectBase64MimeType(base64: string): string {
  if (!base64) return 'image/png'
  if (base64.startsWith('/9j/')) return 'image/jpeg'
  if (base64.startsWith('R0lG')) return 'image/gif'
  if (base64.startsWith('UklG')) return 'image/webp'
  // Default to PNG (iVBOR prefix and others)
  return 'image/png'
}

/**
 * Build a data URI from base64 image content.
 */
export function base64ToDataUri(base64: string): string {
  return `data:${detectBase64MimeType(base64)};base64,${base64}`
}
