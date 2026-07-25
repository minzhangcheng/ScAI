// Allowed/blocked upload extensions per section and data type.
// Split out of frontend/src/components/DataPanel.tsx; SEQUENCING_TYPES has since
// been rewired to derive from utils/dataTypes instead of restating it.

import { SEQUENCING_TYPES as PREDEFINED_SEQUENCING_TYPES, CUSTOM_TYPE_SENTINEL } from '../../utils/dataTypes'

// The upload form offers the predefined sequencing types plus the custom sentinel, so a
// user can name a format the list does not cover. Derived rather than spelled out: the
// two lists were maintained separately and had already drifted.
export const SEQUENCING_TYPES = [...PREDEFINED_SEQUENCING_TYPES, CUSTOM_TYPE_SENTINEL] as const

// Allowed extensions per section + data type (used for manual validation instead of
// react-dropzone accept filter, which silently rejects files with unrecognized MIME types
// like .h5, .h5ad, .loom)
// Dangerous file extensions that should never be uploaded regardless of data type
export const BLOCKED_EXTENSIONS = new Set([
  '.exe', '.bat', '.cmd', '.sh', '.ps1', '.vbs', '.js', '.msi',
  '.dll', '.so', '.dylib', '.bin', '.com', '.scr', '.pif',
])

export function getAllowedExtensions(section: string, dataType: string): string[] {
  if (section === 'pathology') return ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg']
  if (section === 'sequencing') {
    if (dataType === 'Other') return []  // allow-all mode, blocked extensions checked separately
    if (dataType === 'Loom') return ['.loom']
    if (dataType === 'bulk RNA-seq') return ['.csv', '.tsv', '.txt']
    return ['.h5ad', '.h5']
  }
  if (section === 'biochemistry') {
    if (dataType === 'image') return ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg']
    return ['.csv', '.txt', '.tsv', '.xls', '.xlsx']
  }
  return []
}

export function isBlockedExtension(filename: string): boolean {
  const ext = '.' + (filename.split('.').pop()?.toLowerCase() || '')
  return BLOCKED_EXTENSIONS.has(ext)
}
