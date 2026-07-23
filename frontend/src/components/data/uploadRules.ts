// Allowed/blocked upload extensions per section and data type.
// Split out of frontend/src/components/DataPanel.tsx verbatim.



export const SEQUENCING_TYPES = ['scRNA-seq', 'bulk RNA-seq', 'Loom', 'Other'] as const  // DataPanel adds 'Other' for upload flexibility

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
