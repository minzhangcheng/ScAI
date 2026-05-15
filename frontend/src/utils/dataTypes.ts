/** Shared data type constants and helpers for file upload/selection */

/** Sentinel value indicating the user should provide a custom type name */
export const CUSTOM_TYPE_SENTINEL = 'Other'

export const PATHOLOGY_TYPES = ['H&E', CUSTOM_TYPE_SENTINEL] as const
export const SEQUENCING_TYPES = ['scRNA-seq', 'bulk RNA-seq', 'Loom'] as const
export const BIOCHEMISTRY_TYPES = ['image', 'file'] as const

/** Map agent name to its data section identifier */
export function getSection(agent: string): string | null {
  if (agent === 'pathology') return 'pathology'
  if (agent === 'bioinformatics') return 'sequencing'
  if (agent === 'biochemistry') return 'biochemistry'
  return null
}

/** Get file type options for a given agent */
export function getTypeOptions(agent: string): readonly string[] {
  if (agent === 'pathology') return PATHOLOGY_TYPES
  if (agent === 'bioinformatics') return SEQUENCING_TYPES
  if (agent === 'biochemistry') return BIOCHEMISTRY_TYPES
  return []
}

/** True if dataType is the sentinel or a user-entered value not in typeOptions. */
export function isCustomDataType(dataType: string | undefined, typeOptions: readonly string[]): boolean {
  if (!dataType) return false
  return dataType === CUSTOM_TYPE_SENTINEL || !typeOptions.includes(dataType)
}

/** True if the type options include the custom sentinel (i.e. support user-entered values). */
export function hasCustomOption(typeOptions: readonly string[]): boolean {
  return typeOptions.includes(CUSTOM_TYPE_SENTINEL)
}

/** Derive the <select> value: the current dataType if predefined, otherwise CUSTOM_TYPE_SENTINEL. */
export function selectValueForDataType(dataType: string | undefined, typeOptions: readonly string[]): string {
  if (!dataType) return typeOptions[0] || ''
  return typeOptions.includes(dataType) ? dataType : CUSTOM_TYPE_SENTINEL
}
