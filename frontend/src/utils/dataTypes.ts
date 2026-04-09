/** Shared data type constants and helpers for file upload/selection */

export const PATHOLOGY_TYPES = ['H&E', 'Other'] as const
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
