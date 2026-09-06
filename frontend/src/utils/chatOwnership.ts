/** Normalize the branch component used by client-side chat ownership checks. */
export function normalizeBranchId(value: unknown): string {
  if (typeof value !== 'string') return 'main'
  const branchId = value.trim()
  return branchId || 'main'
}

/** Stable key for transient chat state owned by one session branch. */
export function chatOwnerKey(
  sessionId: string | null | undefined,
  branchId: unknown,
): string | null {
  if (!sessionId) return null
  return `${sessionId}\u0000${normalizeBranchId(branchId)}`
}
