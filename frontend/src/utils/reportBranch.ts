/** Query suffix shared by report and publication-artifact entrypoints. */
export function reportBranchQuery(branchId?: string | null): string {
  return branchId ? `?branch_id=${encodeURIComponent(branchId)}` : ''
}

/** Append a lineage without disturbing an existing query string. */
export function withReportBranch(url: string, branchId?: string | null): string {
  if (!branchId || /[?&]branch_id=/.test(url)) return url
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}branch_id=${encodeURIComponent(branchId)}`
}
