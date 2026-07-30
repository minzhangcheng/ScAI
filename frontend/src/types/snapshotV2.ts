export interface ObjectRef {
  sha256: string
  size: number
  media_type: string
}

export interface WorkspaceEntry {
  path: string
  object_hash: string
  size: number
  mode: number
  kind: 'file' | 'empty_dir'
}

export interface GraphNodeCapsule {
  kind: 'graph_node'
  graph_version: string
  next_node: string
  workflow_state_ref: ObjectRef
  planning_stage: 'stage1' | 'stage2' | 'full'
}

export interface SemanticGateCapsule {
  kind: 'semantic_gate'
  graph_version: string
  gate: string
  wait_reason?: string | null
  payload_ref: ObjectRef
}

export interface TerminalCapsule {
  kind: 'terminal'
  graph_version: string
  status: 'completed' | 'stopped'
}

export interface IdleCapsule {
  kind: 'idle'
  graph_version: string
}

export type ContinuationCapsule =
  | GraphNodeCapsule
  | SemanticGateCapsule
  | TerminalCapsule
  | IdleCapsule

export interface RuntimeDependencyEntry {
  name: string
  version: string
  type: string
}

/** Mirrors backend `RuntimeDependency`. */
export interface RuntimeDependency {
  name: string
  serializer: 'dill' | 'h5ad'
  type_name?: string
  version?: string
}

/** Mirrors backend `RuntimeSkip`. */
export interface RuntimeSkip {
  name: string
  type_name?: string
  reason?: string
}

/** Mirrors backend `RuntimeDependencyManifest`. */
export interface RuntimeDependencyManifest {
  required: Record<string, RuntimeDependency>
  incidental_skips: RuntimeSkip[]
}

/** Mirrors backend `RuntimeManifest` — the runtime proof the manifest carries. */
export interface RuntimeManifest {
  status: 'complete' | 'missing' | 'state_only' | 'corrupt'
  dependencies: RuntimeDependencyManifest
  residual_ref?: ObjectRef | null
  captured_residual_names?: string[]
  adata_refs?: Record<string, ObjectRef>
}

export interface RuntimeGenerationManifest {
  capture_command: string
  capture_timestamp: string
  kernel_id: string
  captured_adata_vars?: string[]
  captured_residual_vars?: string[]
  captured_adata_refs?: Record<string, ObjectRef>
  captured_residual_ref?: ObjectRef | null
  warnings?: string[]
}

export interface SnapshotManifestV2 {
  format_version: 2
  session_id: string
  snapshot_id: string
  captured_at: string
  branch: string
  generation: number
  parent_id: string | null
  root_hash: string
  capsule: ContinuationCapsule
  workspace: WorkspaceEntry[]
  objects: Record<string, ObjectRef>
  /**
   * The backend serializes the runtime proof under `runtime`. `runtime_dependencies`
   * exists only on `SnapshotCaptureRequest` as a legacy bridge field and is never
   * part of a serialized manifest, so every consumer that read it short-circuited
   * permanently — and its declared shape (`{python, additional}`) did not match
   * the backend's either, so populating it would have crashed the render.
   */
  runtime?: RuntimeManifest | null
  runtime_snapshot_capture?: RuntimeGenerationManifest | null
}

export interface SnapshotFailureRecord {
  snapshot_id: string
  session_id: string
  failed_at: string
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface SnapshotRecordV2 {
  snapshot_id: string
  manifest: SnapshotManifestV2
  failure: SnapshotFailureRecord | null
}

export interface SnapshotStoreStatus {
  logical_bytes: number
  physical_bytes: number
  dedup_savings: number
  object_count: number
  manifest_count: number
  /**
   * Per-session ceiling the store is measured against, in bytes.
   *
   * The quota trio is optional because the backend resolves it and older
   * responses (and fixtures built before it existed) simply omit it — a panel
   * that assumed it present would render "NaN%" against them rather than
   * staying quiet.
   */
  quota_bytes?: number
  /** physical_bytes / quota_bytes. Not capped at 1: a store can sit above. */
  quota_usage_ratio?: number
  /**
   * Set by the backend once usage reaches the advisory fraction of the quota.
   * The threshold lives there, with the ceiling that enforces it, so the client
   * must not re-derive one from the ratio.
   */
  quota_warning?: boolean
}

export interface SnapshotTransactionRecord {
  transaction_id: string
  session_id: string
  snapshot_id: string
  status: 'queued' | 'waiting_safe_point' | 'staging' | 'staged' | 'applying' | 'committed' | 'failed' | 'cancelled' | 'timed_out'
  recovery_mode: 'rollback' | 'forward_only'
  failure: Record<string, unknown> | null
  retry_of_transaction_id?: string | null
  /** Backend may signal client reconnect after commit */
  reconnect?: boolean
  /**
   * Client-only: the retry monitor gave up polling before the transaction
   * reached a terminal status. The server record is unaffected — reopening the
   * Snapshot panel resumes tracking.
   */
  poll_timed_out?: boolean
  /** Present on `failed`; `code` drives the client's recovery offer. */
  failure_code?: string
}

export interface RestoreSnapshotV2Request {
  snapshot_id: string
  runtime_choice?: 'reset' | 'keep' | null
  current_state_choice?: 'abort' | 'discard_current'
  discard_confirmation?: boolean
}

export interface SnapshotTransactionResponse {
  transaction_id: string
  status: string
  status_url: string
  cancel_url: string
}

export type SnapshotFailureAction =
  | 'retry'
  // Offered only by the memory-preflight gate, in place of retry: capture
  // anyway, accepting the risk that the copy exhausts memory.
  | 'proceed'
  | 'skip'
  | 'manage'
  | 'terminate'

export interface SnapshotFailureInfo {
  failure: SnapshotFailureRecord
  allowed_actions: SnapshotFailureAction[]
}

export interface DeleteSnapshotV2Result {
  deleted_ids: string[]
  freed_bytes: number
  surviving_count: number
}

export interface ExportV2Result {
  status: string
  download_url: string
  filename: string
  size_bytes: number
}

export interface ImportV2Result {
  success: boolean
  dest_session_id: string
  imported_snapshot_ids: string[]
  committed_count: number
  skipped_runtime_snapshot_ids: string[]
  errors: string[]
  warnings: string[]
}

export interface SnapshotV2ListResult {
  snapshots: SnapshotRecordV2[]
  store_status: SnapshotStoreStatus
}

export interface SnapshotV2PreviewResult {
  manifest: SnapshotManifestV2
}
