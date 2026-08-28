export interface SidebarProgressInfo {
  action: string
  detail: string
  percent?: number
  closable?: boolean
  success?: boolean
  onConfirm?: () => void
  onCancel?: () => void
  cancelLabel?: string
}
