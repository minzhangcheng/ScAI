import { DEFAULT_SIDEBAR_WIDTH } from '../constants/layout'
import { SidebarView } from './sidebar/SidebarView'
import { useSidebarSessions } from './sidebar/useSidebarSessions'

interface SidebarProps {
  isOpen: boolean
  onOpenConfig: () => void
  width?: number
}

export function Sidebar({ isOpen, onOpenConfig, width = DEFAULT_SIDEBAR_WIDTH }: SidebarProps) {
  const controller = useSidebarSessions()
  return <SidebarView controller={controller} isOpen={isOpen} onOpenConfig={onOpenConfig} width={width} />
}
