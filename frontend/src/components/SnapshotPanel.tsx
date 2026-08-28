import { SnapshotPanelView } from './snapshots/SnapshotPanelView'
import {
  useSnapshotPanelController,
  type SnapshotPanelProps,
} from './snapshots/useSnapshotPanelController'

export function SnapshotPanel(props: SnapshotPanelProps) {
  return <SnapshotPanelView controller={useSnapshotPanelController(props)} />
}
