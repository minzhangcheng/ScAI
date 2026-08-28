import { DataSectionView } from './DataSectionView'
import { useDataUploadQueue, type DataSectionProps } from './useDataUploadQueue'

export function DataSection(props: DataSectionProps) {
  return <DataSectionView controller={useDataUploadQueue(props)} />
}
