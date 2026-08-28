import { useCallback, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'

import type { DataFile as DataFileType, TreatmentGroup } from '../../stores/appStore'
import { getAllowedExtensions, isBlockedExtension } from './uploadRules'

interface UploadItem {
  id: number
  file: File
  status: 'pending' | 'uploading' | 'done' | 'error'
  progress: number
  groupId?: string
  sampleNumber?: number
  dataType: string
  errorMsg?: string
  abortController?: AbortController
}

let uploadItemCounter = 0

// ============================================================
// DataSection Component (reusable for pathology/sequencing/biochemistry)
// ============================================================

export interface DataSectionProps {
  section: 'pathology' | 'sequencing' | 'biochemistry'
  title: string
  description: string
  icon: any
  iconColor: string
  groups: TreatmentGroup[]
  dataFiles: DataFileType[]
  typeOptions: readonly string[]
  multipleFiles: boolean
  onUpload: (file: File, section: 'pathology' | 'sequencing' | 'biochemistry', dataType: string, groupId?: string, sampleNumber?: number, description?: string, signal?: AbortSignal) => Promise<any>
  onDelete: (fileId: string) => Promise<void>
  language: string
  t: (key: any, params?: any) => string
  children?: React.ReactNode
}

export function useDataUploadQueue({
  section,
  title,
  description,
  icon: Icon,
  iconColor,
  groups,
  dataFiles,
  typeOptions,
  multipleFiles,
  onUpload,
  onDelete,
  language,
  t,
  children,
}: DataSectionProps) {
  const [selectedGroup, setSelectedGroup] = useState<string>('_none')
  const [selectedSample, setSelectedSample] = useState<string>('_none')
  const [selectedType, setSelectedType] = useState(typeOptions[0])
  const [customType, setCustomType] = useState('')
  const [collapsed, setCollapsed] = useState(false)
  const queueRef = useRef<UploadItem[]>([])
  const [, setRenderTick] = useState(0)
  const syncRender = useCallback(() => setRenderTick(t => t + 1), [])
  const processingRef = useRef(false)
  const onUploadRef = useRef(onUpload)
  onUploadRef.current = onUpload
  const selectedGroupObj = groups.find(g => g.id === selectedGroup)
  const maxSamples = selectedGroupObj?.sampleCount || 0
  const isNoGroup = selectedGroup === '_none'
  const actualType = selectedType === 'Other' ? (customType || 'Other') : selectedType
  const sectionFiles = dataFiles.filter(f => f.section === section)
  const visibleQueue = queueRef.current.filter(u => u.status !== 'done')
  const processQueue = useCallback(async () => {
    if (processingRef.current) return
    processingRef.current = true
    while (true) {
      const nextItem = queueRef.current.find(u => u.status === 'pending')
      if (!nextItem) break
      const controller = new AbortController()
      nextItem.abortController = controller
      nextItem.status = 'uploading'
      nextItem.progress = 0
      syncRender()
      const interval = setInterval(() => {
        nextItem.progress = Math.min(nextItem.progress + 10, 90)
        syncRender()
      }, 200)
      try {
        await onUploadRef.current(nextItem.file, section, nextItem.dataType, nextItem.groupId, nextItem.sampleNumber, undefined, controller.signal)
        clearInterval(interval)
        nextItem.status = 'done'
        nextItem.progress = 100
        syncRender()
        const doneId = nextItem.id
        setTimeout(() => {
          queueRef.current = queueRef.current.filter(u => u.id !== doneId)
          syncRender()
        }, 800)
      } catch (e) {
        clearInterval(interval)
        if (controller.signal.aborted) {
          queueRef.current = queueRef.current.filter(u => u.id !== nextItem.id)
          syncRender()
          continue
        }
        nextItem.status = 'error'
        nextItem.errorMsg = e instanceof Error ? e.message : String(e)
        syncRender()
      }
    }
    processingRef.current = false
  }, [section, syncRender])
  const onDrop = useCallback((files: File[]) => {
    if (files.length === 0) return
    const blockedFiles = files.filter(f => isBlockedExtension(f.name))
    if (blockedFiles.length > 0) {
      const names = blockedFiles.map(f => f.name).join(', ')
      alert(language === 'zh'
        ? `不允许上传可执行文件: ${names}`
        : `Executable files not allowed: ${names}`)
    }
    const safeFiles = files.filter(f => !isBlockedExtension(f.name))
    const allowedExts = getAllowedExtensions(section, selectedType)
    const validFiles = allowedExts.length === 0
      ? safeFiles
      : safeFiles.filter(f => {
          const ext = '.' + (f.name.split('.').pop()?.toLowerCase() || '')
          return allowedExts.includes(ext)
        })
    const rejectedFiles = safeFiles.filter(f => !validFiles.includes(f))
    if (rejectedFiles.length > 0) {
      const names = rejectedFiles.map(f => f.name).join(', ')
      alert(language === 'zh'
        ? `不支持的文件格式: ${names}\n支持: ${allowedExts.join(', ')}`
        : `Unsupported format: ${names}\nSupported: ${allowedExts.join(', ')}`)
    }
    if (validFiles.length === 0) return
    const groupId = isNoGroup ? undefined : selectedGroup
    const sampleNum = isNoGroup || selectedSample === '_none' ? undefined : parseInt(selectedSample)
    const newItems: UploadItem[] = validFiles.map(file => ({
      id: ++uploadItemCounter,
      file,
      status: 'pending' as const,
      progress: 0,
      groupId,
      sampleNumber: sampleNum,
      dataType: actualType,
    }))
    queueRef.current = [...queueRef.current, ...newItems]
    syncRender()
    if (!processingRef.current) {
      processQueue()
    }
  }, [section, selectedType, actualType, selectedGroup, selectedSample, isNoGroup, language, syncRender, processQueue])
  const allowedExts = getAllowedExtensions(section, selectedType)
  const acceptAttr = allowedExts.length > 0 ? allowedExts.join(',') : undefined
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: multipleFiles,
  })
  const getGroupName = (groupId?: string) => {
    if (!groupId) return t('noGroupDistinction')
    return groups.find(g => g.id === groupId)?.name || groupId
  }
  return {
    acceptAttr, actualType, allowedExts, children, collapsed, customType, dataFiles,
    description, getGroupName, getInputProps, getRootProps, groups, icon: Icon, iconColor,
    isDragActive, isNoGroup, language, maxSamples, onDelete, onDrop, queueRef, section,
    sectionFiles, selectedGroup, selectedSample, selectedType, setCollapsed, setCustomType,
    setSelectedGroup, setSelectedSample, setSelectedType, syncRender, t, title, typeOptions,
    visibleQueue,
  }
}

export type DataSectionController = ReturnType<typeof useDataUploadQueue>
