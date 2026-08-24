// Reusable upload section for pathology / sequencing / biochemistry, with the upload queue.
// Split out of frontend/src/components/DataPanel.tsx verbatim.

import { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { TreatmentGroup, DataFile as DataFileType } from '../../stores/appStore'
import { Upload, File, Trash2, Download, ChevronDown, ChevronUp, X } from 'lucide-react'
import { getAllowedExtensions, isBlockedExtension } from './uploadRules'

// ============================================================
// Upload queue types
// ============================================================
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
export function DataSection({
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
}: {
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
}) {
  const [selectedGroup, setSelectedGroup] = useState<string>('_none')
  const [selectedSample, setSelectedSample] = useState<string>('_none')
  const [selectedType, setSelectedType] = useState(typeOptions[0])
  const [customType, setCustomType] = useState('')
  const [collapsed, setCollapsed] = useState(false)

  // Upload queue: ref is the source of truth, renderTick triggers re-renders
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

  // Queue processor: picks the next pending item and uploads it
  const processQueue = useCallback(async () => {
    if (processingRef.current) return
    processingRef.current = true

    while (true) {
      const nextItem = queueRef.current.find(u => u.status === 'pending')
      if (!nextItem) break

      // Create AbortController for cancellation support
      const controller = new AbortController()
      nextItem.abortController = controller

      // Mark as uploading
      nextItem.status = 'uploading'
      nextItem.progress = 0
      syncRender()

      // Simulate progress
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
        // Remove done item after a short delay
        const doneId = nextItem.id
        setTimeout(() => {
          queueRef.current = queueRef.current.filter(u => u.id !== doneId)
          syncRender()
        }, 800)
      } catch (e) {
        clearInterval(interval)
        if (controller.signal.aborted) {
          // User cancelled — remove from queue and continue to next
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

  // Manual extension validation (avoids react-dropzone silent rejection of
  // scientific file formats like .h5/.h5ad/.loom whose MIME types browsers don't recognize)
  const onDrop = useCallback((files: File[]) => {
    if (files.length === 0) return

    // Reject dangerous file types regardless of data type
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

    // Start processing if not already running
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

  return (
    <section className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-800 cursor-pointer select-none"
        onClick={() => setCollapsed(!collapsed)}
      >
        <div className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${iconColor}`} />
          <div>
            <h3 className="font-display text-base font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">{description}</p>
          </div>
          {sectionFiles.length > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-full text-xs font-medium">
              {sectionFiles.length}
            </span>
          )}
        </div>
        {collapsed ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronUp className="w-4 h-4 text-slate-400" />}
      </div>

      {!collapsed && (
        <div className="p-4 space-y-4">
          {/* Dropdowns row */}
          <div className="flex flex-wrap items-end gap-3">
            {/* Group dropdown */}
            <div className="min-w-[140px]">
              <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">{t('selectGroup')}</label>
              <select
                value={selectedGroup}
                onChange={(e) => {
                  setSelectedGroup(e.target.value)
                  if (e.target.value === '_none') setSelectedSample('_none')
                }}
                className="w-full px-2 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm bg-white dark:bg-surface-900"
              >
                <option value="_none">{t('noGroupDistinction')}</option>
                {groups.map(g => (
                  <option key={g.id} value={g.id}>{g.name}</option>
                ))}
              </select>
            </div>

            {/* Sample dropdown */}
            <div className="min-w-[120px]">
              <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">{t('selectSample')}</label>
              <select
                value={selectedSample}
                onChange={(e) => setSelectedSample(e.target.value)}
                disabled={isNoGroup}
                className="w-full px-2 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm bg-white dark:bg-surface-900 disabled:opacity-50 disabled:bg-slate-100 dark:disabled:bg-slate-800"
              >
                <option value="_none">{t('noSampleDistinction')}</option>
                {!isNoGroup && Array.from({ length: maxSamples }, (_, i) => (
                  <option key={i + 1} value={String(i + 1)}>
                    {t('sample')} {i + 1}
                  </option>
                ))}
              </select>
            </div>

            {/* Data type dropdown */}
            <div className="min-w-[160px]">
              <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">{t('selectDataType')}</label>
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="w-full px-2 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm bg-white dark:bg-surface-900"
              >
                {typeOptions.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>

            {/* Custom type input for "Other" */}
            {selectedType === 'Other' && (
              <div className="min-w-[140px]">
                <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">{t('customType')}</label>
                <input
                  type="text"
                  value={customType}
                  onChange={(e) => setCustomType(e.target.value)}
                  placeholder={
                    section === 'sequencing'
                      ? (language === 'zh' ? '如 spatial transcriptomics' : 'e.g. spatial transcriptomics')
                      : (language === 'zh' ? '如 Sirius Red' : 'e.g. Sirius Red')
                  }
                  className="w-full px-2 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg text-sm"
                />
              </div>
            )}
          </div>

          {/* Upload Area */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
              isDragActive
                ? 'border-accent-500 bg-accent-50 dark:bg-accent-900/40'
                : 'border-slate-300 dark:border-slate-600 hover:border-accent-400 hover:bg-slate-50 dark:hover:bg-slate-800'
            }`}
          >
            <input {...getInputProps()} accept={acceptAttr} />
            <Upload className="w-7 h-7 text-slate-400 mx-auto mb-1.5" />
            <p className="text-slate-600 dark:text-slate-300 font-medium text-sm">
              {isDragActive
                ? (language === 'zh' ? '释放以上传' : 'Release to upload')
                : t('dragOrClick')}
            </p>
            {allowedExts.length > 0 && (
              <p className="text-xs text-slate-400 mt-1">
                {language === 'zh' ? '支持格式: ' : 'Supported: '}{allowedExts.join(', ')}
              </p>
            )}
          </div>

          {/* Upload Queue */}
          {visibleQueue.length > 0 && (
            <div className="space-y-2">
              {visibleQueue.map(item => (
                <div key={item.id} className="flex items-center gap-3 px-3 py-2 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <File className="w-4 h-4 text-slate-400 flex-shrink-0" />
                  <span className="text-sm text-slate-700 dark:text-slate-200 truncate flex-1 min-w-0">{item.file.name}</span>
                  {item.status === 'uploading' && (
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <div className="w-24 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                        <div
                          className="bg-accent-500 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${item.progress}%` }}
                        />
                      </div>
                      <span className="text-xs text-accent-600 dark:text-accent-400 font-medium w-8">{Math.round(item.progress)}%</span>
                    </div>
                  )}
                  {item.status === 'pending' && (
                    <span className="text-xs px-2 py-0.5 bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400 rounded-full flex-shrink-0">
                      {language === 'zh' ? '等待中' : 'Pending'}
                    </span>
                  )}
                  {item.status === 'error' && (
                    <span className="text-xs px-2 py-0.5 bg-red-100 dark:bg-red-900/60 text-red-600 dark:text-red-400 rounded-full flex-shrink-0" title={item.errorMsg}>
                      {language === 'zh' ? '失败' : 'Failed'}
                    </span>
                  )}
                  {item.status !== 'done' && (
                    <button
                      onClick={() => {
                        if (item.status === 'uploading' && item.abortController) {
                          item.abortController.abort()
                        } else {
                          queueRef.current = queueRef.current.filter(u => u.id !== item.id)
                          syncRender()
                        }
                      }}
                      className="p-0.5 text-slate-400 hover:text-red-500 transition-colors flex-shrink-0"
                      title={language === 'zh' ? '取消' : 'Cancel'}
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Extra children (e.g. indicator form) */}
          {children}

          {/* File List Table */}
          {sectionFiles.length > 0 && (
            <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 dark:bg-slate-800">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium text-slate-600 dark:text-slate-300">{t('filename')}</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600 dark:text-slate-300">{t('group')}</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600 dark:text-slate-300">{t('sample')}</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600 dark:text-slate-300">{t('dataType')}</th>
                    <th className="px-3 py-2 text-right font-medium text-slate-600 dark:text-slate-300 w-20">{t('actions')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {sectionFiles.map(file => (
                    <tr key={file.id} className="hover:bg-slate-50 dark:hover:bg-slate-800">
                      <td className="px-3 py-2 text-slate-900 dark:text-slate-100">
                        <div className="flex items-center gap-1.5">
                          <File className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
                          <span className="truncate max-w-[200px]">{file.filename}</span>
                        </div>
                      </td>
                      <td className="px-3 py-2 text-slate-600 dark:text-slate-300 text-xs">
                        {getGroupName(file.groupId)}
                      </td>
                      <td className="px-3 py-2 text-slate-600 dark:text-slate-300 text-xs">
                        {file.sampleNumber ? `${t('sample')} ${file.sampleNumber}` : '-'}
                      </td>
                      <td className="px-3 py-2">
                        <span className="px-1.5 py-0.5 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 rounded text-xs font-medium">
                          {file.dataType}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <div className="flex items-center justify-end gap-1">
                          {file.filePath && (
                            <a
                              href={`/api/download?path=${encodeURIComponent(file.filePath)}`}
                              download={file.filename}
                              className="p-1 text-slate-400 hover:text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950/40 rounded transition-colors"
                              title={language === 'zh' ? '下载' : 'Download'}
                            >
                              <Download className="w-3.5 h-3.5" />
                            </a>
                          )}
                          <button
                            onClick={() => onDelete(file.id)}
                            className="p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950/40 rounded transition-colors"
                            title={language === 'zh' ? '删除' : 'Delete'}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
