import { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { useAppStore, TreatmentGroup, DataFile as DataFileType, BiochemistryIndicator } from '../stores/appStore'
import { useSession } from '../hooks/useApi'
import { useTranslation } from '../i18n/translations'
import {
  Upload,
  FlaskConical,
  Plus,
  Check,
  File,
  Trash2,
  Download,
  ChevronDown,
  ChevronUp,
  Users,
  Grid3X3,
  Microscope,
  Dna,
  Pencil,
  X,
} from 'lucide-react'

// --- Data type options ---
import { PATHOLOGY_TYPES, BIOCHEMISTRY_TYPES } from '../utils/dataTypes'
const SEQUENCING_TYPES = ['scRNA-seq', 'bulk RNA-seq', 'Loom', 'Other'] as const  // DataPanel adds 'Other' for upload flexibility

// Allowed extensions per section + data type (used for manual validation instead of
// react-dropzone accept filter, which silently rejects files with unrecognized MIME types
// like .h5, .h5ad, .loom)
// Dangerous file extensions that should never be uploaded regardless of data type
const BLOCKED_EXTENSIONS = new Set([
  '.exe', '.bat', '.cmd', '.sh', '.ps1', '.vbs', '.js', '.msi',
  '.dll', '.so', '.dylib', '.bin', '.com', '.scr', '.pif',
])

function getAllowedExtensions(section: string, dataType: string): string[] {
  if (section === 'pathology') return ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg']
  if (section === 'sequencing') {
    if (dataType === 'Other') return []  // allow-all mode, blocked extensions checked separately
    if (dataType === 'Loom') return ['.loom']
    if (dataType === 'bulk RNA-seq') return ['.csv', '.tsv', '.txt']
    return ['.h5ad', '.h5']
  }
  if (section === 'biochemistry') {
    if (dataType === 'image') return ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg']
    return ['.csv', '.txt', '.tsv', '.xls', '.xlsx']
  }
  return []
}

function isBlockedExtension(filename: string): boolean {
  const ext = '.' + (filename.split('.').pop()?.toLowerCase() || '')
  return BLOCKED_EXTENSIONS.has(ext)
}

// ============================================================
// GroupManager Component
// ============================================================
function GroupManager({
  groups,
  onAdd,
  onUpdate,
  onDelete,
  language,
  t,
}: {
  groups: TreatmentGroup[]
  onAdd: (name: string, description: string, sampleCount: number) => Promise<any>
  onUpdate: (id: string, updates: { name?: string; description?: string; sampleCount?: number }) => Promise<any>
  onDelete: (id: string) => Promise<void>
  language: string
  t: (key: any, params?: any) => string
}) {
  const [adding, setAdding] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newSampleCount, setNewSampleCount] = useState(1)
  const [editingId, setEditingId] = useState<string | null>(null)

  const handleAdd = async () => {
    if (!newName.trim()) return
    try {
      await onAdd(newName.trim(), newDesc.trim(), newSampleCount)
      setNewName('')
      setNewDesc('')
      setNewSampleCount(1)
      setAdding(false)
    } catch (e) {
      console.error('Failed to add group:', e)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm(t('deleteGroupConfirm'))) return
    try {
      await onDelete(id)
    } catch (e) {
      console.error('Failed to delete group:', e)
    }
  }

  return (
    <section className="mb-8">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5 text-accent-500" />
          <div>
            <h3 className="font-display text-lg font-semibold text-slate-900">{t('treatmentGroups')}</h3>
            <p className="text-sm text-slate-500">{t('treatmentGroupsDesc')}</p>
          </div>
        </div>
        <button
          onClick={() => setAdding(true)}
          className="flex items-center gap-1.5 px-3 py-2 bg-accent-100 text-accent-700 rounded-lg text-sm font-medium hover:bg-accent-200 transition-colors"
        >
          <Plus className="w-4 h-4" />
          {t('addGroup')}
        </button>
      </div>

      {/* Group Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {groups.map((group) => (
          <div
            key={group.id}
            className="border border-slate-200 rounded-xl p-4 bg-white hover:border-accent-200 transition-colors"
          >
            {editingId === group.id ? (
              <div className="space-y-2">
                <input
                  type="text"
                  defaultValue={group.name}
                  onBlur={(e) => {
                    const val = e.target.value.trim()
                    if (val && val !== group.name) onUpdate(group.id, { name: val })
                  }}
                  className="w-full px-2 py-1 border border-slate-300 rounded text-sm font-medium"
                  autoFocus
                />
                <input
                  type="text"
                  defaultValue={group.description}
                  onBlur={(e) => {
                    const val = e.target.value.trim()
                    if (val !== group.description) onUpdate(group.id, { description: val })
                  }}
                  placeholder={t('groupDescPlaceholder')}
                  className="w-full px-2 py-1 border border-slate-300 rounded text-xs"
                />
                <div className="flex items-center gap-2">
                  <label className="text-xs text-slate-500">{t('sampleCount')}:</label>
                  <input
                    type="number"
                    defaultValue={group.sampleCount}
                    min={1}
                    max={100}
                    onBlur={async (e) => {
                      const val = parseInt(e.target.value) || 1
                      if (val !== group.sampleCount) {
                        try {
                          await onUpdate(group.id, { sampleCount: val })
                        } catch (err: any) {
                          alert(err.message || (language === 'zh' ? '更新失败' : 'Update failed'))
                          e.target.value = String(group.sampleCount)
                        }
                      }
                    }}
                    className="w-16 px-2 py-1 border border-slate-300 rounded text-xs"
                  />
                </div>
                <button
                  onClick={() => setEditingId(null)}
                  className="text-xs text-accent-600 hover:underline"
                >
                  {language === 'zh' ? '完成' : 'Done'}
                </button>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-1">
                  <h4 className="font-semibold text-sm text-slate-900">
                    {group.name}
                  </h4>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setEditingId(group.id)}
                      className="p-1 text-slate-400 hover:text-accent-500 hover:bg-accent-50 rounded transition-colors"
                      title={language === 'zh' ? '编辑' : 'Edit'}
                    >
                      <Pencil className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => handleDelete(group.id)}
                      className="p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
                {group.description && (
                  <p className="text-xs text-slate-500 mb-2 line-clamp-2">{group.description}</p>
                )}
                <div className="flex items-center gap-1 text-xs text-slate-500">
                  <span className="px-1.5 py-0.5 bg-accent-50 text-accent-600 rounded font-medium">
                    {group.sampleCount} {t('sample')}{group.sampleCount > 1 && language === 'en' ? 's' : ''}
                  </span>
                </div>
              </>
            )}
          </div>
        ))}

        {/* Add New Group Card */}
        {adding && (
          <div className="border-2 border-dashed border-accent-300 rounded-xl p-4 bg-accent-50/50">
            <div className="space-y-2">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder={t('groupNamePlaceholder')}
                className="w-full px-2 py-1.5 border border-slate-300 rounded text-sm"
                autoFocus
                onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
              />
              <input
                type="text"
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder={t('groupDescPlaceholder')}
                className="w-full px-2 py-1.5 border border-slate-300 rounded text-xs"
              />
              <div className="flex items-center gap-2">
                <label className="text-xs text-slate-600">{t('sampleCount')}:</label>
                <input
                  type="number"
                  value={newSampleCount}
                  onChange={(e) => setNewSampleCount(Math.max(1, parseInt(e.target.value) || 1))}
                  min={1}
                  max={100}
                  className="w-16 px-2 py-1 border border-slate-300 rounded text-xs"
                />
              </div>
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setAdding(false)}
                  className="px-3 py-1 text-xs text-slate-600 hover:bg-slate-200 rounded"
                >
                  {t('cancel')}
                </button>
                <button
                  onClick={handleAdd}
                  disabled={!newName.trim()}
                  className="px-3 py-1 text-xs bg-accent-600 text-white rounded hover:bg-accent-700 disabled:opacity-50"
                >
                  {t('add')}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {groups.length === 0 && !adding && (
        <p className="text-center text-sm text-slate-400 py-4">{t('noGroups')}</p>
      )}
    </section>
  )
}

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
function DataSection({
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
    <section className="border border-slate-200 rounded-xl overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 bg-slate-50 cursor-pointer select-none"
        onClick={() => setCollapsed(!collapsed)}
      >
        <div className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${iconColor}`} />
          <div>
            <h3 className="font-display text-base font-semibold text-slate-900">{title}</h3>
            <p className="text-xs text-slate-500">{description}</p>
          </div>
          {sectionFiles.length > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-slate-200 text-slate-600 rounded-full text-xs font-medium">
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
              <label className="block text-xs text-slate-500 mb-1">{t('selectGroup')}</label>
              <select
                value={selectedGroup}
                onChange={(e) => {
                  setSelectedGroup(e.target.value)
                  if (e.target.value === '_none') setSelectedSample('_none')
                }}
                className="w-full px-2 py-1.5 border border-slate-300 rounded-lg text-sm bg-white"
              >
                <option value="_none">{t('noGroupDistinction')}</option>
                {groups.map(g => (
                  <option key={g.id} value={g.id}>{g.name}</option>
                ))}
              </select>
            </div>

            {/* Sample dropdown */}
            <div className="min-w-[120px]">
              <label className="block text-xs text-slate-500 mb-1">{t('selectSample')}</label>
              <select
                value={selectedSample}
                onChange={(e) => setSelectedSample(e.target.value)}
                disabled={isNoGroup}
                className="w-full px-2 py-1.5 border border-slate-300 rounded-lg text-sm bg-white disabled:opacity-50 disabled:bg-slate-100"
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
              <label className="block text-xs text-slate-500 mb-1">{t('selectDataType')}</label>
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="w-full px-2 py-1.5 border border-slate-300 rounded-lg text-sm bg-white"
              >
                {typeOptions.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>

            {/* Custom type input for "Other" */}
            {selectedType === 'Other' && (
              <div className="min-w-[140px]">
                <label className="block text-xs text-slate-500 mb-1">{t('customType')}</label>
                <input
                  type="text"
                  value={customType}
                  onChange={(e) => setCustomType(e.target.value)}
                  placeholder={
                    section === 'sequencing'
                      ? (language === 'zh' ? '如 spatial transcriptomics' : 'e.g. spatial transcriptomics')
                      : (language === 'zh' ? '如 Sirius Red' : 'e.g. Sirius Red')
                  }
                  className="w-full px-2 py-1.5 border border-slate-300 rounded-lg text-sm"
                />
              </div>
            )}
          </div>

          {/* Upload Area */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
              isDragActive
                ? 'border-accent-500 bg-accent-50'
                : 'border-slate-300 hover:border-accent-400 hover:bg-slate-50'
            }`}
          >
            <input {...getInputProps()} accept={acceptAttr} />
            <Upload className="w-7 h-7 text-slate-400 mx-auto mb-1.5" />
            <p className="text-slate-600 font-medium text-sm">
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
                <div key={item.id} className="flex items-center gap-3 px-3 py-2 bg-slate-50 rounded-lg">
                  <File className="w-4 h-4 text-slate-400 flex-shrink-0" />
                  <span className="text-sm text-slate-700 truncate flex-1 min-w-0">{item.file.name}</span>
                  {item.status === 'uploading' && (
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <div className="w-24 bg-slate-200 rounded-full h-1.5">
                        <div
                          className="bg-accent-500 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${item.progress}%` }}
                        />
                      </div>
                      <span className="text-xs text-accent-600 font-medium w-8">{Math.round(item.progress)}%</span>
                    </div>
                  )}
                  {item.status === 'pending' && (
                    <span className="text-xs px-2 py-0.5 bg-slate-200 text-slate-500 rounded-full flex-shrink-0">
                      {language === 'zh' ? '等待中' : 'Pending'}
                    </span>
                  )}
                  {item.status === 'error' && (
                    <span className="text-xs px-2 py-0.5 bg-red-100 text-red-600 rounded-full flex-shrink-0" title={item.errorMsg}>
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
            <div className="border border-slate-200 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium text-slate-600">{t('filename')}</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600">{t('group')}</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600">{t('sample')}</th>
                    <th className="px-3 py-2 text-left font-medium text-slate-600">{t('dataType')}</th>
                    <th className="px-3 py-2 text-right font-medium text-slate-600 w-20">{t('actions')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {sectionFiles.map(file => (
                    <tr key={file.id} className="hover:bg-slate-50">
                      <td className="px-3 py-2 text-slate-900">
                        <div className="flex items-center gap-1.5">
                          <File className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
                          <span className="truncate max-w-[200px]">{file.filename}</span>
                        </div>
                      </td>
                      <td className="px-3 py-2 text-slate-600 text-xs">
                        {getGroupName(file.groupId)}
                      </td>
                      <td className="px-3 py-2 text-slate-600 text-xs">
                        {file.sampleNumber ? `${t('sample')} ${file.sampleNumber}` : '-'}
                      </td>
                      <td className="px-3 py-2">
                        <span className="px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded text-xs font-medium">
                          {file.dataType}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <div className="flex items-center justify-end gap-1">
                          {file.filePath && (
                            <a
                              href={`/api/download?path=${encodeURIComponent(file.filePath)}`}
                              download={file.filename}
                              className="p-1 text-slate-400 hover:text-blue-500 hover:bg-blue-50 rounded transition-colors"
                              title={language === 'zh' ? '下载' : 'Download'}
                            >
                              <Download className="w-3.5 h-3.5" />
                            </a>
                          )}
                          <button
                            onClick={() => onDelete(file.id)}
                            className="p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
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

// ============================================================
// IndicatorForm Component (for biochemistry manual input)
// ============================================================
function IndicatorForm({
  groups,
  indicators,
  onAdd,
  onDelete,
  language,
  t,
}: {
  groups: TreatmentGroup[]
  indicators: BiochemistryIndicator[]
  onAdd: (indicator: string, value: number, unit: string, groupId?: string, sampleNumber?: number) => Promise<any>
  onDelete: (id: string) => Promise<void>
  language: string
  t: (key: any, params?: any) => string
}) {
  const [showForm, setShowForm] = useState(false)
  const [indicator, setIndicator] = useState('')
  const [value, setValue] = useState('')
  const [unit, setUnit] = useState('')
  const [groupId, setGroupId] = useState<string>('_none')
  const [sampleNum, setSampleNum] = useState<string>('_none')

  const selectedGroup = groups.find(g => g.id === groupId)
  const isNoGroup = groupId === '_none'

  const handleAdd = async () => {
    const numValue = parseFloat(value)
    if (!indicator || isNaN(numValue) || !unit) return

    try {
      await onAdd(
        indicator,
        numValue,
        unit,
        isNoGroup ? undefined : groupId,
        isNoGroup || sampleNum === '_none' ? undefined : parseInt(sampleNum),
      )
      setIndicator('')
      setValue('')
      setUnit('')
    } catch (e) {
      console.error('Failed to add indicator:', e)
    }
  }

  const getGroupName = (gId?: string) => {
    if (!gId) return '-'
    return groups.find(g => g.id === gId)?.name || gId
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-slate-700">
          {language === 'zh' ? '手动输入指标' : 'Manual Indicators'}
        </h4>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-1 px-2 py-1 bg-amber-100 text-amber-700 rounded text-xs font-medium hover:bg-amber-200 transition-colors"
        >
          <Plus className="w-3 h-3" />
          {t('addIndicator')}
        </button>
      </div>

      {showForm && (
        <div className="p-3 bg-slate-50 rounded-lg space-y-2">
          <div className="flex flex-wrap gap-2">
            <select
              value={groupId}
              onChange={(e) => {
                setGroupId(e.target.value)
                if (e.target.value === '_none') setSampleNum('_none')
              }}
              className="px-2 py-1.5 border border-slate-300 rounded text-xs bg-white"
            >
              <option value="_none">{t('noGroupDistinction')}</option>
              {groups.map(g => (
                <option key={g.id} value={g.id}>{g.name}</option>
              ))}
            </select>
            <select
              value={sampleNum}
              onChange={(e) => setSampleNum(e.target.value)}
              disabled={isNoGroup}
              className="px-2 py-1.5 border border-slate-300 rounded text-xs bg-white disabled:opacity-50"
            >
              <option value="_none">{t('noSampleDistinction')}</option>
              {!isNoGroup && selectedGroup && Array.from({ length: selectedGroup.sampleCount }, (_, i) => (
                <option key={i + 1} value={String(i + 1)}>{t('sample')} {i + 1}</option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <input
              type="text"
              value={indicator}
              onChange={(e) => setIndicator(e.target.value)}
              placeholder={t('indicatorName')}
              className="px-2 py-1.5 border border-slate-300 rounded text-sm"
            />
            <input
              type="number"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder={t('value')}
              className="px-2 py-1.5 border border-slate-300 rounded text-sm"
            />
            <input
              type="text"
              value={unit}
              onChange={(e) => setUnit(e.target.value)}
              placeholder={t('unit')}
              className="px-2 py-1.5 border border-slate-300 rounded text-sm"
            />
          </div>
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setShowForm(false)}
              className="px-3 py-1 text-xs text-slate-600 hover:bg-slate-200 rounded"
            >
              {t('cancel')}
            </button>
            <button
              onClick={handleAdd}
              disabled={!indicator || !value || !unit}
              className="px-3 py-1 text-xs bg-amber-600 text-white rounded hover:bg-amber-700 disabled:opacity-50"
            >
              {t('add')}
            </button>
          </div>
        </div>
      )}

      {/* Indicators table */}
      {indicators.length > 0 && (
        <div className="border border-slate-200 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-slate-600">{t('indicator')}</th>
                <th className="px-3 py-2 text-left font-medium text-slate-600">{t('value')}</th>
                <th className="px-3 py-2 text-left font-medium text-slate-600">{t('unit')}</th>
                <th className="px-3 py-2 text-left font-medium text-slate-600">{t('group')}</th>
                <th className="px-3 py-2 text-left font-medium text-slate-600">{t('sample')}</th>
                <th className="px-3 py-2 text-right font-medium text-slate-600 w-12"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {indicators.map(ind => (
                <tr key={ind.id} className="hover:bg-slate-50">
                  <td className="px-3 py-2 font-medium text-slate-900">{ind.indicator}</td>
                  <td className="px-3 py-2 text-slate-600">{ind.value}</td>
                  <td className="px-3 py-2 text-slate-500">{ind.unit}</td>
                  <td className="px-3 py-2 text-xs text-slate-500">{getGroupName(ind.groupId)}</td>
                  <td className="px-3 py-2 text-xs text-slate-500">
                    {ind.sampleNumber ? `${t('sample')} ${ind.sampleNumber}` : '-'}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <button
                      onClick={() => onDelete(ind.id)}
                      className="p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ============================================================
// DataOverviewMatrix Component
// ============================================================
function DataOverviewMatrix({
  groups,
  dataFiles,
  indicators,
  language,
  t,
}: {
  groups: TreatmentGroup[]
  dataFiles: DataFileType[]
  indicators: BiochemistryIndicator[]
  language: string
  t: (key: any, params?: any) => string
}) {
  if (groups.length === 0) return null

  const sections = [
    { key: 'pathology', label: t('pathologySlides'), icon: Microscope, color: 'violet' },
    { key: 'sequencing', label: t('sequencingData'), icon: Dna, color: 'blue' },
    { key: 'biochemistry', label: t('biochemistryIndicators'), icon: FlaskConical, color: 'amber' },
  ]

  // Also count files not assigned to any group
  const ungroupedFiles = dataFiles.filter(f => !f.groupId)
  const ungroupedIndicators = indicators.filter(i => !i.groupId)

  return (
    <section className="mt-8">
      <div className="flex items-center gap-2 mb-3">
        <Grid3X3 className="w-5 h-5 text-slate-500" />
        <div>
          <h3 className="font-display text-lg font-semibold text-slate-900">{t('dataOverview')}</h3>
          <p className="text-sm text-slate-500">{t('dataOverviewDesc')}</p>
        </div>
      </div>

      <div className="border border-slate-200 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-4 py-2.5 text-left font-medium text-slate-600">{t('group')}</th>
              {sections.map(s => (
                <th key={s.key} className="px-4 py-2.5 text-center font-medium text-slate-600">
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {groups.map(group => {
              const groupFiles = dataFiles.filter(f => f.groupId === group.id)
              const groupIndicators = indicators.filter(i => i.groupId === group.id)
              return (
                <tr key={group.id} className="hover:bg-slate-50">
                  <td className="px-4 py-2.5 font-medium text-slate-900">
                    {group.name}
                    <span className="text-xs text-slate-400 ml-1">
                      ({group.sampleCount} {t('sample')}{group.sampleCount > 1 && language === 'en' ? 's' : ''})
                    </span>
                  </td>
                  {sections.map(s => {
                    const fileCount = groupFiles.filter(f => f.section === s.key).length
                    const indCount = s.key === 'biochemistry' ? groupIndicators.length : 0
                    const total = fileCount + indCount
                    return (
                      <td key={s.key} className="px-4 py-2.5 text-center">
                        {total > 0 ? (
                          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-${s.color}-100 text-${s.color}-700`}>
                            <Check className="w-3 h-3" />
                            {fileCount > 0 && t('filesCount', { n: fileCount })}
                            {fileCount > 0 && indCount > 0 && ' + '}
                            {indCount > 0 && t('indicatorsCount', { n: indCount })}
                          </span>
                        ) : (
                          <span className="text-slate-300">-</span>
                        )}
                      </td>
                    )
                  })}
                </tr>
              )
            })}

            {/* Ungrouped row */}
            {(ungroupedFiles.length > 0 || ungroupedIndicators.length > 0) && (
              <tr className="hover:bg-slate-50 bg-slate-50/50">
                <td className="px-4 py-2.5 text-slate-500 italic text-xs">
                  {t('noGroupDistinction')}
                </td>
                {sections.map(s => {
                  const fileCount = ungroupedFiles.filter(f => f.section === s.key).length
                  const indCount = s.key === 'biochemistry' ? ungroupedIndicators.length : 0
                  const total = fileCount + indCount
                  return (
                    <td key={s.key} className="px-4 py-2.5 text-center">
                      {total > 0 ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-slate-200 text-slate-600">
                          {fileCount > 0 && t('filesCount', { n: fileCount })}
                          {fileCount > 0 && indCount > 0 && ' + '}
                          {indCount > 0 && t('indicatorsCount', { n: indCount })}
                        </span>
                      ) : (
                        <span className="text-slate-300">-</span>
                      )}
                    </td>
                  )
                })}
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}

// ============================================================
// Main DataPanel Component
// ============================================================
export function DataPanel() {
  const { currentSession, language } = useAppStore()
  const {
    addGroup,
    updateGroup,
    deleteGroup,
    uploadData,
    deleteDataFile,
    addIndicator,
    deleteIndicator,
  } = useSession()
  const { t } = useTranslation(language)

  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 rounded-full mb-6">
          <Upload className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="font-display text-xl font-semibold text-slate-900 mb-2">
          {t('uploadData')}
        </h2>
        <p className="text-slate-500 max-w-md">
          {t('uploadDataDesc')}
        </p>
      </div>
    )
  }

  const groups = currentSession.treatmentGroups || []
  const dataFiles = currentSession.dataFiles || []
  const indicators = currentSession.biochemistryIndicators || []

  return (
    <div className="space-y-6">
      {/* Treatment Groups */}
      <GroupManager
        groups={groups}
        onAdd={addGroup}
        onUpdate={updateGroup}
        onDelete={deleteGroup}
        language={language}
        t={t}
      />

      {/* Pathology Section */}
      <DataSection
        section="pathology"
        title={t('pathologySlides')}
        description={t('pathologySlidesDesc')}
        icon={Microscope}
        iconColor="text-violet-500"
        groups={groups}
        dataFiles={dataFiles}
        typeOptions={PATHOLOGY_TYPES}
        multipleFiles={true}
        onUpload={uploadData}
        onDelete={deleteDataFile}
        language={language}
        t={t}
      />

      {/* Sequencing Section */}
      <DataSection
        section="sequencing"
        title={t('sequencingData')}
        description={t('sequencingDataDesc')}
        icon={Dna}
        iconColor="text-blue-500"
        groups={groups}
        dataFiles={dataFiles}
        typeOptions={SEQUENCING_TYPES}
        multipleFiles={true}
        onUpload={uploadData}
        onDelete={deleteDataFile}
        language={language}
        t={t}
      />

      {/* Biochemistry Section */}
      <DataSection
        section="biochemistry"
        title={t('biochemistryIndicators')}
        description={t('biochemistryIndicatorsDesc')}
        icon={FlaskConical}
        iconColor="text-amber-500"
        groups={groups}
        dataFiles={dataFiles}
        typeOptions={BIOCHEMISTRY_TYPES}
        multipleFiles={true}
        onUpload={uploadData}
        onDelete={deleteDataFile}
        language={language}
        t={t}
      >
        <IndicatorForm
          groups={groups}
          indicators={indicators}
          onAdd={addIndicator}
          onDelete={deleteIndicator}
          language={language}
          t={t}
        />
      </DataSection>

      {/* Data Overview Matrix */}
      <DataOverviewMatrix
        groups={groups}
        dataFiles={dataFiles}
        indicators={indicators}
        language={language}
        t={t}
      />
    </div>
  )
}
