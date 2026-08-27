import { FileText, X } from 'lucide-react'

import { AGENT_OPTIONS } from '../../constants/agents'
import { getSection, getTypeOptions, hasCustomOption, selectValueForDataType } from '../../utils/dataTypes'
import { CustomTypeInput } from '../shared/CustomTypeInput'
import type { ChatHistoryController } from './useChatHistory'
import type { ReturnTypeOfChatSend } from './useChatSend'

interface ChatAttachmentsProps {
  history: ChatHistoryController
  actions: ReturnTypeOfChatSend
}

export function ChatAttachments({ history, actions }: ChatAttachmentsProps) {
  const { currentSession, filePreviews, language, selectedFiles, t } = history
  const {
    handleFileAgentChange, handleFileGroupChange, handleFileSampleChange,
    handleFileTypeChange, handleRemoveFile,
  } = actions
  if (selectedFiles.length === 0) return null
  return (
    <>
      {/* Selected files preview with per-file agent/group/sample/type selectors */}
      {selectedFiles.length > 0 && (
        <div className="mx-4 mb-2 space-y-1.5">
          {selectedFiles.map((sf, i) => {
            const preview = filePreviews[sf.file.name + sf.file.size]
            const section = getSection(sf.targetAgent)
            const typeOptions = getTypeOptions(sf.targetAgent)
            const groups = currentSession?.treatmentGroups || []
            const selectedGroupObj = groups.find(g => g.id === sf.groupId)
            const maxSamples = selectedGroupObj?.sampleCount || 0

            return (
              <div
                key={i}
                className="bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 dark:text-slate-200 space-y-1"
              >
                {/* Row 1: preview + filename + agent + delete */}
                <div className="flex items-center gap-2">
                  {preview ? (
                    <img src={preview} alt="" className="w-6 h-6 rounded object-cover shrink-0" />
                  ) : (
                    <FileText className="w-4 h-4 text-slate-400 shrink-0" />
                  )}
                  <span className="truncate max-w-[140px]">{sf.file.name}</span>
                  <select
                    value={sf.targetAgent}
                    onChange={(e) => handleFileAgentChange(i, e.target.value)}
                    className="ml-auto shrink-0 px-1.5 py-0.5 border border-slate-300 dark:border-slate-600 rounded text-[11px] text-slate-600 dark:text-slate-300 bg-white dark:bg-surface-900 focus:ring-1 focus:ring-accent-500"
                  >
                    {AGENT_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {t(opt.labelKey)}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={() => handleRemoveFile(i)}
                    className="p-0.5 hover:bg-slate-200 dark:hover:bg-slate-700 rounded shrink-0"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>

                {/* Row 2: group + sample + type — only for non-PI agents */}
                {section && (
                  <div className="flex items-center gap-1.5 pl-1">
                    {/* Treatment group */}
                    <select
                      value={sf.groupId || '_none'}
                      onChange={(e) => handleFileGroupChange(i, e.target.value === '_none' ? undefined : e.target.value)}
                      className="px-1 py-0.5 border border-slate-200 dark:border-slate-700 rounded text-[11px] bg-white dark:bg-surface-900 max-w-[100px]"
                    >
                      <option value="_none">{language === 'zh' ? '无分组' : 'No Group'}</option>
                      {groups.map(g => (
                        <option key={g.id} value={g.id}>{g.name}</option>
                      ))}
                    </select>

                    {/* Sample number */}
                    <select
                      value={sf.sampleNumber != null ? String(sf.sampleNumber) : '_none'}
                      disabled={!sf.groupId}
                      onChange={(e) => handleFileSampleChange(i, e.target.value === '_none' ? undefined : parseInt(e.target.value))}
                      className="px-1 py-0.5 border border-slate-200 dark:border-slate-700 rounded text-[11px] bg-white dark:bg-surface-900 disabled:opacity-40 max-w-[60px]"
                    >
                      <option value="_none">-</option>
                      {Array.from({ length: maxSamples }, (_, j) => (
                        <option key={j + 1} value={String(j + 1)}>S{j + 1}</option>
                      ))}
                    </select>

                    {/* Data type */}
                    <select
                      value={hasCustomOption(typeOptions) ? selectValueForDataType(sf.dataType, typeOptions) : (sf.dataType || typeOptions[0] || '')}
                      onChange={(e) => handleFileTypeChange(i, e.target.value)}
                      className="px-1 py-0.5 border border-slate-200 dark:border-slate-700 rounded text-[11px] bg-white dark:bg-surface-900 max-w-[90px]"
                    >
                      {typeOptions.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                    {hasCustomOption(typeOptions) && (
                      <CustomTypeInput
                        dataType={sf.dataType || ''}
                        typeOptions={typeOptions}
                        onCustomChange={(val) => handleFileTypeChange(i, val)}
                        language={language}
                      />
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </>
  )
}
