import { Paperclip, Send, Square } from 'lucide-react'

import { AGENT_OPTIONS } from '../../constants/agents'
import type { ChatHistoryController } from './useChatHistory'
import type { ReturnTypeOfChatSend } from './useChatSend'

interface ChatComposerProps {
  history: ChatHistoryController
  actions: ReturnTypeOfChatSend
}

export function ChatComposer({ history, actions }: ChatComposerProps) {
  const {
    authUser, chatLoading, defaultAgent, fileInputRef, input, language, selectedFiles,
    setDefaultAgent, setInput, t, textareaHeight, textareaRef,
  } = history
  const { handleAddFiles, handleKeyDown, handleResizeMouseDown, handleSend, handleStop } = actions
  return (
    <>
      {/* Input area */}
      <div className="border-t border-slate-200 dark:border-slate-700 p-4 bg-white dark:bg-surface-900">
        <div className="flex items-end gap-2">
          {/* Attach button */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.csv,.xlsx,.xls,.txt,.json,.tsv"
            onChange={handleAddFiles}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={chatLoading || authUser?.status === 'readonly'}
            className="shrink-0 p-3 text-slate-500 dark:text-slate-400 hover:text-accent-600 dark:hover:text-accent-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-colors disabled:opacity-50"
            title={t('chatAttach')}
          >
            <Paperclip className="w-5 h-5" />
          </button>

          {/* Default agent selector for new files */}
          <select
            value={defaultAgent}
            onChange={(e) => setDefaultAgent(e.target.value)}
            disabled={chatLoading || authUser?.status === 'readonly'}
            className="shrink-0 px-2 py-3 border border-slate-300 dark:border-slate-600 rounded-xl text-xs text-slate-600 dark:text-slate-300 bg-white dark:bg-surface-900 focus:ring-2 focus:ring-accent-500 focus:border-accent-500 disabled:opacity-50"
            title={language === 'zh' ? '新文件默认Agent' : 'Default agent for new files'}
          >
            {AGENT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {t(opt.labelKey)}
              </option>
            ))}
          </select>

          {/* Textarea with top-edge resize handle */}
          <div className="flex-1 flex flex-col">
            {/* Drag handle at top */}
            <div
              onMouseDown={handleResizeMouseDown}
              className="flex justify-center py-1 cursor-ns-resize group"
            >
              <div className="w-8 h-1 rounded-full bg-slate-300 dark:bg-slate-600 group-hover:bg-accent-400 transition-colors" />
            </div>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={authUser?.status === 'readonly' ? 'Read-only mode' : t('chatPlaceholder')}
              className="w-full resize-none px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-xl bg-white dark:bg-surface-900 text-surface-800 dark:text-surface-100 focus:ring-2 focus:ring-accent-500 focus:border-accent-500 text-sm"
              style={{ height: textareaHeight + 'px' }}
              disabled={chatLoading || authUser?.status === 'readonly'}
            />
          </div>

          {/* Send / Stop button */}
          {chatLoading ? (
            <button
              onClick={handleStop}
              className="shrink-0 p-3 bg-red-500 hover:bg-red-600 text-white rounded-xl transition-colors"
              title={t('chatStop')}
            >
              <Square className="w-5 h-5 fill-current" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={(!input.trim() && selectedFiles.length === 0) || authUser?.status === 'readonly'}
              className="shrink-0 p-3 bg-accent-600 hover:bg-accent-700 text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    </>
  )
}
