import {
  AlertCircle, Bot, Download, FileText, Image as ImageIcon, Loader2, User,
} from 'lucide-react'

import { base64ToDataUri } from '../utils/image'
import { ChatAttachments } from './chat/ChatAttachments'
import { ChatComposer } from './chat/ChatComposer'
import { ChatMarkdown } from './chat/ChatMarkdown'
import { isImageFile, useChatHistory } from './chat/useChatHistory'
import { useChatSend } from './chat/useChatSend'

export function ChatPanel() {
  const history = useChatHistory()
  const actions = useChatSend(history)
  const {
    chatDelegationActive, chatDelegationPlan, chatLoading, chatMessages, currentSession,
    error, language, messagesEndRef, t,
  } = history

  // No session state
  if (!currentSession) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center text-slate-400">
          <Bot className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p className="text-sm">{t('chatNoSession')}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-auto p-4 space-y-4 bg-surface-50 dark:bg-surface-950">
        {chatMessages.length === 0 ? (
          <div className="flex-1 flex items-center justify-center h-full min-h-[300px]">
            <div className="text-center text-slate-400">
              <Bot className="w-16 h-16 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium mb-1">{t('chatEmpty')}</p>
              <p className="text-sm">{t('chatEmptyDesc')}</p>
            </div>
          </div>
        ) : (
          chatMessages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="shrink-0 w-8 h-8 rounded-full bg-accent-100 dark:bg-accent-900/60 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-accent-600 dark:text-accent-400" />
                </div>
              )}
              <div
                className={`min-w-0 max-w-[80%] rounded-2xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-accent-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-100'
                }`}
              >
                {/* File attachments */}
                {msg.files && msg.files.length > 0 && (
                  <div className="mb-2 space-y-1">
                    {msg.files.map((f: any, j: number) => (
                      <div
                        key={j}
                        className={`flex items-center gap-1.5 text-xs rounded-lg px-2 py-1 ${
                          msg.role === 'user' ? 'bg-accent-500/30' : 'bg-slate-200 dark:bg-slate-700'
                        }`}
                      >
                        {isImageFile(f.filename) ? (
                          <ImageIcon className="w-3 h-3 shrink-0" />
                        ) : (
                          <FileText className="w-3 h-3 shrink-0" />
                        )}
                        <span className="truncate">{f.filename}</span>
                        {f.target_agent && f.target_agent !== 'pi' && (
                          <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            msg.role === 'user' ? 'bg-accent-400/40' : 'bg-accent-100 dark:bg-accent-900/60 text-accent-700 dark:text-accent-300'
                          }`}>
                            {f.target_agent}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {/* Message content */}
                {msg.content && (
                  msg.role === 'assistant' ? (
                    <ChatMarkdown content={msg.content} />
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  )
                )}
                {msg.pending_figures && (
                  <div className="mt-2 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-accent-500" />
                    <span>Generating figures...</span>
                  </div>
                )}
                {/* Referenced / generated figures */}
                {msg.figures && msg.figures.length > 0 && (
                  <div className="mt-3 space-y-3">
                    {msg.figures.map((fig, idx) => (
                      <div key={idx} className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden bg-white dark:bg-surface-900">
                        <img
                          src={fig.base64
                            ? base64ToDataUri(fig.base64)
                            : `/api/file?path=${encodeURIComponent(fig.path || '')}`}
                          alt={fig.name}
                          className="publication-canvas w-full h-auto"
                          loading="lazy"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                        />
                        <div className="px-3 py-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
                          <div className="flex min-w-0 items-center gap-2">
                            <p className="truncate text-xs font-medium text-slate-700 dark:text-slate-200">{fig.name}</p>
                            <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                              {fig.figure_role === 'publication' ? 'Publication figure' : 'Quick view'}
                            </span>
                          </div>
                          <div className="flex gap-1.5">
                            {fig.pdfPath && (
                              <a
                                href={`/api/file?path=${encodeURIComponent(fig.pdfPath)}`}
                                download={`${fig.name}.pdf`}
                                className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-accent-700 dark:text-accent-300 bg-accent-50 dark:bg-accent-900/40 hover:bg-accent-100 dark:hover:bg-accent-900/60 rounded-md transition-colors"
                              >
                                <Download className="w-3 h-3" />
                                PDF
                              </a>
                            )}
                            {fig.path && (
                              <a
                                href={`/api/file?path=${encodeURIComponent(fig.path)}`}
                                download={`${fig.name}.png`}
                                className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-md transition-colors"
                              >
                                <Download className="w-3 h-3" />
                                PNG
                              </a>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {msg.role === 'user' && (
                <div className="shrink-0 w-8 h-8 rounded-full bg-accent-600 flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
              )}
            </div>
          ))
        )}
        {(chatLoading || chatDelegationActive)
          && !(
            chatLoading
            && !chatDelegationActive
            && chatMessages[chatMessages.length - 1]?.role === 'assistant'
            && chatMessages[chatMessages.length - 1]?.pending_figures
          ) && (
          <div className="flex gap-3 justify-start">
            <div className="shrink-0 w-8 h-8 rounded-full bg-accent-100 dark:bg-accent-900/60 flex items-center justify-center">
              <Bot className="w-4 h-4 text-accent-600 dark:text-accent-400" />
            </div>
            <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl px-4 py-3">
              {chatDelegationActive ? (
                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                  <Loader2 className="w-4 h-4 animate-spin text-accent-500 shrink-0" />
                  <div>
                    <div className="font-medium text-accent-700 dark:text-accent-300">{chatDelegationPlan || 'Coordinating agents...'}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{language === 'zh' ? '分析进度请查看 Analysis 标签页' : 'Progress in Analysis tab'}</div>
                  </div>
                </div>
              ) : (
                <Loader2 className="w-4 h-4 animate-spin text-accent-500" />
              )}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 flex items-center gap-2 p-2 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400 text-xs">
          <AlertCircle className="w-3 h-3 shrink-0" />
          {error}
        </div>
      )}

      <ChatAttachments history={history} actions={actions} />

      <ChatComposer history={history} actions={actions} />
    </div>
  )
}
