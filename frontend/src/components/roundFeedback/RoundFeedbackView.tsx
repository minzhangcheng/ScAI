import {
  AlertTriangle, BookOpen, CheckCircle2, ChevronDown, Clock, Code2, Download, FlaskConical,
  Loader2, Microscope, Pause, Play, Send, Square, Upload, Users, X,
} from 'lucide-react'

import { AGENT_OPTIONS } from '../../constants/agents'
import {
  getSection, getTypeOptions, hasCustomOption, selectValueForDataType,
} from '../../utils/dataTypes'
import { CustomTypeInput } from '../shared/CustomTypeInput'
import { SupplementaryAnalysisModal } from '../SupplementaryAnalysisModal'
import type { RoundFeedbackController } from './useRoundFeedbackController'

interface RoundFeedbackViewProps {
  controller: RoundFeedbackController
}

function RoundFeedbackComposer({ controller }: RoundFeedbackViewProps) {
  const {
    customTypeError, dropdownRef, fileInputRef, files, handleFileSelect, handleInputChange,
    handleSend, isLastRound, isSupplementaryRunning, language, openDropdownIndex,
    openSupplementaryInput, removeFile, sending, setCustomTypeError, setOpenDropdownIndex,
    t, treatmentGroups, updateFileAgent, updateFileGroup, updateFileSample, updateFileType,
    userInput,
  } = controller
  return (
    <div className="border-t border-slate-200 px-6 py-4 space-y-3 bg-slate-50">

          {/* Last round notice */}
          {isLastRound && (
            <div className="flex items-center gap-2 px-3 py-2 bg-amber-100 border border-amber-300 rounded-lg text-sm text-amber-800">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              {t('lastRoundNotice')}
            </div>
          )}

          {/* File list */}
          {files.length > 0 && (
            <div className="space-y-1.5">
              {files.map((f, index) => {
                const section = getSection(f.targetAgent)
                const typeOptions = getTypeOptions(f.targetAgent)
                const selectedGroupObj = treatmentGroups.find(g => g.id === f.groupId)
                const maxSamples = selectedGroupObj?.sampleCount || 0

                return (
                  <div key={index} className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs space-y-1">
                    {/* Row 1: filename + Agent + delete */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-slate-700 max-w-[140px] truncate" title={f.file.name}>
                        {f.file.name}
                      </span>
                      <span className="text-slate-400">→</span>
                      {/* Agent selector dropdown */}
                      <div className="relative">
                        <button
                          onClick={(e) => { e.stopPropagation(); setOpenDropdownIndex(openDropdownIndex === index ? null : index) }}
                          className="flex items-center gap-0.5 text-blue-600 hover:text-blue-800 font-medium"
                        >
                          {t(AGENT_OPTIONS.find(a => a.value === f.targetAgent)?.labelKey ?? 'chatTargetPi')}
                          <ChevronDown className="w-3 h-3" />
                        </button>
                        {openDropdownIndex === index && (
                          <div ref={dropdownRef} className="absolute top-full left-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-lg z-10 py-1 min-w-[120px]">
                            {AGENT_OPTIONS.map(agent => (
                              <button
                                key={agent.value}
                                onClick={(e) => { e.stopPropagation(); updateFileAgent(index, agent.value) }}
                                className={`w-full text-left px-3 py-1.5 text-xs hover:bg-slate-50 ${
                                  f.targetAgent === agent.value ? 'text-blue-600 font-medium' : 'text-slate-700'
                                }`}
                              >
                                {t(agent.labelKey)}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-slate-400 hover:text-red-500 ml-auto"
                        title={t('removeFile')}
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>

                    {/* Row 2: Group + Sample + Type — only for non-PI agents */}
                    {section && (
                      <div className="flex items-center gap-1.5 pl-1">
                        {/* Treatment group */}
                        <select
                          value={f.groupId || '_none'}
                          onChange={(e) => updateFileGroup(index, e.target.value === '_none' ? undefined : e.target.value)}
                          className="px-1 py-0.5 border border-slate-200 rounded text-[11px] bg-white max-w-[100px]"
                        >
                          <option value="_none">{language === 'zh' ? '无分组' : 'No Group'}</option>
                          {treatmentGroups.map(g => (
                            <option key={g.id} value={g.id}>{g.name}</option>
                          ))}
                        </select>

                        {/* Sample number */}
                        <select
                          value={f.sampleNumber != null ? String(f.sampleNumber) : '_none'}
                          disabled={!f.groupId}
                          onChange={(e) => updateFileSample(index, e.target.value === '_none' ? undefined : parseInt(e.target.value))}
                          className="px-1 py-0.5 border border-slate-200 rounded text-[11px] bg-white disabled:opacity-40 max-w-[60px]"
                        >
                          <option value="_none">-</option>
                          {Array.from({ length: maxSamples }, (_, i) => (
                            <option key={i + 1} value={String(i + 1)}>S{i + 1}</option>
                          ))}
                        </select>

                        {/* Data type */}
                        <select
                          value={hasCustomOption(typeOptions) ? selectValueForDataType(f.dataType, typeOptions) : (f.dataType || typeOptions[0] || '')}
                          onChange={(e) => { updateFileType(index, e.target.value); setCustomTypeError('') }}
                          className="px-1 py-0.5 border border-slate-200 rounded text-[11px] bg-white max-w-[90px]"
                        >
                          {typeOptions.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                        {hasCustomOption(typeOptions) && (
                          <CustomTypeInput
                            dataType={f.dataType || ''}
                            typeOptions={typeOptions}
                            onCustomChange={(val) => { updateFileType(index, val); setCustomTypeError('') }}
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

          {customTypeError && (
            <p className="text-xs text-red-500 mt-1">{customTypeError}</p>
          )}

          {/* Input row */}
          <div className="flex items-end gap-2">
            <textarea
              value={userInput}
              onChange={handleInputChange}
              placeholder={t('roundInputPlaceholder')}
              className="flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
              rows={3}
              disabled={sending}
            />
            <div className="flex flex-col gap-2">
              {/* Upload button */}
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={sending}
                className="p-2 rounded-lg border border-slate-300 hover:bg-slate-100 text-slate-600 transition-colors disabled:opacity-50"
                title={t('uploadFilesDialog')}
              >
                <Upload className="w-4 h-4" />
              </button>
              {/* Send button */}
              <button
                onClick={handleSend}
                disabled={sending || isSupplementaryRunning || (!userInput.trim() && files.length === 0)}
                className="p-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:bg-slate-300"
                title={t('sendInput')}
              >
                {sending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Supplementary analysis — an alternative to the guidance above: it runs
              now and comes back to this same gate instead of ending the round. */}
          <div className="flex items-center gap-3">
            <button
              onClick={openSupplementaryInput}
              disabled={sending || isSupplementaryRunning}
              className="px-4 py-2.5 rounded-xl text-sm font-medium border border-amber-300 text-amber-700 hover:bg-amber-50 transition-all disabled:opacity-50"
            >
              {isSupplementaryRunning
                ? t('roundSupplementaryRunning')
                : t('roundRequestSupplementary')}
            </button>
            <span className="text-xs text-slate-500">{t('roundSupplementaryHint')}</span>
          </div>

          {/* Sending status */}
          {sending && (
            <div className="flex items-center gap-2 text-sm text-blue-600">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              {t('sendingInput')}
            </div>
          )}

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.csv,.xlsx,.xls,.txt,.json,.tsv"
            onChange={handleFileSelect}
            className="hidden"
          />
    </div>
  )
}

export function RoundFeedbackView({ controller }: RoundFeedbackViewProps) {
  const {
    reportUrl, downloadingReport, reportDownloadError, handleDownloadReport,
    bioinformaticsFindings, biochemistryFindings, handleSkip, hasCountdown, hasDataRequests,
    isManualMode, isSupplementaryRunning, language, literatureFindings, onRequestSupplementary,
    onStopImmediate, onStopSummarize, pathologyFindings, roundNumber, roundSummary, sending,
    setShowSupplementaryInput, showSupplementaryInput, t, timerDisplay, timerPaused,
    timerUrgent, toggleTimerPause,
  } = controller
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-modal w-full max-w-4xl mx-4 max-h-[85vh] flex flex-col overflow-hidden animate-scale-in">

        {/* Header with timer */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-700 bg-gradient-to-r from-blue-50 dark:from-blue-950/40 to-accent-50 dark:to-accent-900/40">
          <h2 className="text-lg font-display font-semibold text-slate-800 dark:text-slate-100">
            {t('roundComplete').replace('{round}', String(roundNumber))}
          </h2>
          <div className="flex items-center gap-3">
            {/* Timer — only shown in countdown mode */}
            {hasCountdown && (
              <>
                <div className="flex items-center gap-2">
                  <Clock className={`w-4 h-4 ${timerUrgent ? 'text-red-500' : 'text-slate-500'}`} />
                  <span className={`font-mono text-lg tabular-nums ${
                    timerUrgent ? 'text-red-600 animate-pulse' :
                    timerPaused ? 'text-amber-600' : 'text-slate-700'
                  }`}>
                    {timerDisplay}
                  </span>
                  {timerPaused && (
                    <span className="text-xs text-amber-600 font-medium">{t('timerPaused')}</span>
                  )}
                </div>
                {/* Pause/Resume timer */}
                <button
                  onClick={toggleTimerPause}
                  className="p-1.5 rounded-lg hover:bg-white/60 text-slate-600 hover:text-slate-800 transition-colors"
                  title={timerPaused ? t('resumeTimer') : t('pauseTimer')}
                >
                  {timerPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                </button>
              </>
            )}
            {/* Manual mode indicator */}
            {isManualMode && (
              <span className="text-xs text-slate-500 font-medium">
                {language === 'zh' ? '手动模式' : 'Manual mode'}
              </span>
            )}
            {/* Continue button (renamed from Skip) */}
            <button
              onClick={handleSkip}
              disabled={sending || isSupplementaryRunning}
              className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-lg bg-blue-100 hover:bg-blue-200 text-blue-700 transition-colors disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5" />
              {t('skipRound')}
            </button>
            {/* Terminate immediately — left enabled during a supplementary run, since
                stopping must stay reachable while the gate waits */}
            {onStopImmediate && (
              <button
                onClick={onStopImmediate}
                disabled={sending}
                className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-lg bg-red-100 hover:bg-red-200 text-red-700 transition-colors disabled:opacity-50"
                title={language === 'zh' ? '立即终止分析' : 'Terminate immediately'}
              >
                <Square className="w-3.5 h-3.5" />
                {language === 'zh' ? '立即终止' : 'Terminate'}
              </button>
            )}
            {/* Terminate and summarize */}
            {onStopSummarize && (
              <button
                onClick={onStopSummarize}
                disabled={sending}
                className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-lg bg-amber-100 hover:bg-amber-200 text-amber-700 transition-colors disabled:opacity-50"
                title={language === 'zh' ? '终止分析并生成报告' : 'Terminate and summarize'}
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                {language === 'zh' ? '终止并总结' : 'Summarize'}
              </button>
            )}
          </div>
        </div>

        {/* Content area — top 2/3 scrollable */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4" style={{ minHeight: 0 }}>

          {reportUrl && (
            <div>
              <button
                type="button"
                onClick={handleDownloadReport}
                disabled={downloadingReport}
                className="flex items-center gap-2 text-sm text-blue-600 hover:underline disabled:opacity-50"
              >
                {downloadingReport
                  ? <Loader2 className="w-4 h-4 animate-spin" />
                  : <Download className="w-4 h-4" />}
                {t('downloadRoundReport')}
              </button>
              <p className="mt-1 text-xs text-slate-500">{t('roundReportDownloadHint')}</p>
              {reportDownloadError && (
                <p role="alert" className="mt-2 text-sm text-red-600">{reportDownloadError}</p>
              )}
            </div>
          )}

          {/* PI Summary */}
          {roundSummary.summary && (
            <div>
              <h3 className="text-sm font-semibold text-slate-600 mb-2">{t('roundSummaryTitle')}</h3>
              <div className="bg-slate-50 rounded-lg p-4 text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
                {roundSummary.summary}
              </div>
            </div>
          )}

          {/* Per-agent findings */}
          <div>
            <h3 className="text-sm font-semibold text-slate-600 mb-2">{t('keyFindings')}</h3>
            <div className="space-y-2">
              {pathologyFindings.length > 0 && (
                <FindingsSection
                  icon={<Microscope className="w-4 h-4 text-purple-600" />}
                  label={t('pathologyFindings')}
                  findings={pathologyFindings}
                />
              )}
              {bioinformaticsFindings.length > 0 && (
                <FindingsSection
                  icon={<Code2 className="w-4 h-4 text-blue-600" />}
                  label={t('bioinformaticsFindings')}
                  findings={bioinformaticsFindings}
                />
              )}
              {biochemistryFindings.length > 0 && (
                <FindingsSection
                  icon={<FlaskConical className="w-4 h-4 text-green-600" />}
                  label={t('biochemistryFindings')}
                  findings={biochemistryFindings}
                />
              )}
              {literatureFindings.length > 0 && (
                <FindingsSection
                  icon={<BookOpen className="w-4 h-4 text-orange-600" />}
                  label={t('literatureFindings')}
                  findings={literatureFindings}
                />
              )}
              {pathologyFindings.length === 0 && bioinformaticsFindings.length === 0 &&
               biochemistryFindings.length === 0 && literatureFindings.length === 0 &&
               roundSummary.findings?.length > 0 && (
                <FindingsSection
                  icon={<Users className="w-4 h-4 text-slate-600" />}
                  label={t('keyFindings')}
                  findings={roundSummary.findings}
                />
              )}
            </div>
          </div>

          {/* Data requests */}
          {hasDataRequests && (
            <div>
              <h3 className="text-sm font-semibold text-slate-600 mb-2">{t('additionalDataSuggested')}</h3>
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm space-y-1">
                {roundSummary.dataRequests.pathology?.map((req, i) => (
                  <div key={`p-${i}`} className="flex items-start gap-2">
                    <Microscope className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
                    <span className="text-amber-800">{req}</span>
                  </div>
                ))}
                {roundSummary.dataRequests.biochemistry?.map((req, i) => (
                  <div key={`b-${i}`} className="flex items-start gap-2">
                    <FlaskConical className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
                    <span className="text-amber-800">{req}</span>
                  </div>
                ))}
                {roundSummary.dataRequests.other?.map((req, i) => (
                  <div key={`o-${i}`} className="flex items-start gap-2">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
                    <span className="text-amber-800">{req}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>


        <RoundFeedbackComposer controller={controller} />
      </div>

      {showSupplementaryInput && (
        <SupplementaryAnalysisModal
          language={language}
          placeholder={t('roundSupplementaryPlaceholder')}
          onSubmit={onRequestSupplementary}
          onClose={() => setShowSupplementaryInput(false)}
        />
      )}
    </div>
  )

}

function FindingsSection({
  icon,
  label,
  findings,
}: {
  icon: React.ReactNode
  label: string
  findings: string[]
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1.5">
        {icon}
        <span className="text-sm font-medium text-slate-700">{label}</span>
      </div>
      <ul className="space-y-1 pl-6">
        {findings.map((finding, i) => (
          <li key={i} className="text-sm text-slate-600 list-disc">{finding}</li>
        ))}
      </ul>
    </div>
  )
}
