// Group-by-modality coverage matrix.
// Split out of frontend/src/components/DataPanel.tsx; the per-cell counts were
// since rewritten from repeated Array.filter passes to one pre-pass into Maps.

import { TreatmentGroup, DataFile as DataFileType, BiochemistryIndicator } from '../../stores/appStore'
import { FlaskConical, Check, Grid3X3, Microscope, Dna } from 'lucide-react'

// DataOverviewMatrix Component
// ============================================================
export function DataOverviewMatrix({
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

  const fileCounts = new Map<string | undefined, Map<string, number>>()
  for (const file of dataFiles) {
    const groupId = file.groupId || undefined
    const sectionCounts = fileCounts.get(groupId) ?? new Map<string, number>()
    sectionCounts.set(file.section, (sectionCounts.get(file.section) ?? 0) + 1)
    fileCounts.set(groupId, sectionCounts)
  }

  const indicatorCounts = new Map<string | undefined, number>()
  for (const indicator of indicators) {
    const groupId = indicator.groupId || undefined
    indicatorCounts.set(groupId, (indicatorCounts.get(groupId) ?? 0) + 1)
  }

  const ungroupedFileCounts = fileCounts.get(undefined)
  const ungroupedIndicatorCount = indicatorCounts.get(undefined) ?? 0

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
              const groupFileCounts = fileCounts.get(group.id)
              const groupIndicatorCount = indicatorCounts.get(group.id) ?? 0
              return (
                <tr key={group.id} className="hover:bg-slate-50">
                  <td className="px-4 py-2.5 font-medium text-slate-900">
                    {group.name}
                    <span className="text-xs text-slate-400 ml-1">
                      ({group.sampleCount} {t('sample')}{group.sampleCount > 1 && language === 'en' ? 's' : ''})
                    </span>
                  </td>
                  {sections.map(s => {
                    const fileCount = groupFileCounts?.get(s.key) ?? 0
                    const indCount = s.key === 'biochemistry' ? groupIndicatorCount : 0
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
            {(ungroupedFileCounts || ungroupedIndicatorCount > 0) && (
              <tr className="hover:bg-slate-50 bg-slate-50/50">
                <td className="px-4 py-2.5 text-slate-500 italic text-xs">
                  {t('noGroupDistinction')}
                </td>
                {sections.map(s => {
                  const fileCount = ungroupedFileCounts?.get(s.key) ?? 0
                  const indCount = s.key === 'biochemistry' ? ungroupedIndicatorCount : 0
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
