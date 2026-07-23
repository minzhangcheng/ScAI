// Group-by-modality coverage matrix.
// Split out of frontend/src/components/DataPanel.tsx verbatim.

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
