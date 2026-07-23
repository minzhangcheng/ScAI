// Manual biochemistry indicator entry form.
// Split out of frontend/src/components/DataPanel.tsx verbatim.

import { useState } from 'react'
import { TreatmentGroup, BiochemistryIndicator } from '../../stores/appStore'
import { Plus, Trash2 } from 'lucide-react'

// ============================================================
// IndicatorForm Component (for biochemistry manual input)
// ============================================================
export function IndicatorForm({
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
