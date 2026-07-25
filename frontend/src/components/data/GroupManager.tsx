// Treatment-group create/edit/delete panel.
// Split out of frontend/src/components/DataPanel.tsx verbatim.

import { useState } from 'react'
import { TreatmentGroup } from '../../stores/appStore'
import { Plus, Trash2, Users, Pencil } from 'lucide-react'

// ============================================================
// GroupManager Component
// ============================================================
export function GroupManager({
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
