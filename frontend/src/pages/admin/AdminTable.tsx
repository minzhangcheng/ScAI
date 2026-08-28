import { Loader2 } from 'lucide-react'

import type { AdminController } from './useAdminController'

function EnumCell({ readOnly, value, options, onChange }: {
  readOnly: boolean
  value: string
  options: string[]
  onChange: (value: string) => void
}) {
  if (readOnly) {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 dark:bg-purple-900/60 text-purple-700 dark:text-purple-300">
        {value}
      </span>
    )
  }
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="text-sm border border-surface-200 dark:border-surface-700 rounded-lg px-2 py-1 bg-white dark:bg-surface-900 focus:ring-2 focus:ring-accent-500/20 focus:border-accent-500"
    >
      {options.map(option => <option key={option} value={option}>{option}</option>)}
    </select>
  )
}

export function AdminTable({ controller }: { controller: AdminController }) {
  const {
    authUser, formatDate, handleRoleChange, handleStatusChange, loading,
    setChangePasswordTarget, setDeleteTarget, users, zh,
  } = controller
  return (
    <div className="bg-white dark:bg-surface-900 rounded-xl border border-surface-200 dark:border-surface-700 overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="bg-surface-50 dark:bg-surface-950 border-b border-surface-200 dark:border-surface-700">
            <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider">{zh ? '用户名' : 'Username'}</th>
            <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider">Role</th>
            <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider">Status</th>
            <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider">{zh ? '创建时间' : 'Created'}</th>
            <th className="text-left px-6 py-3 text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider">{zh ? '操作' : 'Actions'}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-surface-100 dark:divide-surface-800">
          {loading ? (
            <tr><td colSpan={5} className="px-6 py-12 text-center text-surface-400">
              <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
              <p className="text-sm">{zh ? '加载中...' : 'Loading...'}</p>
            </td></tr>
          ) : users.length === 0 ? (
            <tr><td colSpan={5} className="px-6 py-12 text-center text-surface-400 text-sm">{zh ? '暂无用户' : 'No users found'}</td></tr>
          ) : users.map(user => {
            const isSelf = user.id === authUser?.id
            const isDeleting = user.status === 'deleting'
            return (
              <tr key={user.id} className="hover:bg-surface-50 dark:hover:bg-surface-950 transition-colors">
                <td className="px-6 py-4">
                  <p className="text-sm font-medium text-surface-900 dark:text-surface-100">{user.username}</p>
                  {user.email && <p className="text-xs text-surface-500 dark:text-surface-400">{user.email}</p>}
                </td>
                <td className="px-6 py-4">
                  <EnumCell
                    readOnly={isSelf || isDeleting}
                    value={user.role}
                    options={['user', 'admin']}
                    onChange={role => handleRoleChange(user.id, role)}
                  />
                </td>
                <td className="px-6 py-4">
                  <EnumCell
                    readOnly={isSelf || isDeleting}
                    value={user.status}
                    options={user.role === 'admin' ? ['readonly', 'active'] : ['pending', 'readonly', 'active']}
                    onChange={status => handleStatusChange(user.id, status)}
                  />
                </td>
                <td className="px-6 py-4 text-sm text-surface-600 dark:text-surface-300">{formatDate(user.created_at)}</td>
                <td className="px-6 py-4">
                  {isSelf ? (
                    <span className="text-xs text-surface-400 italic">{zh ? '本人' : 'You'}</span>
                  ) : isDeleting ? (
                    <div className="flex items-center gap-4">
                      <span className="text-xs text-amber-700 dark:text-amber-300">{zh ? '删除清理中' : 'Deletion in progress'}</span>
                      <button onClick={() => setDeleteTarget(user)} className="text-sm text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 font-medium transition-colors">
                        {zh ? '重试清理' : 'Retry cleanup'}
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-4">
                      <button onClick={() => setChangePasswordTarget(user)} className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium transition-colors">{zh ? '改密' : 'Password'}</button>
                      <button onClick={() => setDeleteTarget(user)} className="text-sm text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 font-medium transition-colors">{zh ? '删除' : 'Delete'}</button>
                    </div>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
