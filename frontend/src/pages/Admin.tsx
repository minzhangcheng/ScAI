import { Key, LogOut, User, UserPlus } from 'lucide-react'

import { AdminDialogs } from './admin/AdminDialogs'
import { AdminTable } from './admin/AdminTable'
import { useAdminController } from './admin/useAdminController'

export function Admin() {
  const controller = useAdminController()
  const {
    error, handleSignOut, navigate, setShowCreateUser, setShowPasswordPrompt, zh,
  } = controller
  return (
    <div className="min-h-screen bg-surface-50 dark:bg-surface-950">
      <header className="bg-white dark:bg-surface-900 border-b border-surface-200 dark:border-surface-700 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent-500 to-accent-700 flex items-center justify-center">
              <User className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-display font-semibold text-surface-900 dark:text-surface-100">{zh ? '管理员面板' : 'Admin Panel'}</h1>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => navigate('/')} className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-surface-600 dark:text-surface-300 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg transition-colors">{zh ? '返回' : 'Back'}</button>
            <button onClick={() => setShowPasswordPrompt(true)} className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-surface-600 dark:text-surface-300 hover:bg-surface-100 dark:hover:bg-surface-800 rounded-lg transition-colors">
              <Key className="w-4 h-4" />{zh ? '修改我的密码' : 'Change My Password'}
            </button>
            <button onClick={handleSignOut} className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 rounded-lg transition-colors">
              <LogOut className="w-4 h-4" />{zh ? '退出登录' : 'Sign Out'}
            </button>
          </div>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-surface-800 dark:text-surface-100">{zh ? '用户管理' : 'User Management'}</h2>
          <button onClick={() => setShowCreateUser(true)} className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-accent-600 hover:bg-accent-700 rounded-lg transition-colors">
            <UserPlus className="w-4 h-4" />{zh ? '创建用户' : 'Create User'}
          </button>
        </div>
        {error && <div className="mb-4 p-4 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 rounded-xl text-red-700 dark:text-red-300 text-sm">{error}</div>}
        <AdminTable controller={controller} />
      </main>
      <AdminDialogs controller={controller} />
    </div>
  )
}
