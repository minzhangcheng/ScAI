import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Settings, Key, LogOut, ChevronDown, User } from 'lucide-react'
import { useAppStore } from '../stores/appStore'
import { SettingsDialog } from './SettingsDialog'
import { useClickOutside } from '../hooks/useClickOutside'

export function UserMenu() {
  const { authUser, logout, language } = useAppStore()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

  const isReadOnly = authUser?.status === 'readonly'
  const zh = language === 'zh'

  useClickOutside([menuRef, buttonRef], () => setOpen(false), { enabled: open })

  const handleSignOut = () => {
    setOpen(false)
    logout()
    navigate('/login')
  }

  const handleSettings = () => {
    setOpen(false)
    setShowSettings(true)
  }

  const handleApiKeys = () => {
    setOpen(false)
    navigate('/api-keys')
  }

  if (!authUser) return null

  return (
    <>
      <div className="relative" ref={menuRef}>
        <button
          ref={buttonRef}
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 px-3 py-2 text-sm text-surface-600 hover:bg-surface-100 rounded-lg transition-colors"
          title={authUser.email}
        >
          <div className="w-7 h-7 rounded-full bg-accent-100 flex items-center justify-center">
            <User className="w-4 h-4 text-accent-600" />
          </div>
          <span className="hidden sm:inline text-xs font-medium truncate flex-1 min-w-0">{authUser.email}</span>
          <ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>

        {open && (
          <div className="absolute bottom-full right-0 mb-1 w-48 bg-white rounded-lg shadow-lg border border-surface-200 py-1 z-50">
            <div className="px-3 py-2 border-b border-surface-100">
              <p className="text-xs text-surface-500 truncate">{authUser.email}</p>
              {authUser.role && (
                <p className="text-[10px] text-surface-400 uppercase tracking-wide mt-0.5">{authUser.role}</p>
              )}
            </div>

            <button
              onClick={handleSettings}
              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-surface-600 hover:bg-surface-50 transition-colors"
            >
              <Settings className="w-4 h-4" />
              {zh ? '设置' : 'Settings'}
            </button>

            {!isReadOnly && (
              <button
                onClick={handleApiKeys}
                className="w-full flex items-center gap-2 px-4 py-2 text-sm text-surface-600 hover:bg-surface-50 transition-colors"
              >
                <Key className="w-4 h-4" />
                API Keys
              </button>
            )}

            <div className="border-t border-surface-100 my-1" />

            <button
              onClick={handleSignOut}
              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              {zh ? '退出登录' : 'Sign Out'}
            </button>
          </div>
        )}
      </div>

      <SettingsDialog open={showSettings} onClose={() => setShowSettings(false)} />
    </>
  )
}
