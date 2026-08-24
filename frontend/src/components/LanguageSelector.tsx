import { useAppStore } from '../stores/appStore'
import { Globe } from 'lucide-react'

export function LanguageSelector() {
  const { language, setLanguage } = useAppStore()

  const toggleLanguage = () => {
    setLanguage(language === 'zh' ? 'en' : 'zh')
  }

  return (
    <button
      onClick={toggleLanguage}
      className="flex items-center gap-2 px-3 py-2 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
      title={language === 'zh' ? 'Switch to English' : '切换到中文'}
    >
      <Globe className="w-4 h-4" />
      <span className="font-medium">{language === 'zh' ? 'EN' : '中'}</span>
    </button>
  )
}
