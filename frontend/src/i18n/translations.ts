// Internationalization translations

export type Language = 'zh' | 'en'


import { zh } from './locales/zh'
import { en } from './locales/en'

export const translations = { zh, en } as const

export type TranslationKey = keyof typeof translations.zh

export function useTranslation(lang: Language) {
  const t = (key: TranslationKey, params?: Record<string, string | number>): string => {
    let text: string = translations[lang][key] || translations.zh[key] || key
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        text = text.replace(`{${k}}`, String(v))
      })
    }
    return text
  }
  return { t, lang }
}

// Re-exported so existing importers keep resolving these from this module.
export { DEFAULT_ENGLISH_PROMPTS, getDefaultPrompts } from './prompts'
