import type { Dispatch, SetStateAction } from 'react'

import type { TranslationKey } from '../../i18n/translations'
import type { Message } from '../../stores/appStore'

export interface ResultContent {
  findings?: string[]
  summary?: string
  message?: string
  output?: string
  text?: unknown
}

export interface CompletedContent {
  status?: string
  final_summary?: string
}

export interface FigureContent {
  approved?: boolean
  revision_requested?: boolean
  figure_name?: string
  feedback?: string
  png_base64?: string
}

export interface ReportContent {
  report?: string
  figures?: unknown[]
}

export interface MessageRenderContext {
  message: Message
  language: 'zh' | 'en'
  expanded: boolean
  setExpanded: Dispatch<SetStateAction<boolean>>
  isDark: boolean
  bgColor: string
  t: (key: TranslationKey, params?: Record<string, string | number>) => string
}
