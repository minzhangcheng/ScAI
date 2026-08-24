import { Language } from '../../i18n/translations'
import { useTranslation } from '../../i18n/translations'
import { CUSTOM_TYPE_SENTINEL, isCustomDataType } from '../../utils/dataTypes'

interface CustomTypeInputProps {
  /** The current raw dataType value (may be CUSTOM_TYPE_SENTINEL or a custom string) */
  dataType: string
  /** The predefined type options for this agent */
  typeOptions: readonly string[]
  /** Callback when the custom type value changes */
  onCustomChange: (value: string) => void
  /** Current UI language for placeholder localization */
  language: Language
}

export function CustomTypeInput({ dataType, typeOptions, onCustomChange, language }: CustomTypeInputProps) {
  if (!isCustomDataType(dataType, typeOptions)) return null

  const { t } = useTranslation(language)
  const inputValue = dataType === CUSTOM_TYPE_SENTINEL ? '' : dataType

  return (
    <input
      type="text"
      value={inputValue}
      onChange={(e) => {
        const val = e.target.value
        onCustomChange(val || CUSTOM_TYPE_SENTINEL)
      }}
      placeholder={t('customTypePlaceholder')}
      className="w-[90px] px-1 py-0.5 border border-slate-200 dark:border-slate-700 rounded text-[11px] bg-white dark:bg-surface-900"
    />
  )
}
