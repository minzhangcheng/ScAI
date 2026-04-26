export interface RoundWaitOption {
  value: number
  labelZh: string
  labelEn: string
}

export const ROUND_WAIT_OPTIONS: RoundWaitOption[] = [
  { value: 0,    labelZh: '不等待',   labelEn: 'No wait' },
  { value: 300,  labelZh: '5 分钟',  labelEn: '5 minutes' },
  { value: 900,  labelZh: '15 分钟', labelEn: '15 minutes' },
  { value: 3600, labelZh: '60 分钟', labelEn: '60 minutes' },
  { value: -1,   labelZh: '手动跳过', labelEn: 'Manual skip' },
]