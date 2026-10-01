import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import LlmTimeoutDialog from '../LlmTimeoutDialog'

function props(language: 'en' | 'zh', failureKind?: string) {
  return {
    open: true,
    llmFailureInfo: {
      agent: 'bioinformatics', model: 'test-model', llmServer: 'test-server',
      error: 'Analysis source preparation is missing or no longer exact',
      nodeName: 'bioinformatics_analysis', generation: 1, failureKind,
    },
    timeRemaining: 0, hasCountdown: false, language,
    onRetry: vi.fn(), onSkip: vi.fn(), onTerminate: vi.fn(),
    onModifyAgent: vi.fn(), onClose: vi.fn(),
  }
}

describe('Annotation failure intervention', () => {
  it.each(['en', 'zh'] as const)('shows tool recovery actions in %s', (language) => {
    const input = props(language, 'annotation_tool_failure')
    render(<LlmTimeoutDialog {...input} />)
    const labels = language === 'en'
      ? ['Cell-type annotation failed', 'Retry', 'Skip', 'Terminate Analysis', 'Error Details']
      : ['细胞类型注释失败', '重试', '跳过', '终止分析', '错误详情']
    expect(screen.getByRole('dialog', { name: labels[0] })).toBeTruthy()
    expect(screen.queryByText('test-server')).toBeNull()
    expect(screen.queryByText('test-model')).toBeNull()
    expect(screen.queryByRole('button', { name: /Modify Agent|修改 Agent/ })).toBeNull()
    for (const [index, callback] of [input.onRetry, input.onSkip, input.onTerminate].entries()) {
      fireEvent.click(screen.getByRole('button', { name: new RegExp(labels[index + 1]) }))
      expect(callback).toHaveBeenCalledOnce()
    }
    fireEvent.click(screen.getByRole('button', { name: labels[4] }))
    expect(screen.getByRole('log').textContent).toBe(input.llmFailureInfo.error)
  })

  it('keeps the upstream dialog for legacy payloads without a failure kind', () => {
    render(<LlmTimeoutDialog {...props('en')} />)
    expect(screen.getByRole('dialog', { name: 'LLM Call Failed' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Modify Agent Settings' })).toBeTruthy()
    expect(screen.getByText('test-server')).toBeTruthy()
  })

  it.each(['en', 'zh'] as const)('classifies invalid model output in %s', (language) => {
    const input = props(language, 'parse_error')
    input.llmFailureInfo.error = 'No complete JSON plan'
    render(<LlmTimeoutDialog {...input} />)
    const title = language === 'en' ? 'Invalid Model Output' : '模型输出格式错误'
    expect(screen.getByRole('dialog', { name: title })).toBeTruthy()
    expect(screen.getByText(language === 'en'
      ? 'After closing, click the "Invalid Model Output" badge to reopen'
      : '关闭后，可点击“模型输出格式错误”标签重新打开')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Modify Agent|修改 Agent/ })).toBeTruthy()
    expect(screen.getByText('test-model')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Error Details|错误详情/ }))
    expect(screen.getByRole('log').textContent).toBe('No complete JSON plan')
  })
})
