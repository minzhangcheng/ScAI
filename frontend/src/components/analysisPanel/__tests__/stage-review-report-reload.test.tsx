import { renderHook } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { useAnalysisPanelEffects } from '../useAnalysisPanelEffects'

it('uses the loaded stage report URL instead of a historical message and clears it on failure', () => {
  const setExplorationReportData = vi.fn()
  const options = {
    currentSession: { id: 's', status: 'waiting_input', currentRound: 1,
      explorationReportUrl: '/api/session/s/stage-review-report/v2/html' },
    messages: [{ id: 'gate', type: 'exploration_report', content: {
      html_url: '/api/session/s/stage-review-report/v1/html', report_a: {},
    } }],
    roundSummaries: [], roundSupplementaryPendingRef: { current: false },
    lastExplorationMsgRef: { current: '' }, showExplorationDialog: true,
    setExplorationReportData, setIsSupplementaryRunning: vi.fn(),
    setShowIntegrationGateDialog: vi.fn(), setIntegrationGateData: vi.fn(),
  } as unknown as Parameters<typeof useAnalysisPanelEffects>[0]
  const { rerender } = renderHook(({ currentSession }) => useAnalysisPanelEffects({
    ...options, currentSession,
  }), { initialProps: { currentSession: options.currentSession } })
  expect(setExplorationReportData).toHaveBeenLastCalledWith(expect.objectContaining({
    html_url: '/api/session/s/stage-review-report/v2/html',
  }))
  rerender({ currentSession: { ...options.currentSession!, explorationReportUrl: '' } })
  expect(setExplorationReportData).toHaveBeenLastCalledWith(expect.objectContaining({ html_url: '' }))
})
