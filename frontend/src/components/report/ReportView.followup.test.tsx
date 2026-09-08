import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ReportView } from './ReportView'
import { useReportController } from './useReportController'
import { useAppStore, type Session } from '../../stores/appStore'

function session(id: string, restoreBranchId: string): Session {
  return {
    id,
    name: id,
    status: 'completed',
    currentRound: 1,
    createdAt: '2026-08-03T00:00:00Z',
    instruction: '',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
    restoreBranchId,
  }
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

function ReportHarness() {
  return <ReportView controller={useReportController()} />
}

describe('Report follow-up handoff', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url.includes('/detailed-report')) {
        return Promise.resolve(jsonResponse({
          report: {
            id: 'report-a',
            session_id: 'A',
            branch_id: 'branch-a',
            title: 'Report A',
            abstract: '',
            sections: [],
            conclusions: '',
            figures: [],
            tables: [],
            generated_at: '2026-08-03T00:00:00Z',
            status: 'completed',
            progress: 100,
            current_step: 'done',
            followups: [{
              followup_id: 'fu-a',
              question: 'Compare the selected direction across conditions.',
              disposition: 'needs_new_data',
              suggested_instruction: 'Use the newly uploaded condition data.',
            }],
          },
        }))
      }
      if (url.includes('/report/preview')) {
        return Promise.resolve(jsonResponse({ has_html: false }))
      }
      if (url.includes('/report/archives')) {
        return Promise.resolve(jsonResponse([]))
      }
      return Promise.resolve(jsonResponse({}))
    }))
    useAppStore.setState({
      currentSession: session('A', 'branch-a'),
      activeTab: 'data',
      pendingChatFollowup: null,
      reportGenerated: false,
    })
  })

  it('hands a needs-new-data instruction to the current Chat branch without posting', async () => {
    render(<ReportHarness />)
    await waitFor(() => expect(screen.getByText('Needs new data')).toBeTruthy())

    fireEvent.click(screen.getByRole('button', { name: 'Continue in Chat' }))

    expect(useAppStore.getState().activeTab).toBe('chat')
    expect(useAppStore.getState().pendingChatFollowup).toEqual({
      sessionId: 'A',
      branchId: 'branch-a',
      followupId: 'fu-a',
      instruction: 'Use the newly uploaded condition data.',
    })
    expect(fetch).toHaveBeenCalledTimes(3)
  })
})
