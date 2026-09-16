import { render } from '@testing-library/react'
import { Users } from 'lucide-react'
import { describe, expect, it, vi } from 'vitest'

import type { Session } from '../../../stores/appStore'
import { AnalysisStatusView } from '../AnalysisStatusView'
import type { AnalysisPanelController } from '../useAnalysisPanelController'

function session(status: Session['status']): Session {
  return {
    id: 'session-a',
    name: 'Session A',
    status,
    currentRound: 2,
    createdAt: '2026-09-16T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }
}

function controller(status: Session['status']): AnalysisPanelController {
  return {
    agents: [{ id: 'pi', name: 'PI Agent', icon: Users, color: 'violet' }],
    currentSession: session(status),
    expandedRound: null,
    getAgentStatus: () => 'working',
    language: 'en',
    roundSummaries: [],
    setExpandedRound: vi.fn(),
    t: (key: string) => key,
    unresolvedErrors: [],
  } as unknown as AnalysisPanelController
}

describe('AnalysisStatusView agent activity', () => {
  it('replaces a working spinner with a pause icon while the session is paused', () => {
    const { container } = render(<AnalysisStatusView controller={controller('paused')} />)

    expect(container.querySelector('.animate-spin')).toBeNull()
    expect(container.querySelector('.lucide-pause')).toBeInTheDocument()
  })

  it('keeps the working spinner while the session is running', () => {
    const { container } = render(<AnalysisStatusView controller={controller('running')} />)

    expect(container.querySelector('.animate-spin')).toBeInTheDocument()
    expect(container.querySelector('.lucide-pause')).toBeNull()
  })
})
