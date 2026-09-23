import { render, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { RoundSummary } from '../../../types'
import { AnalysisStatusView } from '../AnalysisStatusView'
import type { AnalysisPanelController } from '../useAnalysisPanelController'

function controller(): AnalysisPanelController {
  return {
    agents: [],
    currentSession: { id: 'session-a', status: 'completed' },
    expandedRound: 'integration-6',
    getAgentStatus: () => 'done',
    language: 'en',
    roundSummaries: [],
    setExpandedRound: vi.fn(),
    t: (key: string) => key,
    unresolvedErrors: [],
  } as unknown as AnalysisPanelController
}

function roundSummary(): RoundSummary {
  return {
    round: 6,
    resultIdentityKey: 'integration-6',
    phase: 'integration',
    status: 'completed',
    summary: 'Recorded round summary',
    findings: [],
    dataRequests: { pathology: [], biochemistry: [], other: [] },
    integrationCandidates: [],
  }
}

const report = String.raw`## Executive Summary

**Copper** ($\text{Cu}$) and $p < 0.001$.

| Program | Score |
| :--- | ---: |
| Stress | 0.038 |

- First observation
- Second observation

` + '```text\nControl   -> DKO\n| literal | $x$ |\n```'

describe('Analysis page Markdown', () => {
  it('renders final-summary tables, formulas and headings while preserving code', () => {
    const state = controller()
    state.currentSession!.finalSummary = report
    const { container, getByRole } = render(<AnalysisStatusView controller={state} />)

    expect(getByRole('heading', { level: 2, name: 'Executive Summary' })).toBeInTheDocument()
    const table = getByRole('table')
    expect(within(table).getAllByRole('row')).toHaveLength(2)
    expect(within(table).getByRole('columnheader', { name: 'Score' })).toHaveStyle({ textAlign: 'right' })
    expect(table.parentElement).toHaveClass('overflow-x-auto')
    expect(container.querySelectorAll('math')).toHaveLength(2)
    expect(container.querySelector('pre code')?.textContent).toBe('Control   -> DKO\n| literal | $x$ |\n')
    expect(within(getByRole('list')).getAllByRole('listitem')).toHaveLength(2)
  })

  it.each(['summary', 'integrationReportSummary'] as const)(
    'renders Markdown in the expanded %s',
    (field) => {
      const state = controller()
      const round = roundSummary()
      round[field] = report
      state.roundSummaries = [round]
      const { container, getByRole } = render(<AnalysisStatusView controller={state} />)

      expect(getByRole('heading', { level: 2, name: 'Executive Summary' })).toBeInTheDocument()
      expect(getByRole('table')).toBeInTheDocument()
      expect(container.querySelector('strong')).toHaveTextContent('Copper')
      expect(container.querySelectorAll('math')).toHaveLength(2)
      expect(container.querySelector('pre code')).toHaveTextContent('| literal | $x$ |')
    },
  )

  it('renders findings and data requests without displaying formatting markers', () => {
    const state = controller()
    const round = roundSummary()
    round.findings = ['**Stress response** with $p < 0.001$']
    round.dataRequests = {
      pathology: ['**Histology**'],
      biochemistry: ['**Activity assay**'],
      other: ['[Protocol](https://example.org/protocol)'],
    }
    state.roundSummaries = [round]
    const { container, getByRole } = render(<AnalysisStatusView controller={state} />)

    expect(Array.from(container.querySelectorAll('strong')).map(node => node.textContent))
      .toEqual(['Stress response', 'Histology', 'Activity assay'])
    expect(container.querySelector('math')).not.toBeNull()
    expect(getByRole('link', { name: 'Protocol' })).toHaveAttribute('href', 'https://example.org/protocol')
  })
})
