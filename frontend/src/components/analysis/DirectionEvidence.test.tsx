import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import DirectionEvidence from './DirectionEvidence'

const baseEvidence = {
  direction_id: 'd1',
  direction: 'Direction',
  status: 'completed',
  search_tier: 'standard',
  literature_novelty: 'unknown',
  confidence: 'low',
}

describe('DirectionEvidence completion badge', () => {
  it('describes new-policy completion as search workflow completion', () => {
    render(
      <DirectionEvidence
        candidate={{ direction_id: 'd1', direction: 'Direction' }}
        evidence={{ ...baseEvidence, policy_version: 'direction_search_v1' }}
        expanded={false}
        onToggle={vi.fn()}
        language="en"
      />,
    )

    expect(screen.getByText('Search workflow completed')).toBeInTheDocument()
    expect(screen.queryByText('Evidence verified')).not.toBeInTheDocument()
  })

  it('preserves the legacy evidence-verification label', () => {
    render(
      <DirectionEvidence
        candidate={{ direction_id: 'd1', direction: 'Direction' }}
        evidence={baseEvidence}
        expanded={false}
        onToggle={vi.fn()}
        language="en"
      />,
    )

    expect(screen.getByText('Evidence verified')).toBeInTheDocument()
  })
})
