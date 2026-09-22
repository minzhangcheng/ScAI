import { render, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ChatMarkdown } from './ChatMarkdown'

describe('ChatMarkdown', () => {
  it('renders GFM tables, emphasis and scientific notation in the same message', () => {
    const content = String.raw`| Substate | Composition | Mean score |
| :--- | :--- | ---: |
| **Baseline** | 1,358 / 168 | -0.012 |
| Stress | 6 / 2,731 | 0.038 |

$\text{x}_\text{c}^-$

(mean: $0.038$, $p = 0.32$; $H = 751.4, p = 6.8 \times 10^{-164}$). &#x20;`
    const { container, getByRole } = render(<ChatMarkdown content={content} />)
    const table = getByRole('table')
    expect(within(table).getAllByRole('columnheader')).toHaveLength(3)
    expect(within(table).getAllByRole('row')).toHaveLength(3)
    expect(within(table).getByRole('cell', { name: 'Baseline' }).querySelector('strong')).not.toBeNull()
    expect(within(table).getByRole('columnheader', { name: 'Mean score' })).toHaveStyle({ textAlign: 'right' })
    expect(table.parentElement).toHaveClass('overflow-x-auto')
    expect(container.querySelectorAll('math')).toHaveLength(4)
    expect(container.querySelector('msubsup')).not.toBeNull()
    expect(container.querySelector('msup')).not.toBeNull()
    expect(container.querySelector('.katex-error')).toBeNull()
    expect(container.textContent).not.toContain('&#x20;')
  })

  it('preserves literal code while rendering lists, headings and block math', () => {
    const content = '# Summary\n\n- **First**\n- Second\n\n> Note\n\n' +
      '`$x$`\n\n```text\n| literal | $y$ |\n```\n\n$$\n\\frac{x}{y}\n$$'
    const { container, getByRole } = render(<ChatMarkdown content={content} />)
    expect(getByRole('heading', { level: 1 })).toHaveTextContent('Summary')
    expect(within(getByRole('list')).getAllByRole('listitem')).toHaveLength(2)
    expect(container.querySelector('blockquote')).toHaveTextContent('Note')
    expect(container.querySelector('code')).toHaveTextContent('$x$')
    expect(container.querySelector('pre code')).toHaveTextContent('| literal | $y$ |')
    expect(container.querySelectorAll('math')).toHaveLength(1)
    expect(container.querySelector('math[display="block"] mfrac')).not.toBeNull()
  })
})
