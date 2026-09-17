import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { IntegratedReport } from './IntegratedReport'

const staticPath = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../backend/static')
const katexBundle = readFileSync(resolve(staticPath, 'vendor/katex/katex.min.js'), 'utf8')
const offlineRender = readFileSync(resolve(staticPath, 'report_math.js'), 'utf8')

const notation = String.raw`bulk RNA sequencing ($n = 3$ Control vs $n = 3$ DKO);
NADH/NAD$^+$; acid $\beta$-oxidation;
(*ZnT1$^{\text{flox/flox}}$ CTR1$^{\text{flox/flox}}$ Alb-Cre$^+$*)
and (*Alb-Cre$^-$*).`

describe('report math', () => {
  it('renders the reported notation inside ordinary text and emphasis', () => {
    const { container } = render(<IntegratedReport report={notation} />)
    expect(container.querySelectorAll('math')).toHaveLength(8)
    expect(container.querySelectorAll('msup')).toHaveLength(5)
    expect(container.querySelector('mi')?.textContent).toBe('n')
    expect(container.querySelectorAll('em math')).toHaveLength(4)
    expect(container.querySelector('.katex-error')).toBeNull()
  })

  it('preserves code and renders subscripts and block equations', () => {
    const source = '`$x$`\n\n```python\nvalue = "$y$"\n```\n\n' +
      String.raw`$x_i + y_j$` + '\n\n$$\n\\frac{x}{y}\n$$'
    const { container } = render(<IntegratedReport report={source} />)
    expect(container.querySelectorAll('math')).toHaveLength(2)
    expect(container.querySelectorAll('msub')).toHaveLength(2)
    expect(container.querySelector('math[display="block"] mfrac')).not.toBeNull()
    expect(container.querySelector('code')?.textContent).toBe('$x$')
    expect(container.querySelector('pre code')?.textContent).toContain('$y$')
  })

  it('renders the offline HTML payload with the same pinned engine', () => {
    const require = createRequire(import.meta.url)
    expect(katexBundle).toBe(readFileSync(require.resolve('katex/dist/katex.min.js'), 'utf8'))
    const root = document.createElement('div')
    const formulas = ['n = 3', '^+', String.raw`\beta`, String.raw`^{\text{flox/flox}}`,
      String.raw`\frac{x_1}{y_2}`, String.raw`\notACommand{`]
    formulas.forEach((tex, index) => {
      const element = document.createElement('span')
      element.setAttribute('data-scai-math', tex)
      element.setAttribute('data-display', String(index === 4))
      element.textContent = `$${tex}$`
      root.append(element)
    })
    document.body.append(root)
    try {
      // Execute precisely the two scripts embedded by the Python exporter.
      const execute = new Function('document',
        'var module, exports, define; var self = {};\n' + katexBundle +
        '\nvar katex = self.katex;\n' + offlineRender)
      execute(document)
      expect(root.querySelectorAll('math')).toHaveLength(5)
      expect(root.querySelectorAll('msup')).toHaveLength(2)
      expect(root.querySelector('math[display="block"] mfrac')).not.toBeNull()
      expect(root.querySelector('.katex-error')?.textContent).toBe(formulas[5])
      expect(root.querySelector('mi')?.textContent).toBe('n')
    } finally {
      root.remove()
    }
  })

  it('does not allow TeX links or HTML injection', () => {
    const { container } = render(<IntegratedReport report={String.raw`$\href{javascript:alert(1)}{click}$ $\text{<img src=x onerror=alert(1)>}$`} />)
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })
})
